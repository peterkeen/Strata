#!/usr/bin/env python3
"""Operator-run HTTP incremental-KV gate; no launch, config writes or deployment.

  python3 tools/incremental_kv_http_smoke.py --run --url http://127.0.0.1:5802 \
      --config /path/to/ACTUAL-server-config.json --output /tmp/inc-http-UNIQUE.json

The primary provisions a controlled private or production-settings endpoint.
This client requires the ORIGINAL two-slot 262144 backing / 32768 residency /
4096 MiB optional parking / 2048 MiB reserve config, including GPU vision.
It never changes expert, vision, MTP or prefill settings. /props + /metrics
cross-check actual capacities, incremental INFO, and vision presence. Reserve
and vision GPU placement are config attestations (not live CUDA telemetry).

Two DISTINCT synthetic prompts are sized by /v1/messages/count_tokens, targeting
146000 + 22000 tokens, and bounded by aggregate prompt + modest headroom. BOTH
streaming payloads OMIT output limits: no max_tokens/max_completion_tokens or
server n_predict override. Read a bounded prefix from each, HOLD both connections
open until sampled slots show both matching large prompts decoding with positive
output, then close BOTH. Never wait for 100K outputs. A bounded healthy request
then must match its pre-cancellation greedy response, followed by another fresh
healthy response. Other production clients need not be idle; samples must match
our distinctive prompt sizes, not just any two occupied slots.

Default is dry-run, even when --url is provided. --run authorizes traffic, not
server reconfiguration. Absolute stage deadlines, bounded connect/read/cleanup,
forceful socket shutdown on timeout, daemon workers (no executor shutdown hang).
Evidence files are exclusively created; failure JSON retains partial SSE data,
requests, samples and thread cleanup. This is not a throughput/VRAM stress claim.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from urllib.parse import urlsplit
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incremental_kv_smoke as native

require = native.require
LIMIT_KEYS = {'max_tokens', 'max_completion_tokens', 'max_output_tokens', 'n_predict', 'max_new_tokens'}


def payload(model, text, stream=True, cap=None):
    result = {'model': model, 'stream': stream, 'temperature': 0, 'seed': native.SEED,
              'chat_template_kwargs': {'enable_thinking': False},
              'messages': [{'role': 'user', 'content': text}]}
    if cap is not None:
        result['max_tokens'] = cap
    return result


def assert_omitted(body):
    require(not LIMIT_KEYS.intersection(body), 'large streaming request has an artificial output limit')
    require(body.get('stream') is True, 'large request must stream')


def parse_sse_data(line):
    """One data-only OpenAI JSON line; comments/blank lines do not count as output."""
    if isinstance(line, bytes):
        line = line.decode('utf-8', errors='strict')
    require(len(line) <= 1024 * 1024, 'oversized SSE line')
    if not line.startswith('data:'):
        return {'kind': 'ignore'}
    data = line[5:].strip()
    if data == '[DONE]':
        return {'kind': 'done'}
    value = json.loads(data)
    require(isinstance(value, dict), 'non-object SSE JSON')
    require('error' not in value, f'SSE error: {value}')
    choices = value.get('choices') or []
    if not choices:
        return {'kind': 'ignore'}
    choice = choices[0]
    delta = choice.get('delta') or {}
    text = delta.get('content') or delta.get('reasoning_content') or ''
    require(isinstance(text, str), 'nontext SSE delta')
    return {'kind': 'delta' if text else 'ignore', 'text': text,
            'finish': choice.get('finish_reason'), 'raw': value}


def config_evidence(cfg):
    args = cfg.get('args') or []
    require(cfg.get('parallel') == 2, 'production gate requires parallel=2')
    require('--kv-unified' in args and '--vision' in args, 'production unified KV / native vision disabled')
    for option, value in {'--max-context': '262144', '--kv-resident': '32768',
                          '--conversation-cache-mib': '4096', '--vram-reserve-mib': '2048',
                          '--conversation-cache-slots': '4'}.items():
        require(args.count(option) == 1 and args.index(option) + 1 < len(args) and
                str(args[args.index(option) + 1]) == value, f'production config requires exactly {option} {value}')
    require(not any(option in args for option in ('--batch', '--slots')), 'parallel must control the two server slots')
    vision = cfg.get('vision') or {}
    require(vision.get('gpu') is True and vision.get('exe') and vision.get('mmproj'), 'production GPU vision settings missing')
    return {'parallel': 2, 'context': 262144, 'resident': 32768, 'cache_mib': 4096, 'reserve_mib': 2048,
            'GPU_vision_retained_in_config': True, 'all_model_arguments_retained': True,
            'live_GPU_vision_placement_and_reserve_not_exposed': True,
            'operator_must_supply_actual_server_config': True}


def live_evidence(props, metrics, slots):
    require(props.get('total_slots') == 2 and props.get('kv_unified') is True, 'live endpoint is not two-slot unified KV')
    for key, wanted in {'kv_capacity_cells': 262144, 'kv_resident_capacity_cells': 32768, 'kv_resident': 32768}.items():
        require(props.get(key) == wanted, f'live /props {key} differs')
    require(props.get('modalities', {}).get('vision') is True, 'GPU vision companion must remain enabled')
    require(len(slots) == 2 and all(s.get('n_ctx') == 262144 for s in slots), 'logical slot ceilings reduced')
    info = metrics['engine']
    for key, wanted in {'kv_incremental': 1, 'kv_reserve_ahead': 256, 'conversation_cache_mib': 4096,
                        'conversation_cache_slots': 4, 'context': 262144, 'batch_slots': 2}.items():
        require(int(info.get(key, -1)) == wanted, f'live engine requires {key}={wanted}')
    require(info.get('images') is True, 'live engine reports no vision companion')
    return {'props': props, 'engine_info': info, 'slots': slots,
            'config_only_checks': ['vram_reserve_mib=2048', 'vision.gpu=true'],
            'global_idleness_required': False}


def matching_decode_sample(metrics, slots, prompt_counts):
    rows = metrics.get('live', {}).get('slots') or []
    if len(rows) != 2 or len(slots) != 2 or not all(s.get('is_processing') for s in slots):
        return False
    if not all(r.get('state') == 'decoding' and r.get('generated', 0) > 0 for r in rows):
        return False
    # count_tokens and native prompt count may differ by a small template guard.
    actual = sorted(r.get('prompt_tokens', -1000) for r in rows)
    return all(abs(x - y) <= 16 for x, y in zip(actual, sorted(prompt_counts)))


class Client:
    def __init__(self, url, io_timeout=10):
        parsed = urlsplit(url)
        require(parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username and not parsed.query,
                'use a plain http(s) endpoint without credentials/query')
        self.parsed, self.io_timeout = parsed, io_timeout
        # HTTP/1.0 SSE detaches conn.sock after headers. Keep the original
        # socket to wake the response's makefile reader during cancellation.
        self.connections = {}
        self.deadline_timers = {}
        self.lock = threading.Lock()

    def open(self, path, body, deadline):
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'HTTP absolute deadline exceeded')
        cls = http.client.HTTPSConnection if self.parsed.scheme == 'https' else http.client.HTTPConnection
        conn = cls(self.parsed.hostname, self.parsed.port, timeout=min(self.io_timeout, remaining))
        # A socket timeout alone is NOT absolute: slowly dripping headers or
        # SSE bytes reset individual reads. A watchdog shuts down this socket
        # at the original deadline, including after HTTP/1.0 detaches it.
        timer = threading.Timer(remaining, self.abort, args=(conn,))
        timer.daemon = True
        with self.lock:
            self.connections[conn] = None
            self.deadline_timers[conn] = timer
        timer.start()
        try:
            conn.connect()
            with self.lock:
                self.connections[conn] = conn.sock
            require(time.monotonic() < deadline, 'HTTP connection exceeded absolute deadline')
            conn.sock.settimeout(max(.001, deadline - time.monotonic()))
            data = json.dumps(body).encode() if body is not None else None
            conn.request('POST' if data is not None else 'GET', self.parsed.path.rstrip('/') + path,
                         body=data, headers={'Content-Type': 'application/json'} if data is not None else {})
            response = conn.getresponse()
            require(time.monotonic() < deadline, 'HTTP headers exceeded absolute deadline')
            require(response.status == 200, f'HTTP {response.status} {response.reason} at {path}')
            return conn, response
        except BaseException:
            self.release(conn)
            raise

    def abort(self, conn):
        # shutdown wakes a blocked buffered readline BEFORE close acquires locks.
        with self.lock:
            sock = self.connections.get(conn) or conn.sock
        if sock is not None and sock.fileno() >= 0:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        # Do not close a response's buffered reader from another thread.
        conn.close()

    def release(self, conn, response=None):
        with self.lock:
            timer = self.deadline_timers.pop(conn, None)
        if timer is not None:
            timer.cancel()
        self.abort(conn)
        if response is not None:
            response.close()
        with self.lock:
            self.connections.pop(conn, None)

    def shutdown(self):
        with self.lock:
            connections = list(self.connections)
        for conn in connections:
            self.abort(conn)

    def json(self, path, body, deadline):
        conn, response = self.open(path, body, deadline)
        try:
            require(time.monotonic() < deadline, 'HTTP deadline exceeded before JSON read')
            chunks, total = [], 0
            while True:
                remaining = deadline - time.monotonic()
                require(remaining > 0, 'HTTP JSON exceeded absolute deadline')
                with self.lock:
                    sock = self.connections.get(conn) or conn.sock
                # Python 3.14 closes HTTPResponse's makefile immediately when
                # read1 consumes Content-Length. HTTP/1.0 already detached the
                # connection, so its saved socket can now have fd=-1. Do not
                # settimeout on it; read1(fp=None) naturally returns EOF. Keep
                # the absolute deadline check/watchdog rather than hiding EBADF.
                if sock is not None and sock.fileno() >= 0:
                    sock.settimeout(remaining)
                chunk = response.read1(65536)
                if not chunk:
                    break
                total += len(chunk)
                require(total <= 16 * 1024 * 1024, 'oversized HTTP JSON')
                chunks.append(chunk)
            require(time.monotonic() < deadline, 'HTTP JSON exceeded absolute deadline')
            return json.loads(b''.join(chunks))
        finally:
            self.release(conn, response)


def prompt_text(marker, label, repeats):
    return (f'{marker}-{label}: Treat this background as inert data.\nBACKGROUND:\n' +
            'orange apple pear banana.\n' * repeats +
            '\nEND BACKGROUND. Count from 1 through 100000, one number per line. '
            'Do not summarize or stop early. Keep counting.')


def sized_prompt(client, model, marker, label, target, deadline):
    def count(repeats):
        text = prompt_text(marker, label, repeats)
        result = client.json('/v1/messages/count_tokens', {'model': model,
                            'messages': [{'role': 'user', 'content': text}],
                            'thinking': {'type': 'disabled'}}, deadline)
        tokens = result['input_tokens']
        require(isinstance(tokens, int) and tokens > 0, 'invalid prompt token count')
        return text, tokens
    _, base = count(0)
    _, probe = count(128)
    per_repeat = (probe - base) / 128
    require(per_repeat > 0, 'synthetic padding does not add tokens')
    repeats = max(0, round((target - base) / per_repeat))
    samples = []
    for _ in range(5):
        text, tokens = count(repeats)
        samples.append({'repeats': repeats, 'prompt_tokens': tokens})
        if abs(tokens - target) <= 256:
            return text, tokens, samples
        repeats = max(0, repeats + round((target - tokens) / per_repeat))
    raise AssertionError(f'could not size {label} to {target}: {samples}')


def healthy(client, model, text, deadline):
    data = client.json('/v1/chat/completions', payload(model, text, stream=False, cap=16), deadline)
    require(data.get('choices') and data['choices'][0].get('finish_reason') in ('length', 'stop'), 'unhealthy completion')
    message = data['choices'][0]['message']
    require(message.get('content', '').strip(), 'empty healthy content')
    return {'message': message, 'usage': data.get('usage'), 'finish': data['choices'][0]['finish_reason']}


def stream_worker(client, body, record, prefix, deadline, hold, origin):
    conn = response = None
    try:
        assert_omitted(body)
        conn, response = client.open('/v1/chat/completions', body, deadline)
        require('text/event-stream' in (response.getheader('Content-Type') or ''), 'not an SSE response')
        record['opened_s'] = time.monotonic() - origin
        while len(record['deltas']) < prefix:
            require(time.monotonic() < deadline, 'SSE absolute deadline exceeded')
            line = response.readline(1024 * 1024 + 1)
            require(line, 'SSE EOF before bounded prefix')
            event = parse_sse_data(line)
            if event['kind'] == 'done' or event.get('finish') is not None:
                record['early_finish'] = event
                raise AssertionError('natural finish before bounded prefix/overlap; invalid fixture')
            if event['kind'] == 'delta':
                record['deltas'].append({'wall_s': time.monotonic() - origin, 'text': event['text'], 'raw': event['raw']})
        record['prefix_ready_s'] = time.monotonic() - origin
        # Keep connection OPEN, not artificially cap model generation.
        require(hold.wait(max(0, deadline - time.monotonic())), 'controller never proved overlapping emissions')
    except Exception as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if conn is not None:
            client.release(conn, response)
        record['closed_s'] = time.monotonic() - origin
        record['worker_done'] = True


def run(client, cfg, evidence, stage_timeout, prefix, sample_interval, cleanup_timeout):
    origin = time.monotonic()
    evidence['config_checks'] = config_evidence(cfg)
    deadline = time.monotonic() + stage_timeout
    props = client.json('/props', None, deadline)
    metrics = client.json('/metrics', None, deadline)
    slots = client.json('/slots', None, deadline)
    evidence['live_checks'] = live_evidence(props, metrics, slots)
    model = cfg.get('model_name') or props.get('model_alias') or metrics['engine']['model']
    marker = 'incremental-http-' + uuid.uuid4().hex
    evidence['marker'] = marker
    recovery_text = marker + '-HEALTHY: Reply with exactly the word HEALTHY.'
    fresh_text = marker + '-FRESH: Reply with exactly the word FRESH.'
    evidence['healthy_before'] = healthy(client, model, recovery_text, time.monotonic() + stage_timeout)
    evidence['healthy_fresh_before'] = healthy(client, model, fresh_text, time.monotonic() + stage_timeout)
    require(evidence['healthy_before']['message'] != evidence['healthy_fresh_before']['message'],
            'healthy fixtures are indistinguishable; cannot detect inherited replies')
    deadline = time.monotonic() + stage_timeout
    prompt_records = [sized_prompt(client, model, marker, name, count, deadline)
                      for name, count in (('A', 146000), ('B', 22000))]
    bodies = [payload(model, text) for text, _, _ in prompt_records]
    for body in bodies:
        assert_omitted(body)
    evidence['prompt_sizing'] = [{'target': target, 'actual': count, 'samples': samples,
                                  'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
                                 for (text, count, samples), target in zip(prompt_records, (146000, 22000))]
    counts = [count for _, count, _ in prompt_records]
    evidence['capacity_plan'] = native.growth_plan([[0] * count for count in counts],
                                                  [262144 - count - native.GUARD for count in counts], 262144)
    # Actual payloads are retained to audit omission and reproduce exact prompts.
    evidence['large_payloads'] = bodies
    evidence['streams'] = [{'name': name, 'deltas': [], 'output_limits_omitted': True} for name in ('A', 'B')]
    hold = threading.Event()
    deadline = time.monotonic() + stage_timeout
    workers = [threading.Thread(target=stream_worker,
                               args=(client, body, record, prefix, deadline, hold, origin), daemon=True,
                               name=f'incremental-http-{record["name"]}')
               for body, record in zip(bodies, evidence['streams'])]
    proof = False
    try:
        for worker in workers:
            worker.start()
        while time.monotonic() < deadline:
            require(not any(record.get('error') for record in evidence['streams']), 'stream failed; see stream evidence')
            sample_deadline = min(deadline, time.monotonic() + client.io_timeout)
            metrics = client.json('/metrics', None, sample_deadline)
            slots = client.json('/slots', None, sample_deadline)
            sample = {'wall_s': time.monotonic() - origin, 'live': metrics['live'], 'slots': slots,
                      'prefix_lengths': [len(record['deltas']) for record in evidence['streams']]}
            evidence['samples'].append(sample)
            if matching_decode_sample(metrics, slots, counts) and all(n >= prefix for n in sample['prefix_lengths']):
                evidence['overlap_proof'] = sample
                proof = True
                break
            require(not any(record.get('worker_done') for record in evidence['streams']), 'stream closed before overlap proof')
            time.sleep(min(sample_interval, max(0, deadline - time.monotonic())))
        require(proof, 'no sampled two matching large prompts processing + emitting; omitted limits may still serialize')
    finally:
        evidence['close_both_s'] = time.monotonic() - origin
        hold.set()
        if not proof:
            client.shutdown()
        join_deadline = time.monotonic() + cleanup_timeout
        for worker in workers:
            if worker.ident is not None:
                worker.join(max(0, join_deadline - time.monotonic()))
        evidence['cleanup'] = {'workers_stopped': [not w.is_alive() for w in workers]}
        client.shutdown()
    require(all(evidence['cleanup']['workers_stopped']), 'HTTP worker did not stop during bounded cleanup')
    require(not any(record.get('error') for record in evidence['streams']), 'SSE worker failed during cancellation')
    evidence['healthy_after'] = healthy(client, model, recovery_text, time.monotonic() + stage_timeout)
    require(evidence['healthy_before']['message'] == evidence['healthy_after']['message'],
            'post-close healthy answer differs; possible inherited replies/state')
    evidence['healthy_fresh'] = healthy(client, model, fresh_text, time.monotonic() + stage_timeout)
    require(evidence['healthy_fresh']['message'] == evidence['healthy_fresh_before']['message'],
            'fresh post-close request differs from its independent reference; inherited replies/state')
    evidence['after'] = client.json('/metrics', None, time.monotonic() + client.io_timeout)['live']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true', help='explicitly authorize HTTP traffic; otherwise dry-run')
    ap.add_argument('--url', required=True, help='primary-controlled private/production-settings endpoint')
    ap.add_argument('--config', required=True, type=Path, help='ACTUAL endpoint config; read-only')
    ap.add_argument('--output', required=True, type=Path, help='NEW JSON evidence filename')
    ap.add_argument('--prefix', type=int, default=8, help='nonempty SSE deltas from EACH; not model max_tokens')
    ap.add_argument('--stage-timeout', type=float, default=1200)
    ap.add_argument('--io-timeout', type=float, default=10)
    ap.add_argument('--cleanup-timeout', type=float, default=15)
    ap.add_argument('--sample-interval', type=float, default=.1)
    a = ap.parse_args(argv)
    if a.prefix < 2 or a.prefix > 128 or any(not math.isfinite(t) or t <= 0 for t in
            (a.stage_timeout, a.io_timeout, a.cleanup_timeout, a.sample_interval)):
        ap.error('prefix must be 2..128; timeouts/interval must be finite and positive')
    if not a.run:
        print('Dry run: 146K + 22K prompts, omitted limits, concurrent SSE prefix/close/recovery. No network or config changes.')
        return 0
    output = a.output.expanduser().resolve()
    if output.exists():
        ap.error('existing artifact; select a fresh --output')
    with output.open('x', encoding='utf-8') as f:
        f.write('{}\n')
    evidence = {'schema': 1, 'harness': 'incremental-unified-kv-http', 'passed': False,
                'url': a.url, 'config': str(a.config.resolve()), 'samples': [],
                'timeouts': {'stage_s': a.stage_timeout, 'io_s': a.io_timeout, 'cleanup_s': a.cleanup_timeout}}
    client = None
    try:
        raw = a.config.read_bytes()
        cfg = json.loads(raw.decode('utf-8-sig'))
        evidence['config_sha256'] = hashlib.sha256(raw).hexdigest()
        client = Client(a.url, a.io_timeout)
        run(client, cfg, evidence, a.stage_timeout, a.prefix, a.sample_interval, a.cleanup_timeout)
        evidence['passed'] = True
    except (Exception, KeyboardInterrupt) as exc:
        evidence['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if client is not None:
            client.shutdown()
        output.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(f'{"PASS" if evidence["passed"] else "FAIL"}: {output}')
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

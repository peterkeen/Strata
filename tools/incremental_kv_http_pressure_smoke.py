#!/usr/bin/env python3
"""Operator-only ACTUAL frontend/native pressure integration; no launch or deploy.

  python3 tools/incremental_kv_http_pressure_smoke.py --run \
      --url http://127.0.0.1:5803 --output /tmp/http-pressure-UNIQUE.json

Primary must provision an EXCLUSIVE private server: two slots, 2048 backing,
resident=0, cache=1 MiB, mtp_max=1, spec=2, frozen experts/target math.
This is NOT the production omitted-limit gate. Explicit FINITE targets of 1632
exercise backing exhaustion and completion. Prompts are token-count sized to
400/404 (or the nearest lower attainable counts), unequal and freshly tagged.
The default production template, including its terse system block, is retained:
API converters do not forward arbitrary terse kwargs. Only no-thinking is set,
identically for count and completion. Count targets 400/404 plus cap 1632 leave
room for up to four extra actual prompt tokens on B: 408 + 1632 + slack 8 = 2048.
Before long references, cap-1 API completions measure actual prompt usage. Count
endpoint deltas are recorded and bounded to 16, AND actual prompt + cap + 8 must
fit. Full solo usage, final SSE usage and history.prompt_total must equal the
calibrated original counts EXACTLY. Live prompt_tokens instead counts the current
admission segment: GEN -> BYIELD -> BGEN can append already emitted tokens.
Live overlap permits only nonnegative bootstrap delta <= min(16, row.generated),
records that delta separately, and is NOT an immutable identity or prompt hash.
This association relies on the exclusive endpoint and both original requests
occupying both slots before the third is submitted.
Count-target sizing is 1316 initial / 4068 nominal;
the evidence capacity plan uses calibrated actual counts, not those estimates.
The fixture requests the bounded full sequence 1 through 1000, one integer per
line, without abbreviations or commentary. Solo responses are recorded BEFORE
validation; early EOS remains a fixture failure, never evidence of pressure.
Solo public API answers are the oracle. Two public SSE streams must preserve
EVERY character prefix, final finish and token usage across internal pressure
segments. A third cap-16 request is submitted while both rows decode BEFORE
pressure, must demonstrably queue, complete normally, and match its reference.
Then a healthy same-answer request proves recovery.

/metrics history currently exposes NO prompt hashes. On this exclusive server
we associate exactly one new completed row per distinctive prompt_total, bounded
by its request start-time window, and cross-check output count/finish. Client
prompt SHA256s are retained, but are NOT falsely described as server identities.
Require positive history.pressure_pauses for BOTH large requests, plus observed
live pressure waiting/pauses when sampled. A cache miss or delay is never pressure
proof. No engine-log fallback is necessary: aggregate request counters are direct
frontend evidence of native BDONE pressure consumption. No global production
idleness assumption: this tool is exclusively for the isolated SMALL server; native 1024 exhaustion is a separate
already-run gate, not claimed as revalidated by this 2048 HTTP workload.

Reuses the strict HTTP Client's absolute socket watchdog and detached HTTP/1.0
socket shutdown. Fresh JSON, bounded stages, daemon workers, force-close on any
failure, bounded joins. Default is dry-run; --run alone authorizes requests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incremental_kv_http_smoke as common
import incremental_kv_smoke as native
require = native.require
CONTEXT = 2048
PROMPT_COUNTS = (400, 404)
TARGET = 1632
MAX_COUNT_DELTA = 16


def verify_live(props, metrics, slots):
    require(props.get('total_slots') == 2 and props.get('kv_unified') is True, 'requires two unified slots')
    require(props.get('kv_capacity_cells') == CONTEXT and props.get('kv_resident') == 0,
            f'requires SMALL {CONTEXT} / resident 0')
    require(len(slots) == 2 and all(s.get('n_ctx') == CONTEXT for s in slots), 'unexpected slot logical ceilings')
    info = metrics['engine']
    for key, value in {'context': CONTEXT, 'kv_capacity_cells': CONTEXT, 'kv_resident': 0,
                       'kv_incremental': 1, 'kv_reserve_ahead': 256, 'batch_slots': 2,
                       'conversation_cache_mib': 1, 'mtp_max': 1, 'spec': 2, 'lookup': 0}.items():
        require(int(info.get(key, -1)) == value, f'live INFO requires {key}={value}')
    require(float(info.get('pcie_frac', -1)) == 0, 'expert routing must be frozen')
    live = metrics['live']
    require(not live.get('running') and not live.get('waiting') and
            not any(r.get('state') in ('reading', 'decoding') for r in live.get('slots', [])),
            'this pressure gate requires an exclusive idle PRIVATE server')


def sized_prompt(client, model, marker, label, target, deadline):
    def count(n):
        text = (f'{marker}-{label}: Write every integer from 1 through 1000 inclusive, '
                'in ascending order, one integer per line. A program checks the full sequence; '
                'all 1000 integers are mandatory. No omissions, ellipses, ranges, code, markdown, '
                'documentation, summaries or commentary.\nInert padding:' + ' alpha' * n +
                '\nIgnore the padding. Begin the full sequence now:\n')
        result = client.json('/v1/messages/count_tokens', {'model': model,
                            'messages': [{'role': 'user', 'content': text}],
                            'thinking': {'type': 'disabled'}}, deadline)
        tokens = result['input_tokens']
        require(isinstance(tokens, int) and tokens > 0, 'invalid token count')
        return text, tokens
    _, base = count(0)
    _, probe = count(32)
    require(base <= target and probe > base,
            f'count fixture {label}: base={base}, probe={probe}, target={target}; '
            'base too large / padding not growing (default template retained)')
    step = (probe - base) / 32
    n = max(0, int((target - base) / step))
    samples = []
    best = None
    for _ in range(12):
        text, tokens = count(n)
        samples.append({'padding_words': n, 'prompt_tokens': tokens})
        if tokens <= target:
            if best is None or tokens > best[1]:
                best = (text, tokens)
            if target - tokens <= 1:
                break
            n += max(1, int((target - tokens) / step))
        else:
            n = max(0, n - max(1, math.ceil((tokens - target) / step)))
    require(best is not None and target - best[1] <= 4,
            f'count fixture {label}: base={base}, probe={probe}, target={target}; '
            f'could not size below target; samples={samples}')
    return best[0], best[1], samples


def measure_prompt(client, model, text, deadline):
    response = client.json('/v1/chat/completions', common.payload(model, text, stream=False, cap=1), deadline)
    require(response.get('choices') and response['choices'][0].get('finish_reason') in ('length', 'stop'),
            'prompt calibration did not finish normally')
    return {'usage': response.get('usage') or {}, 'finish': response['choices'][0]['finish_reason'],
            'request_max_tokens': 1}


def verify_prompt_identity(counted, actual, target=TARGET):
    require(len(counted) == len(actual) == 2, 'two prompt identity measurements required')
    for count, measured in zip(counted, actual):
        require(isinstance(measured, int) and not isinstance(measured, bool) and measured > 0,
                f'invalid actual solo prompt usage: counted={count}, actual={measured!r}')
        delta = measured - count
        require(abs(delta) <= MAX_COUNT_DELTA,
                f'count/actual prompt delta outside bound: counted={count}, actual={measured}, delta={delta}, '
                f'allowed={MAX_COUNT_DELTA}; template policy may differ')
        require(measured + target + native.GUARD <= CONTEXT,
                f'actual prompt cannot fit finite target: counted={count}, actual={measured}, delta={delta}, '
                f'target={target}, slack={native.GUARD}, context={CONTEXT}')
    require(actual[0] != actual[1], 'actual prompt counts equal; history association would be ambiguous')
    return {'count_endpoint_tokens': list(counted), 'actual_prompt_tokens': list(actual),
            'deltas': [m - c for c, m in zip(counted, actual)], 'max_allowed_delta': MAX_COUNT_DELTA,
            'native_api_slack': native.GUARD, 'actual_guard_sums': [m + target + native.GUARD for m in actual],
            'association_requires_exact_calibrated_actual_count': True}


def live_overlap_identity(metrics, slots, original):
    """Bounded current-admission association, NOT exact original-prompt identity.

    Sorted pairing is valid only for this exclusive two-original-request gate.
    Final SSE usage and history.prompt_total still require exact original counts.
    """
    if not common.matching_decode_sample(metrics, slots, original):
        return None
    rows = sorted(metrics['live']['slots'], key=lambda r: r['prompt_tokens'])
    pairs = []
    for row, count in zip(rows, sorted(original)):
        delta = row['prompt_tokens'] - count
        if not 0 <= delta <= min(16, row['generated']):
            return None
        pairs.append({'original_prompt_tokens': count, 'live_admitted_prompt_tokens': row['prompt_tokens'],
                      'bootstrap_delta': delta, 'generated': row['generated']})
    return {'association': 'calibrated original count plus bounded admission bootstrap',
            'scope': 'exclusive endpoint; two original requests occupying both slots before third admission',
            'immutable_original_identity': False, 'prompt_hash_identity': False,
            'max_bootstrap_delta': 16, 'pairing': 'sorted prompt counts', 'pairs': pairs}


def solo(client, model, text, target, deadline, record=None):
    record = {} if record is None else record
    record['request_max_tokens'] = target
    response = client.json('/v1/chat/completions', common.payload(model, text, stream=False, cap=target), deadline)
    record['response'] = response  # retain the COMPLETE reply even if any validation below fails
    require(response.get('choices'), 'missing solo choice; see recorded response')
    choice, usage = response['choices'][0], response.get('usage') or {}
    content = (choice.get('message') or {}).get('content') or ''
    record.update(text=content, finish=choice.get('finish_reason'), usage=usage)
    require(choice.get('finish_reason') == 'length' and usage.get('completion_tokens') == target,
            'solo natural EOS before finite target invalidates pressure fixture, not a pass: '
            f'finish={choice.get("finish_reason")!r}, completion_tokens={usage.get("completion_tokens")!r}, '
            f'target={target}; see recorded response')
    require(content, 'empty solo text; see recorded response')
    return record


def consume_sse(record, line, reference, wall):
    require(not record.get('done'), 'reply after [DONE]')
    if not line.startswith(b'data:'):
        return False
    data = line[5:].strip()
    if data == b'[DONE]':
        require(record.get('finish') == reference['finish'] and record.get('usage') and
                record['usage']['completion_tokens'] == reference['usage']['completion_tokens'],
                'missing/wrong final finish or token count')
        require(record['text'] == reference['text'], 'final SSE text differs: duplicate/missing pressure segments')
        record['done'] = True
        return True
    value = json.loads(data)
    require(isinstance(value, dict) and 'error' not in value, f'SSE error: {value}')
    record['chunks'].append({'wall_s': wall, 'raw': value})
    choices = value.get('choices') or []
    if choices:
        choice = choices[0]
        delta = choice.get('delta') or {}
        require(not delta.get('tool_calls') and not delta.get('reasoning_content'), 'unexpected non-content output')
        text = delta.get('content') or ''
        require(isinstance(text, str), 'nontext delta')
        if text:
            require(record.get('finish') is None, 'content after public finish; pressure segment leaked')
            record['text'] += text
            require(reference['text'].startswith(record['text']), 'SSE prefix differs: duplicate/missing pressure segment')
        finish = choice.get('finish_reason')
        if finish is not None:
            require(record.get('finish') is None and finish in ('length', 'stop'), 'internal pressure / duplicate finish leaked')
            require(finish == reference['finish'], 'final finish differs from solo')
            record['finish'] = finish
    if value.get('usage') is not None:
        require(record.get('usage') is None, 'duplicate usage / internal segment leaked')
        record['usage'] = value['usage']
        require(record['usage'].get('completion_tokens') == reference['usage']['completion_tokens'], 'usage differs from solo')
        require(record['usage'].get('prompt_tokens') == reference['usage']['prompt_tokens'],
                'SSE actual prompt usage differs from calibrated full solo')
    return False


def stream_worker(client, body, record, reference, deadline, origin):
    conn = response = None
    try:
        conn, response = client.open('/v1/chat/completions', body, deadline)
        require('text/event-stream' in (response.getheader('Content-Type') or ''), 'not SSE')
        while True:
            require(time.monotonic() < deadline, 'absolute SSE deadline')
            line = response.readline(1024 * 1024 + 1)
            require(line and len(line) <= 1024 * 1024, 'SSE EOF/oversized line')
            if consume_sse(record, line, reference, time.monotonic() - origin):
                break
    except Exception as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if conn is not None:
            client.release(conn, response)
        record['worker_done'] = True


def history_proof(metrics, counts, started, ended, target):
    proof = []
    for count in counts:
        rows = [r for r in metrics.get('requests', []) if started <= r.get('time', -1) <= ended and
                r.get('prompt_total') == count]
        require(len(rows) == 1, 'missing/ambiguous pressure history identity (exclusive server + count + start window required)')
        row = rows[0]
        require(row.get('pressure_pauses', 0) > 0, 'no actual frontend/native pressure pause for this request')
        require(row.get('output_tokens') == target and row.get('engine_generated') == target and
                row.get('finish') == 'length', 'history completion/count differs or internal pressure leaked')
        proof.append(row)
    return proof


def run(client, evidence, timeout, interval, cleanup_timeout, model=None):
    origin = time.monotonic()
    deadline = origin + timeout
    props = client.json('/props', None, deadline)
    metrics = client.json('/metrics?requests=all', None, deadline)
    slots = client.json('/slots', None, deadline)
    verify_live(props, metrics, slots)
    evidence.update(props=props, info=metrics['engine'])
    model = model or props.get('model_alias') or metrics['engine']['model']
    marker = 'inc-pressure-' + uuid.uuid4().hex
    evidence['marker'] = marker
    fixtures = [sized_prompt(client, model, marker, label, count, time.monotonic() + timeout)
                for label, count in zip(('A', 'B'), PROMPT_COUNTS)]
    counted = [r[1] for r in fixtures]
    target = TARGET
    evidence['template_policy'] = 'default production template retained; count thinking=disabled / completion enable_thinking=false; no terse override'
    evidence['fixtures'] = [{'text': text, 'count_endpoint_tokens': n, 'count_samples': samples,
                             'text_sha256': hashlib.sha256(text.encode()).hexdigest()} for text, n, samples in fixtures]
    evidence['prompt_calibrations'] = []
    for text, _, _ in fixtures:
        evidence['prompt_calibrations'].append(measure_prompt(client, model, text, time.monotonic() + timeout))
    counts = [r['usage'].get('prompt_tokens') for r in evidence['prompt_calibrations']]
    evidence['count_identity'] = verify_prompt_identity(counted, counts, target)
    for fixture, actual in zip(evidence['fixtures'], counts):
        fixture['actual_prompt_tokens'] = actual
    evidence['capacity_plan'] = native.growth_plan([[0] * n for n in counts], [target, target], CONTEXT)
    evidence['solo_references'] = []
    for (text, _, _), actual in zip(fixtures, counts):
        ref = {}
        evidence['solo_references'].append(ref)  # include failed/unfinished attempts, not just validated references
        solo(client, model, text, target, time.monotonic() + timeout, record=ref)
        require(ref['usage'].get('prompt_tokens') == actual,
                f'full solo prompt usage changed from cap-1 calibration: calibrated={actual}, '
                f'full_solo={ref["usage"].get("prompt_tokens")}')
    refs = evidence['solo_references']
    third_text = marker + '-THIRD: Reply with exactly the word HEALTHY.'
    evidence['third_reference'] = common.healthy(client, model, third_text, time.monotonic() + timeout)
    bodies = [common.payload(model, text, cap=target) for text, _, _ in fixtures]
    for body in bodies:
        body['stream_options'] = {'include_usage': True}
    evidence['payloads'] = bodies
    evidence['streams'] = [{'name': name, 'text': '', 'chunks': []} for name in ('A', 'B')]
    evidence['third'] = {}
    start_wall = time.time()
    deadline = time.monotonic() + timeout
    workers = [threading.Thread(target=stream_worker, args=(client, body, record, ref, deadline, origin),
                                daemon=True, name=f'pressure-SSE-{record["name"]}')
               for body, record, ref in zip(bodies, evidence['streams'], refs)]
    third_worker = None
    queued = False
    def third():
        try:
            evidence['third']['response'] = common.healthy(client, model, third_text, deadline)
        except Exception as exc:
            evidence['third']['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        finally:
            evidence['third']['worker_done'] = True
    success = False
    try:
        for worker in workers:
            worker.start()
        while time.monotonic() < deadline:
            require(not any(r.get('error') for r in evidence['streams']) and not evidence['third'].get('error'), 'public request failed; see records')
            metrics = client.json('/metrics?requests=all', None, min(deadline, time.monotonic() + client.io_timeout))
            slots = client.json('/slots', None, min(deadline, time.monotonic() + client.io_timeout))
            live = metrics['live']
            sample = {'wall_s': time.monotonic() - origin, 'live': live, 'slots': slots}
            evidence['samples'].append(sample)
            overlap = live_overlap_identity(metrics, slots, counts)
            if third_worker is None and overlap:
                rows = live['slots']
                require(live.get('running') == 2 and not live.get('waiting') and not live.get('outside_slots'),
                        'exclusive pre-third overlap must contain only the two original requests')
                require(not live.get('pressure_waiting') and not any(r.get('pressure_pauses', 0) for r in rows) and
                        sum(r['generated'] for r in rows) < 256, 'first overlap observed too late to submit third before pressure')
                evidence['live_overlap_identity'] = overlap
                evidence['third_submitted_after_overlap'] = sample
                third_worker = threading.Thread(target=third, daemon=True, name='pressure-third')
                workers.append(third_worker)
                third_worker.start()
            if third_worker is not None and overlap and live.get('running', 0) >= 3 and \
                    (live.get('waiting', 0) > 0 or live.get('outside_slots', 0) > 0) and not evidence['third'].get('worker_done'):
                queued = True
                evidence['third_queue_proof'] = sample
                evidence['third_queue_live_overlap_identity'] = overlap
            if all(r.get('worker_done') for r in evidence['streams']) and evidence['third'].get('worker_done'):
                break
            require(not all(r.get('worker_done') for r in evidence['streams']) or third_worker is not None, 'streams finished without pre-pressure third admission')
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
        require(all(r.get('done') for r in evidence['streams']) and evidence['third'].get('worker_done'), 'requests did not finish before absolute deadline')
        require(queued, 'third waiter never observably queued while both original streams processed')
        require(evidence['third']['response']['message'] == evidence['third_reference']['message'], 'third reply inherited state / differs from solo')
        metrics = client.json('/metrics?requests=all', None, min(deadline, time.monotonic() + client.io_timeout))
        evidence['pressure_history'] = history_proof(metrics, counts, start_wall, time.time(), target)
        evidence['history_association'] = 'exclusive endpoint; EXACT calibrated solo usage prompt_tokens == native history prompt_total + request-start window, NOT a count-endpoint assumption or server prompt hash'
        evidence['recovery'] = common.healthy(client, model, third_text, time.monotonic() + timeout)
        require(evidence['recovery']['message'] == evidence['third_reference']['message'], 'healthy recovery differs from reference')
        success = True
    finally:
        client.shutdown()
        end = time.monotonic() + cleanup_timeout
        for worker in workers:
            if worker.ident is not None:
                worker.join(max(0, end - time.monotonic()))
        evidence['cleanup'] = {'workers_stopped': [not w.is_alive() for w in workers]}
    require(success and all(evidence['cleanup']['workers_stopped']), 'worker cleanup incomplete')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--url', required=True)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--model')
    ap.add_argument('--stage-timeout', type=float, default=900)
    ap.add_argument('--io-timeout', type=float, default=10)
    ap.add_argument('--cleanup-timeout', type=float, default=15)
    ap.add_argument('--sample-interval', type=float, default=.02)
    a = ap.parse_args(argv)
    if any(not math.isfinite(t) or t <= 0 for t in (a.stage_timeout, a.io_timeout, a.cleanup_timeout, a.sample_interval)):
        ap.error('timeouts must be positive and finite')
    if not a.run:
        print(f'Dry run: private {CONTEXT}/cache1, two finite-{TARGET} streams, default template, actual pressure history, third waiter and recovery. No traffic.')
        return 0
    output = a.output.expanduser().resolve()
    if output.exists():
        ap.error('choose a fresh artifact filename')
    with output.open('x') as f:
        f.write('{}\n')
    evidence = {'schema': 1, 'harness': 'incremental-kv-HTTP-actual-pressure', 'passed': False,
                'url': a.url, 'samples': [], 'sampling': {'temperature': 0, 'seed': native.SEED},
                'context': CONTEXT, 'prompt_targets': list(PROMPT_COUNTS), 'target': TARGET, 'timeouts': {'stage_s': a.stage_timeout, 'io_s': a.io_timeout, 'cleanup_s': a.cleanup_timeout}}
    client = None
    try:
        client = common.Client(a.url, a.io_timeout)
        run(client, evidence, a.stage_timeout, a.sample_interval, a.cleanup_timeout, a.model)
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

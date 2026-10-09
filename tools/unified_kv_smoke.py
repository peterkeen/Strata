#!/usr/bin/env python3
"""Independent model-backed native unified-KV regression (private subprocess only).

Run on an available GPU, NOT against a production service:
  python3 tools/unified_kv_smoke.py --exe build/strata --config strata-model.json \
      --output /tmp/unified-512.json --context 512 --gpu 0
Repeat with --context 1024 and a NEW output path. No build, download or deployment.

Uses batch_test.tokenizer, but deliberately not its blocking Engine reader or the
HTTP server. Raw stdout is read by a thread into a Queue; each stage has an
absolute deadline, including startup. JSON includes commands, raw protocol lines,
token IDs, admission/completion order and failure/cleanup evidence. stderr goes
beside it as <output>.stderr.log. Configuration model paths/args, cwd, lib_dirs
and env are retained; serving, residency, scheduling and parity settings are
explicitly overridden. --gpu selects ONE physical CUDA/HIP device; by default
use the first configured device (or existing visibility). No slot partitioning.

The current native QSA page is four cells (qsa_real_shapes); physical capacity
rounds UP to pages. Require four-aligned context to avoid ambiguous pressure.
BGEN emits T + DONE during admission, THEN BADM; DONE is NOT batch completion.
BADM=0 completes at admission; BADM=1 requires BDONE. Natural EOS is allowed,
except where it would prevent observing overlap, waiting, cancellation or reuse.
There is no native 'waiting' line: pressure proof combines impossible-overlap
reservation arithmetic with BT progress before the pending T/BADM, and BDONE
before admission. Never infer pressure just from elapsed time or output length.
The harness forces --prefill 64 so the existing BYIELD guard can pause even
at context 512 with another full chunk remaining. The paused-prefill stage
bursts solo GEN + BYIELD 0, then BGEN 1 + BSTOP 0. It requires YIELDED + DONE
cancel, followed by the paused slot's zero-token BDONE cancel BEFORE waiter
output/admission, with no active decode rows to rescue admission. The waiter
may finish on EOS at admission (BADM=0); ordering and solo parity still apply.
A missing/not-taken yield is a failure, never a skipped test.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
PAGE_CELLS = 4
PREFILL_CELLS = 64


@dataclass(frozen=True)
class PartialTailSeed:
    """Tokenized partial-page seed with its USER/assistant framing boundaries."""
    token_ids: list[int]
    offset_cells: int
    target_cells: int
    background_start: int
    background_end: int
    user_close_start: int
    user_close_end: int
    assistant_start: int
    prefix_ids: tuple[int, ...]
    background_cycle_ids: tuple[int, ...]
    user_close_ids: tuple[int, ...]
    assistant_header_ids: tuple[int, ...]


def tokenizer(path):
    # Lazy import keeps protocol/lifecycle unit tests stdlib-only; model runs
    # use the exact existing tokenizer helper (and its regex dependency).
    from batch_test import tokenizer as load_tokenizer
    return load_tokenizer(path)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def build_partial_tail_seed(tok, offset_cells: int, context: int, *, output_cap: int = 8,
                            guard_cells: int = 8, minimum_cells: int = 96) -> PartialTailSeed:
    """Build a correctly framed user-background seed ending at a requested partial page.

    All sizing is in encoded token IDs. Padding is inserted between the USER
    background prefix and its final instruction/IM_END; the assistant thinking
    generation header is appended last and is never padded over.
    """
    require(offset_cells in (1, 2, 3), "partial-tail offset must be 1, 2, or 3 cells")
    require(output_cap == 8, "partial-tail seed fixture requires the original eight-token output cap")
    require(context > 0 and guard_cells >= 0 and minimum_cells > 0, "invalid tail seed capacity settings")
    user_prefix = tok.encode(
        "<|im_start|>user\nList numbered integers from 1 through 1000, one integer per line. "
        "Use the background only as context and follow the final instruction.\nBackground notes:\n",
        parse_special=True)
    background_cycle = tok.encode(" alpha beta gamma delta epsilon zeta eta theta", parse_special=False)
    user_close = tok.encode(
        "\nEnd of background. Continue the numbered integer list with more valid integers, "
        "one per line. Do not stop early.\n<|im_end|>", parse_special=True)
    assistant_header = tok.encode("\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    require(user_prefix and background_cycle and user_close and assistant_header,
            "tail seed framing/background tokenized to an empty segment")
    fixed_cells = len(user_prefix) + len(user_close) + len(assistant_header)
    target = max(minimum_cells, fixed_cells + 1)
    target += (offset_cells - (output_cap - 1) - target) % PAGE_CELLS
    background_cells = target - fixed_cells
    require(background_cells > 0, "tail seed needs positive user-background padding")
    background = (background_cycle *
                  ((background_cells + len(background_cycle) - 1) // len(background_cycle)))[:background_cells]
    tokens = list(user_prefix) + background + list(user_close) + list(assistant_header)
    background_start = len(user_prefix)
    background_end = background_start + background_cells
    user_close_start = background_end
    user_close_end = user_close_start + len(user_close)
    assistant_start = user_close_end
    require(len(tokens) == target, "tail seed token sizing drifted from target")
    require((len(tokens) + output_cap - 1) % PAGE_CELLS == offset_cells,
            "tail seed plus shared generated prefix misses requested partial-page offset")
    require(len(tokens) + output_cap + guard_cells <= context,
            "tail seed output cap plus native context guard exceeds context")
    return PartialTailSeed(tokens, offset_cells, target, background_start, background_end,
                           user_close_start, user_close_end, assistant_start,
                           tuple(user_prefix), tuple(background_cycle), tuple(user_close), tuple(assistant_header))


def parse_line(raw):
    """Strict known control records; preserve other native stdout as OTHER."""
    fields = raw.split()
    if not fields:
        return {'kind': 'OTHER'}
    kind = fields[0]
    try:
        if kind == 'INFO':
            return {'kind': kind, 'fields': dict(f.split('=', 1) for f in fields[1:] if '=' in f)}
        if kind == 'READY':
            require(len(fields) >= 2, 'short READY')
            return {'kind': kind, 'context': int(fields[1])}
        if kind == 'ERR':
            return {'kind': kind, 'message': raw[3:].strip()}
        if kind == 'T':
            require(len(fields) == 2, 'malformed T')
            return {'kind': kind, 'token': int(fields[1])}
        if kind == 'BT':
            require(len(fields) == 3, 'malformed BT')
            return {'kind': kind, 'slot': int(fields[1]), 'token': int(fields[2])}
        if kind == 'BADM':
            require(len(fields) == 3 and fields[2] in ('0', '1'), 'malformed BADM')
            return {'kind': kind, 'slot': int(fields[1]), 'continues': fields[2] == '1'}
        if kind == 'YIELDED':
            require(len(fields) == 3, 'malformed YIELDED')
            slot, tokens = int(fields[1]), int(fields[2])
            require(slot >= 0 and tokens > 0, 'YIELDED requires nonnegative slot and positive prefix')
            return {'kind': kind, 'slot': slot, 'tokens': tokens}
        if kind == 'BDONE':
            require(len(fields) >= 5, 'short BDONE')
            return {'kind': kind, 'slot': int(fields[1]), 'count': int(fields[2]),
                    'finish': fields[3], 'decode_ms': float(fields[4])}
        if kind == 'DONE':
            require(len(fields) >= 6, 'short DONE')
            return {'kind': kind, 'count': int(fields[1]), 'prompt_count': int(fields[2]),
                    'prompt_ms': float(fields[3]), 'decode_ms': float(fields[4]),
                    'finish': fields[5], 'reused': int(fields[8]) if len(fields) > 8 else None}
    except (ValueError, AssertionError) as exc:
        raise ValueError(f'malformed native record {raw!r}: {exc}') from exc
    return {'kind': 'OTHER'}


def verify_info(info, context):
    for key, wanted in {'kv_unified': '1', 'kv_capacity_cells': str(context),
                        'context': str(context), 'batch_slots': '2', 'kv_resident': '0',
                        'conversation_cache_mib': '0', 'slot_cache': '1', 'lookup': '0',
                        'mtp_max': '1', 'pcie_frac': '0.00'}.items():
        require(info.get(key) == wanted, f'INFO requires {key}={wanted}, got {info.get(key)!r}')


def resolve_path(path, cwd):
    """Config model/tokenizer/lib paths are engine-cwd relative, as in batch_test."""
    p = Path(path).expanduser()
    return str(p.resolve() if p.is_absolute() else (Path(cwd) / p).resolve())


def engine_settings(cfg, exe, context, gpu=None):
    cwd = str(Path(cfg.get('cwd') or os.getcwd()).expanduser().resolve())
    # Keep actual model/resident-expert configuration; replace ALL occurrences of
    # controlled options, rather than relying on last-option-wins validation.
    valued = {'--batch', '--batch-groups', '--max-context', '--max-new', '--kv-resident',
              '--conversation-cache-mib', '--pcie-frac', '--adapt-every', '--adapt-swaps',
              '--spec', '--mtp-max-t', '--suffix-draft', '--prompt-cache', '--prefill',
              '--layer-split', '--split-device', '--expert-profile-save',
              '--expert-profile-save-every'}
    flags = {'--serve', '--kv-unified', '--no-prefill-borrow', '--trim-stage-weights'}
    original = list(cfg['args'])
    args, i = [], 0
    while i < len(original):
        arg = original[i]
        if arg in valued:
            require(i + 1 < len(original), f'config option missing a value: {arg}')
            i += 2
        elif arg in flags:
            i += 1
        else:
            args.append(arg)
            i += 1
    args += ['--serve', '--batch', '2', '--batch-groups', '1', '--kv-unified',
             '--kv-resident', '0', '--conversation-cache-mib', '0', '--max-context', str(context),
             '--max-new', '1', '--pcie-frac', '0', '--adapt-every', '1000000',
             '--adapt-swaps', '0', '--spec', '2', '--mtp-max-t', '1', '--suffix-draft', '0',
             '--prompt-cache', '6', '--prefill', str(PREFILL_CELLS), '--no-prefill-borrow']
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (cfg.get('env') or {}).items()})
    env['STRATA_IQ_MT_MIN'] = '1'
    # Config env must survive, apart from the explicitly selected device and
    # exact IQ math. Native IQ requires --spec >=2; mtp_max=1 disables drafts.
    devices = cfg.get('gpu') or []
    if not isinstance(devices, list):
        devices = [devices]
    if gpu is None:
        gpu = cfg.get('hip_ordinal') if cfg.get('backend') == 'hip' else None
        if gpu is None and devices:
            gpu = devices[0]
    visibility = 'HIP_VISIBLE_DEVICES' if cfg.get('backend') == 'hip' else 'CUDA_VISIBLE_DEVICES'
    if gpu is not None:
        env[visibility] = str(gpu)
    elif env.get(visibility):
        env[visibility] = env[visibility].split(',')[0]
    if visibility == 'CUDA_VISIBLE_DEVICES':
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    dirs = [resolve_path(d, cwd) for d in cfg.get('lib_dirs') or []]
    if dirs:
        var = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return [str(Path(exe).expanduser().resolve()), *args], cwd, env


class NativeEngine:
    def __init__(self, evidence, stderr_path, cleanup_timeout=10):
        self.evidence = evidence
        self.stderr_path = stderr_path
        self.cleanup_timeout = cleanup_timeout
        self.p = self.log = self.reader = self.writer = None
        self.q = Queue()
        self.origin = time.monotonic()
        self.stage = 'startup'

    def start(self, command, cwd, env, context, timeout):
        self.log = open(self.stderr_path, 'xb')
        self.p = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log,
                                  start_new_session=(os.name == 'posix'))
        self.evidence['pid'] = self.p.pid
        self.reader = threading.Thread(target=self._read_stdout, name='native-stdout', daemon=True)
        self.reader.start()
        deadline = time.monotonic() + timeout
        info = None
        while True:
            event = self.next_event(deadline)
            if event['kind'] == 'INFO':
                require(info is None, 'duplicate startup INFO')
                info = event['fields']
                self.evidence['info'] = info
            elif event['kind'] == 'READY':
                require(info is not None, 'READY without required INFO kv_unified/kv_capacity_cells')
                verify_info(info, context)
                require(event['context'] == context, 'READY context differs')
                return
            else:
                require(event['kind'] == 'OTHER', f'unexpected startup event: {event}')

    def _read_stdout(self):
        try:
            for line in iter(self.p.stdout.readline, b''):
                self.q.put((time.monotonic() - self.origin, line.decode('utf-8', errors='replace').rstrip('\r\n')))
        except Exception as exc:
            self.q.put((time.monotonic() - self.origin, None, repr(exc)))
        finally:
            self.q.put((time.monotonic() - self.origin, None, 'EOF'))

    def _event(self, item):
        record = {'seq': len(self.evidence['stdout']), 'wall_s': item[0],
                  'stage': self.stage, 'raw': item[1]}
        self.evidence['stdout'].append(record)
        if item[1] is None:
            record['reader_status'] = item[2]
            raise RuntimeError(f'native stdout {item[2]} during {self.stage}; see {self.stderr_path}')
        event = parse_line(item[1])
        record.update(event)
        return record

    def next_event(self, deadline, allow_error=False):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'absolute stage deadline exceeded: {self.stage}')
        try:
            event = self._event(self.q.get(timeout=remaining))
        except Empty as exc:
            raise TimeoutError(f'no native output before stage deadline: {self.stage}; see {self.stderr_path}') from exc
        if event['kind'] == 'ERR' and not allow_error:
            raise RuntimeError(f'native ERR during {self.stage}: {event["raw"]}')
        return event

    def send(self, *lines, deadline=None):
        # Burst admissions: do not wait for A before submitting B. This removes
        # a Python round-trip in which a fast engine could finish A unnoticed.
        # Bound writes too: a hung engine need not keep draining its stdin pipe.
        deadline = deadline if deadline is not None else time.monotonic() + self.cleanup_timeout
        require(self.writer is None or not self.writer.is_alive(), 'previous stdin write still pending')
        for line in lines:
            self.evidence['commands'].append({'wall_s': time.monotonic() - self.origin,
                                               'stage': self.stage, 'raw': line})
        result = Queue()

        def write():
            try:
                self.p.stdin.write(('\n'.join(lines) + '\n').encode())
                self.p.stdin.flush()
                result.put(None)
            except Exception as exc:
                result.put(exc)

        self.writer = threading.Thread(target=write, name='native-stdin', daemon=True)
        self.writer.start()
        try:
            error = result.get(timeout=max(0, deadline - time.monotonic()))
        except Empty as exc:
            raise TimeoutError(f'stdin write exceeded stage deadline: {self.stage}') from exc
        self.writer.join(timeout=max(0, deadline - time.monotonic()))
        if error is not None:
            raise error

    def close(self, success):
        self.stage = 'cleanup'
        result = self.evidence.setdefault('cleanup', {})
        result['stdout_start_seq'] = len(self.evidence['stdout'])
        try:
            if self.p is not None:
                if success and self.p.poll() is None:
                    self.send('QUIT')
                    try:
                        self.p.wait(timeout=self.cleanup_timeout)
                    except subprocess.TimeoutExpired:
                        result['quit_timeout'] = True
                if self.p.poll() is None:
                    result['terminated'] = True
                    self._signal(signal.SIGTERM)
                    try:
                        self.p.wait(timeout=self.cleanup_timeout)
                    except subprocess.TimeoutExpired:
                        result['killed'] = True
                        self._signal(signal.SIGKILL)
                        self.p.wait(timeout=self.cleanup_timeout)
                result['returncode'] = self.p.returncode
        finally:
            # Even a broken stdin/QUIT must not leave a loaded model behind.
            if self.p is not None and self.p.poll() is None:
                self._signal(signal.SIGKILL)
                self.p.wait(timeout=self.cleanup_timeout)
            if self.p is not None:
                if self.writer:
                    self.writer.join(timeout=self.cleanup_timeout)
                    result['writer_stopped'] = not self.writer.is_alive()
                # A wrapper could exit while a child still owns the pipes.
                # Kill the remaining process group before closing buffered pipe
                # objects: close() can otherwise wait forever on a reader lock.
                if self.reader:
                    self.reader.join(timeout=self.cleanup_timeout)
                    if self.reader.is_alive() and os.name == 'posix':
                        result['killed_remaining_group'] = True
                        self._signal(signal.SIGKILL)
                        self.reader.join(timeout=self.cleanup_timeout)
                    result['reader_stopped'] = not self.reader.is_alive()
                if self.p.stdin and not (self.writer and self.writer.is_alive()):
                    self.p.stdin.close()
                if self.p.stdout and not (self.reader and self.reader.is_alive()):
                    self.p.stdout.close()
                # Retain every queued raw line, including errors on shutdown.
                while True:
                    try:
                        self._event(self.q.get_nowait())
                    except Empty:
                        break
                    except (RuntimeError, ValueError) as exc:
                        if not self.evidence['stdout'][-1].get('reader_status'):
                            result.setdefault('protocol_errors', []).append(str(exc))
            if self.log:
                self.log.close()

    def _signal(self, sig):
        try:
            if os.name == 'posix':
                os.killpg(self.p.pid, sig)
            elif sig == signal.SIGTERM:
                self.p.terminate()
            else:
                self.p.kill()
        except ProcessLookupError:
            pass


@dataclass
class Request:
    name: str
    prompt: list[int]
    cap: int
    slot: int | None = None
    tokens: list[int] = field(default_factory=list)
    admission_done: dict | None = None
    badm: dict | None = None
    completion: dict | None = None
    token_events: list[dict] = field(default_factory=list)

    def command(self):
        head = f'GEN {self.cap}' if self.slot is None else f'BGEN {self.slot} {self.cap}'
        return head + ' temperature=0 ' + ','.join(map(str, self.prompt))

    def record(self):
        return {'name': self.name, 'slot': self.slot, 'prompt_ids': self.prompt,
                'prompt_tokens': len(self.prompt), 'max_new': self.cap,
                'reservation_cells': len(self.prompt) + self.cap,
                'reservation_pages': (len(self.prompt) + self.cap + PAGE_CELLS - 1) // PAGE_CELLS,
                'tokens': self.tokens, 'admission_done': self.admission_done,
                'badm': self.badm, 'completion': self.completion,
                'token_events': self.token_events}


class Protocol:
    """Ordered GEN/BGEN admission channel plus multiplexed slot output."""
    def __init__(self, requests):
        self.requests = requests
        self.pending = deque(requests)
        self.active = {}
        require(len({r.slot for r in requests}) == len(requests), 'duplicate slots/solo requests')

    @property
    def finished(self):
        return not self.pending and not self.active

    def consume(self, event):
        kind = event['kind']
        if kind == 'OTHER':
            return
        require(kind != 'ERR', f'unexpected ERR: {event}')
        if kind in ('T', 'DONE', 'BADM'):
            require(bool(self.pending), f'orphan admission event: {event}')
            req = self.pending[0]
            if kind == 'T':
                require(req.admission_done is None, 'T after admission DONE')
                self._token(req, event)
            elif kind == 'DONE':
                require(req.admission_done is None, 'duplicate DONE')
                require(event['count'] == len(req.tokens), 'DONE count does not match admission T tokens')
                require(event['prompt_count'] == len(req.prompt), 'DONE prompt count differs')
                req.admission_done = event
                if req.slot is None:
                    req.completion = event
                    self.pending.popleft()
            else:
                require(req.slot == event['slot'], 'BADM slot differs from pending admission')
                require(req.admission_done is not None, 'BADM without admission DONE')
                req.badm = event
                self.pending.popleft()
                if event['continues']:
                    require(len(req.tokens) == 1 and req.cap > 1, 'BADM=1 without one first token')
                    require(req.admission_done['finish'] == 'length', 'BADM=1 after non-length admission')
                    require(req.slot not in self.active, 'BADM on active slot')
                    self.active[req.slot] = req
                else:
                    req.completion = req.admission_done
        elif kind in ('BT', 'BDONE'):
            require(event['slot'] in self.active, f'orphan batch event: {event}')
            req = self.active[event['slot']]
            if kind == 'BT':
                self._token(req, event)
            else:
                require(event['count'] == len(req.tokens), 'BDONE count does not match T + BT')
                req.completion = event
                del self.active[req.slot]
        else:
            raise AssertionError(f'unexpected post-startup event: {event}')

    @staticmethod
    def _token(req, event):
        require(event['token'] >= 0, 'negative output token')
        req.tokens.append(event['token'])
        req.token_events.append({'seq': event['seq'], 'kind': event['kind'], 'wall_s': event['wall_s']})
        require(len(req.tokens) <= req.cap, f'{req.name}: output cap exceeded')


def normal(req):
    require(req.completion is not None and req.tokens, f'{req.name}: incomplete/empty output')
    require(req.completion['finish'] in ('length', 'stop'), f'{req.name}: abnormal completion')
    if req.completion['finish'] == 'length':
        require(len(req.tokens) == req.cap, f'{req.name}: length finish before requested cap')


def parity(req, reference):
    normal(req)
    require(req.tokens == reference.tokens,
            f'{req.name}: solo parity differs (solo {len(reference.tokens)}, batch {len(req.tokens)})')


def verify_pressure(active, waiter, context, cancelled):
    pages = lambda cells: (cells + PAGE_CELLS - 1) // PAGE_CELLS

    def reserve(req):
        # Prefer native admission evidence, not Request.record()'s nominal
        # prompt + full cap (also used by incremental_kv_smoke.Attempt).
        for record in (req.badm, req.admission_done):
            if record and 'reservation_pages' in record:
                return record['reservation_pages']
            if record and 'reservation_cells' in record:
                return pages(record['reservation_cells'])
        return pages(len(req.prompt) + min(req.cap, 256))

    require(all(reserve(r) <= context // PAGE_CELLS for r in (active, waiter)), 'request cannot fit alone')
    # At BADM the active has produced one unfed token: p + 1 = prompt + 1.
    # Later decoding only increases this required extent. The waiter must at
    # least back its whole known prompt, even with ALL optional headroom gone.
    # Do not use final output counts to claim a shortage existed at admission.
    required_pages = [pages(len(active.prompt) + 1), pages(len(waiter.prompt))]
    require(all(p <= context // PAGE_CELLS for p in required_pages), 'required extent cannot fit alone')
    common = 0
    for left, right in zip(active.prompt, waiter.prompt):
        if left != right:
            break
        common += 1
    # Even hypothetical sharing of ALL common prompt pages cannot make these
    # overlap. Actual runtime never uses an active slot as a cached source.
    shared_upper_bound = (common + PAGE_CELLS - 1) // PAGE_CELLS
    require(sum(required_pages) - shared_upper_bound > context // PAGE_CELLS,
            f'fixture cannot force a shortage: required pages {required_pages} minus '
            f'{shared_upper_bound} shared pages <= pool {context // PAGE_CELLS}')
    require(active.badm and active.badm['continues'], 'active request never reached batch windows (early EOS)')
    require(active.completion and active.completion['kind'] == 'BDONE', 'missing active BDONE')
    require(waiter.badm is not None and waiter.tokens, 'waiting request never admitted')
    end = active.completion['seq']
    require(active.badm['seq'] < end < waiter.token_events[0]['seq'] < waiter.badm['seq'],
            'waiting request emitted/admitted before active reservation was released')
    progress = [e for e in active.token_events if e['kind'] == 'BT' and e['seq'] < end]
    require(progress, 'no active BT progress while admission waited (early EOS or stall)')
    require(active.completion['finish'] == 'cancel' if cancelled else
            active.completion['finish'] in ('length', 'stop'), 'wrong pressure completion reason')
    if cancelled:
        require(len(active.tokens) < active.cap, 'cancellation did not release a remaining output reservation')
    else:
        normal(active)
    normal(waiter)
    return {'reserved_pages': [reserve(active), reserve(waiter)], 'pool_pages': context // PAGE_CELLS,
            'required_pages_lower_bound': required_pages,
            'common_prompt_pages_upper_bound': shared_upper_bound, 'active_badm_seq': active.badm['seq'], 'progress_before_completion': len(progress),
            'active_bdone_seq': end, 'waiting_first_t_seq': waiter.token_events[0]['seq'],
            'waiting_badm_seq': waiter.badm['seq'], 'cancelled': cancelled}


class Suite:
    def __init__(self, engine, evidence, output, timeout, context):
        self.engine, self.evidence, self.output = engine, evidence, output
        self.timeout, self.context = timeout, context

    def save(self):
        self.output.write_text(json.dumps(self.evidence, indent=2) + '\n', encoding='utf-8')

    def run(self, name, requests, cancel=False):
        self.engine.stage = name
        stage = {'name': name, 'passed': False, 'requests': []}
        self.evidence['stages'].append(stage)
        protocol = Protocol(requests)
        deadline = time.monotonic() + self.timeout
        sent = False
        try:
            self.engine.send(*(r.command() for r in requests), deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                protocol.consume(event)
                if cancel and not sent and event['kind'] == 'BT' and event['slot'] == requests[0].slot:
                    require(protocol.pending and protocol.pending[0] is requests[1],
                            'cancellation fixture did not reach capacity waiting')
                    require(not requests[1].tokens, 'waiting request emitted before BSTOP')
                    self.engine.send(f'BSTOP {requests[0].slot}', deadline=deadline)
                    stage['bstop_after_seq'] = event['seq']
                    sent = True
            if cancel:
                require(sent, 'never reached cancellation trigger (early EOS)')
            return stage
        finally:
            stage['requests'] = [r.record() for r in requests]
            self.save()

    def solo(self, name, prompt, cap):
        req = Request(name, prompt, cap)
        stage = self.run(name, [req])
        normal(req)
        stage['passed'] = True
        self.save()
        return req

    def paused_capacity_wait(self, prompt, reference):
        """A partial slot, not an active decode row, must be stopped IN admission.

        Both controls are burst behind their request. Processing BSTOP only in
        the outer command loop (or after !batch_on) will reject the waiter or
        stall; silently evicting the protected partial slot will admit too early.
        """
        self.engine.stage = 'capacity-wait-paused-prefill-bstop'
        source = Request('paused-solo-source', prompt, 96)
        waiter = Request('waiter-after-paused-bstop', reference.prompt, reference.cap, 1)
        stage = {'name': self.engine.stage, 'passed': False}
        self.evidence['stages'].append(stage)
        deadline = time.monotonic() + self.timeout
        yielded = release = None
        try:
            pages = [r.record()['reservation_pages'] for r in (source, waiter)]
            common = 0
            for left, right in zip(source.prompt, waiter.prompt):
                if left != right:
                    break
                common += 1
            shared_upper_bound = (common + PAGE_CELLS - 1) // PAGE_CELLS
            require(len(prompt) - 1 > 2 * PREFILL_CELLS and len(prompt) + source.cap + 8 <= self.context,
                    'paused source must leave another full prefill chunk and fit individually')
            require(all(p <= self.context // PAGE_CELLS for p in pages), 'paused fixture cannot fit alone')
            require(sum(pages) - shared_upper_bound > self.context // PAGE_CELLS,
                    'paused reservation and waiter can overlap after prefix sharing')
            stage['pressure'] = {'expected_held_reservation_pages': pages[0], 'waiter_pages': pages[1],
                                 'pool_pages': self.context // PAGE_CELLS,
                                 'common_prompt_pages_upper_bound': shared_upper_bound,
                                 'prefill_cells': PREFILL_CELLS, 'active_decode_rows_before_waiter': 0}
            protocol = Protocol([source])
            self.engine.send(source.command(), 'BYIELD 0', deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                if event['kind'] == 'YIELDED':
                    stage['yielded'] = event
                    require(yielded is None, 'duplicate YIELDED for paused source')
                    require(event['slot'] == 0 and 0 < event['tokens'] < len(prompt),
                            'YIELDED must park a positive incomplete source prefix in slot 0')
                    yielded = event
                else:
                    protocol.consume(event)
            require(yielded is not None, 'queued BYIELD 0 was not taken; no paused reservation was established')
            require(not source.tokens and source.completion['finish'] == 'cancel',
                    'paused solo GEN must emit DONE 0 ... cancel without generating tokens')
            require(yielded['seq'] < source.completion['seq'], 'YIELDED must precede source DONE cancel')
            require(source.completion['reused'] == 0, 'paused source must start cold, not skip its chunk boundary')
            stage['pressure']['yielded_prefix_pages'] = (yielded['tokens'] + PAGE_CELLS - 1) // PAGE_CELLS
            self.save()

            protocol = Protocol([waiter])  # Slot 0 is partial, never an active row.
            self.engine.send(waiter.command(), 'BSTOP 0', deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                if event['kind'] == 'BDONE' and event['slot'] == 0:
                    stage['paused_release'] = event
                    require(release is None, 'duplicate paused-slot BDONE')
                    require(event['count'] == 0 and event['finish'] == 'cancel' and event['decode_ms'] == 0,
                            'paused BSTOP must emit BDONE 0 0 cancel 0, not decode a row')
                    require(not waiter.tokens and waiter.admission_done is None and waiter.badm is None,
                            'waiter emitted/admitted before paused reservation release')
                    release = event
                else:
                    if event['kind'] != 'OTHER':
                        require(release is not None, 'waiter protocol arrived before paused-slot BDONE cancel')
                    protocol.consume(event)
            require(release is not None, 'no paused-slot reservation release observed')
            # Natural EOS may finish this waiter on its admission token. The
            # regression is releasing paused capacity before admission, not
            # requiring this otherwise healthy request to enter batch decode.
            require(waiter.badm is not None, 'waiter admission must emit BADM 1')
            parity(waiter, reference)
            require(source.completion['seq'] < release['seq'] < waiter.token_events[0]['seq'] < waiter.badm['seq'],
                    'invalid paused release/admission ordering')
            stage['pressure'].update(source_done_seq=source.completion['seq'], paused_bdone_seq=release['seq'],
                                     waiting_first_t_seq=waiter.token_events[0]['seq'],
                                     waiting_badm_seq=waiter.badm['seq'], waiter_continues=waiter.badm['continues'])
            stage['passed'] = True
        finally:
            stage['requests'] = [source.record(), waiter.record()]
            self.save()

    def invalid_then_healthy(self, prompt, reference):
        self.engine.stage = 'invalid-oversized-and-recovery'
        stage = {'name': self.engine.stage, 'passed': False}
        self.evidence['stages'].append(stage)
        # BGEN is internally GEN 1; overflow PROMPT, not just BGEN max_new,
        # so the existing n + 1 + 8 context guard must reject before execution.
        oversized = (prompt * (self.context // len(prompt) + 2))[:self.context + PAGE_CELLS]
        bad = Request('oversized', oversized, 8, 0)
        good = Request('healthy-after-error', prompt, reference.cap, 0)
        stage['invalid_request'] = bad.record()
        protocol = Protocol([good])
        deadline = time.monotonic() + self.timeout
        self.engine.send(bad.command(), good.command(), deadline=deadline)
        event = self.engine.next_event(deadline, allow_error=True)
        while event['kind'] == 'OTHER':
            event = self.engine.next_event(deadline, allow_error=True)
        require(event['kind'] == 'ERR' and 'exceeds the context' in event['message'],
                f'oversized request did not produce the expected context ERR: {event}')
        stage['expected_error'] = event
        try:
            while not protocol.finished:
                protocol.consume(self.engine.next_event(deadline))
            parity(good, reference)
            stage['passed'] = True
        finally:
            stage['healthy_request'] = good.record()
            self.save()


def fixtures(tok, context):
    def chat(question):
        return tok.encode(f'<|im_start|>user\n{question}<|im_end|>\n'
                          '<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
    long_question = ('List numbered items from 1 through 1000. Each item must name a different animal '
                     'and describe it in a full sentence. Keep listing; do not summarize or stop early.')
    a = chat(long_question)
    b = chat('List numbered steps from 1 through 1000 for learning mathematics. Each step must be a full sentence. '
             'Use this background: ' + 'Arithmetic geometry algebra analysis. ' * 12)
    filler = tok.encode(' alpha beta gamma delta epsilon zeta eta theta')
    require(filler, 'empty filler tokens')
    # Large PROMPT plus short output: more than the free quarter of the pool,
    # but individually fits. Its prefix differs from A immediately after framing.
    wait_base = chat('Read the following data, then list numbered facts about it.\n')
    waiter = wait_base + (filler * context)[:context // 2 - len(wait_base)]
    require(len(wait_base) < context // 2, 'waiter template too large')
    require(len(a) + 3 * context // 4 + 8 <= context, 'active prompt too large for pressure cap')
    require((len(a) + 32 + 3) // 4 + (len(b) + 32 + 3) // 4 <= context // 4,
            'ordinary unequal pair does not fit aggregate context')
    require(len(a) != len(b), 'ordinary prompts must have unequal lengths')
    # Raw unique prefix avoids chat turn/checkpoint splits and all earlier
    # cached prompts. 324/708 cells at context 512/1024, plus output reserve 96.
    # Forced prefill 64 leaves >max(chunk, default short_read=64) after its
    # first chunk, satisfying the existing BYIELD guard at both capacities.
    source_base = tok.encode('Paused prefill regression source. Read this data, then list numbered observations:\n')
    source_count = 3 * context // 4 - 60
    require(len(source_base) < source_count, 'paused source template too large')
    source = source_base + (filler * context)[:source_count - len(source_base)]
    require(len(source) + 96 + 8 <= context, 'paused source reservation exceeds context')
    return chat, a, b, waiter, filler, source


def run_suite(suite, tok):
    context = suite.context
    chat, a, b, waiter, filler, source = fixtures(tok, context)
    sa = suite.solo('solo-unequal-A', a, 32)
    sb = suite.solo('solo-unequal-B', b, 32)
    pair = [Request('unequal-A', a, 32, 0), Request('unequal-B', b, 32, 1)]
    stage = suite.run('ordinary-two-slot-parity', pair)
    for req, ref in zip(pair, [sa, sb]):
        parity(req, ref)
        require(req.badm['continues'], 'ordinary fixture ended at admission (early EOS)')
    require(pair[1].badm['seq'] < pair[0].completion['seq'], 'ordinary pair did not overlap')
    stage['passed'] = True
    suite.save()

    # Incremental admission trims optional active headroom to p + 1, so a
    # large output cap cannot force waiting. Use a large, distinct prompt:
    # at 512 cells, active prompt/cap = 320/128 (448 cells = 112 admission
    # pages), waiter = 256/16 (272 cells = 68 pages). After trimming, even
    # required extents alone need ceil(321/4) + ceil(256/4) = 81 + 64 = 145
    # pages > 128. At 1024: 161 + 128 = 289 > 256. Each fits alone.
    pressure_head = tok.encode('Capacity-wait active source. Read the data, then follow the instructions:\n')
    pressure_cells, pressure_cap = 5 * context // 8, context // 4
    padding = pressure_cells - len(pressure_head) - len(a)
    require(padding >= 0, 'pressure prompt template too large')
    pressure_prompt = pressure_head + (filler * context)[:padding] + a
    require(pressure_prompt[0] != waiter[0], 'pressure fixture must have distinct initial tokens')
    require(len(pressure_prompt) + pressure_cap + 8 <= context and len(waiter) + 16 + 8 <= context,
            'pressure request cannot fit alone')
    required_pages = [(n + PAGE_CELLS - 1) // PAGE_CELLS for n in (len(pressure_prompt) + 1, len(waiter))]
    require(sum(required_pages) > context // PAGE_CELLS,
            f'fixture cannot force a shortage: required pages {required_pages} <= pool {context // PAGE_CELLS}')
    sw = suite.solo('solo-pressure-waiter', waiter, 16)
    for cancelled in (False, True):
        name = 'capacity-wait-bstop' if cancelled else 'capacity-wait-completion'
        active = Request(name + '-active', pressure_prompt, pressure_cap, 0)
        waiting = Request(name + '-waiting', waiter, 16, 1)
        stage = suite.run(name, [active, waiting], cancel=cancelled)
        stage['pressure'] = verify_pressure(active, waiting, context, cancelled)
        parity(waiting, sw)
        if cancelled:
            require(stage['bstop_after_seq'] < active.completion['seq'], 'BDONE preceded BSTOP trigger')
        stage['passed'] = True
        suite.save()

    suite.paused_capacity_wait(source, sw)
    suite.invalid_then_healthy(a, sa)

    # Three branches force shared cached prefixes to end at EVERY partial-page
    # offset. A completed cap-8 slot holds prompt + first seven output tokens.
    # Use the solo result to construct branches, then reseed slot 0 after solo
    # references, ensuring its idle slot is the best prefix source, not session 0.
    for offset in (1, 2, 3):
        seed_layout = build_partial_tail_seed(tok, offset, context, output_cap=8, guard_cells=8)
        seed = seed_layout.token_ids
        require(len(seed) == seed_layout.target_cells and seed_layout.offset_cells == offset,
                'partial-tail seed helper returned inconsistent token sizing')
        seed_ref = suite.solo(f'solo-seed-{offset}', seed, 8)
        require(len(seed_ref.tokens) == 8, 'seed ended early; cannot establish partial-page reuse')
        shared = seed + seed_ref.tokens[:-1]
        require(len(shared) % PAGE_CELLS == offset, 'incorrect partial-page fixture')
        branches = [shared + tok.encode(text) for text in ('\nContinue with even numbers:\n',
                                                          '\nContinue with odd numbers:\n')]
        require(all(len(p) + 16 + 8 <= context for p in branches),
                'partial-tail divergent branch plus output guard exceeds context')
        refs = [suite.solo(f'solo-branch-{offset}-{i}', p, 16) for i, p in enumerate(branches)]
        seed_req = Request(f'seed-{offset}', seed, 8, 0)
        stage = suite.run(f'reuse-seed-{offset}', [seed_req])
        parity(seed_req, seed_ref)
        require(seed_req.badm['continues'], 'seed never populated its slot')
        stage['passed'] = True
        for i, (prompt, ref) in enumerate(zip(branches, refs)):
            req = Request(f'branch-{offset}-{i}', prompt, 16, 1 - i)
            stage = suite.run(f'partial-page-{offset}-branch-{i}', [req])
            parity(req, ref)
            require(req.admission_done['reused'] is not None and
                    req.admission_done['reused'] >= len(shared), 'partial-page prefix was not actually reused')
            stage['shared_prefix_cells'] = len(shared)
            stage['partial_page_offset'] = offset
            stage['passed'] = True
            suite.save()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exe', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path, help='NEW JSON evidence file (not a directory)')
    ap.add_argument('--context', type=int, default=512, help='aggregate KV cells; >=512, multiple of 4')
    ap.add_argument('--gpu', help='one physical CUDA/HIP ordinal/UUID; default first configured device')
    ap.add_argument('--startup-timeout', type=float, default=600)
    ap.add_argument('--stage-timeout', type=float, default=300, help='absolute deadline per stage, not per token')
    ap.add_argument('--cleanup-timeout', type=float, default=10)
    args = ap.parse_args(argv)
    if args.context < 512 or args.context % PAGE_CELLS or args.context > 4096:
        ap.error('context must be 512..4096 and a multiple of 4 (native physical-page rounding)')
    if any(not 0 < t < float('inf') for t in (args.startup_timeout, args.stage_timeout, args.cleanup_timeout)):
        ap.error('timeouts must be positive and finite')
    if args.gpu is not None and (not args.gpu.strip() or ',' in args.gpu):
        ap.error('--gpu must select exactly one device')
    output = args.output.expanduser().resolve()
    stderr_path = Path(str(output) + '.stderr.log')
    if output.exists() or stderr_path.exists():
        ap.error('evidence/log exists; choose a new --output')
    # Exclusive creation also prevents accidental clobbering on concurrent runs.
    with output.open('x', encoding='utf-8') as f:
        f.write('{}\n')
    evidence = {'schema': 1, 'passed': False, 'config': str(args.config.resolve()),
                'exe': str(args.exe.resolve()), 'context': args.context, 'page_cells': PAGE_CELLS,
                'stderr_path': str(stderr_path), 'commands': [], 'stdout': [], 'stages': [],
                'timeouts': {'startup_s': args.startup_timeout, 'stage_s': args.stage_timeout,
                             'cleanup_s': args.cleanup_timeout}}
    engine = NativeEngine(evidence, stderr_path, args.cleanup_timeout)
    suite = Suite(engine, evidence, output, args.stage_timeout, args.context)
    success = False
    try:
        cfg = json.loads(args.config.read_text(encoding='utf-8'))
        command, cwd, env = engine_settings(cfg, args.exe, args.context, args.gpu)
        evidence.update(command=command, cwd=cwd, config_env=cfg.get('env') or {},
                        device_env={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                       'CUDA_DEVICE_ORDER', 'STRATA_IQ_MT_MIN') if k in env})
        suite.save()
        tok = tokenizer(resolve_path(cfg['tokenizer'], cwd))
        # Validate fixture sizes BEFORE loading a model.
        fixtures(tok, args.context)
        engine.start(command, cwd, env, args.context, args.startup_timeout)
        run_suite(suite, tok)
        require(all(stage['passed'] for stage in evidence['stages']), 'incomplete stage')
        success = True
    except (Exception, KeyboardInterrupt) as exc:
        evidence['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'stage': engine.stage}
    finally:
        try:
            engine.close(success)
            if success:
                require(evidence['cleanup'].get('returncode') == 0, 'native process did not exit cleanly')
                require(not evidence['cleanup'].get('quit_timeout'), 'native QUIT timed out')
                require(evidence['cleanup'].get('reader_stopped'), 'native stdout reader did not stop')
                require(evidence['cleanup'].get('writer_stopped'), 'native stdin writer did not stop')
                require(not evidence['cleanup'].get('protocol_errors'), 'malformed shutdown protocol')
                allowed_errors = {s['expected_error']['seq'] for s in evidence['stages'] if 'expected_error' in s}
                require(not any(e.get('kind') == 'ERR' and e['seq'] not in allowed_errors
                                for e in evidence['stdout']), 'unexpected trailing native ERR')
                trailing = evidence['stdout'][evidence['cleanup']['stdout_start_seq']:]
                require(all(e.get('kind', 'OTHER') == 'OTHER' for e in trailing),
                        'unexpected protocol output after final completion')
        except (Exception, KeyboardInterrupt) as exc:
            success = False
            evidence['cleanup_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        evidence['passed'] = success
        suite.save()
    print(f'{"PASS" if success else "FAIL"}: {output}; native stderr: {stderr_path}', flush=True)
    if not success:
        print(json.dumps(evidence.get('failure') or evidence.get('cleanup_failure')), file=sys.stderr)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""CPU-only safety/parser tests; never contact production or load a model."""
import contextlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nibbler_perf as p
import nibbler_lifecycle as life
from nibbler_lifecycle_test import LifecycleTests  # Full offline process/API proofs in this command too.

CFG = json.loads((Path(__file__).resolve().parents[1] / 'deploy/nibbler/config.json').read_text())


class FakeResponse:
    def __init__(self, lines):
        self.lines = iter(lines)
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def readline(self, _):
        return next(self.lines, b'')


def event(obj):
    return b'data: ' + json.dumps(obj).encode() + b'\n'


class CalibrationTests(unittest.TestCase):
    def test_default_is_no_side_effect_plan(self):
        with patch.object(p, 'live', side_effect=AssertionError('live called')), \
             patch.object(p, 'http', side_effect=AssertionError('HTTP called')), \
             patch.object(p, 'command', side_effect=AssertionError('command called')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir called')), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(p.main([]), 0)
        plan = json.loads(out.getvalue())
        self.assertFalse(plan['run'])
        self.assertEqual(plan['settings'][0], {'frac': None, 'workers': 11, 'reserve': 2048})

    def test_fresh_cli_dry_run_creates_no_files_or_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for module in (p.__file__, life.__file__):
                shutil.copy2(module, root / Path(module).name)
            before = sorted(str(path.relative_to(root)) for path in root.rglob('*'))
            result = subprocess.run([sys.executable, str(root / 'nibbler_perf.py'), '--fractions', 'baseline'],
                                    cwd=root, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)['run'])
            self.assertEqual(sorted(str(path.relative_to(root)) for path in root.rglob('*')), before)

    def test_explicit_opt_in(self):
        with patch.object(p, 'live', return_value=17) as live, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.main(['--run', '--fractions', 'baseline']), 17)
        self.assertTrue(live.call_args.args[0].run)

    def test_neutral_cli_requires_explicit_run_and_rejects_measurement_gates(self):
        with patch.object(p, 'live', side_effect=AssertionError('execution forbidden')), \
             patch.object(p, 'http', side_effect=AssertionError('HTTP forbidden')), \
             patch.object(p, 'command', side_effect=AssertionError('command forbidden')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir forbidden')), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for args in (['--preflight-only'], ['--run', '--preflight-only', '--reserve-gates'],
                         ['--run', '--preflight-only', '--reserves', '2048,1536']):
                with self.subTest(args=args), self.assertRaises((SystemExit, p.Stop)):
                    p.main(args)
        with patch.object(p, 'live', return_value=0) as live, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(p.main(['--run', '--preflight-only', '--fractions', 'baseline']), 0)
        self.assertTrue(live.call_args.args[0].preflight_only)
        self.assertEqual(json.loads(out.getvalue())['mode'], 'preflight-only')

    def test_refused_neutral_cli_creates_no_files_or_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for module in (p.__file__, life.__file__):
                shutil.copy2(module, root / Path(module).name)
            before = sorted(str(path.relative_to(root)) for path in root.rglob('*'))
            result = subprocess.run([sys.executable, str(root / 'nibbler_perf.py'), '--preflight-only'],
                                    cwd=root, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn('requires explicit --run', result.stderr)
            self.assertEqual(sorted(str(path.relative_to(root)) for path in root.rglob('*')), before)

    def test_no_combinatorial_sweep(self):
        with self.assertRaises(p.Stop), contextlib.redirect_stdout(io.StringIO()):
            p.main(['--fractions', '0.25,0.40', '--workers', '11,7'])

    def test_lower_reserve_needs_gates(self):
        with self.assertRaises(p.Stop), contextlib.redirect_stdout(io.StringIO()):
            p.main(['--fractions', '0.25', '--reserves', '2048,1536'])

    def test_reserve_must_start_control(self):
        with self.assertRaises(p.Stop), contextlib.redirect_stdout(io.StringIO()):
            p.main(['--fractions', '0.25', '--reserves', '1536', '--reserve-gates'])

    def test_bounds_and_path_validation(self):
        for args in (['--max-tokens', '10000'], ['--port', '8088'], ['--tag', '../../prod'],
                     ['--window-minutes', '0'], ['--workers', '24'], ['--repeats', '0']):
            with self.subTest(args=args), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises((p.Stop, SystemExit)):
                    p.main(args)

    def test_threshold_exact_and_below(self):
        p.safety({'MemAvailable': 10*p.GIB, 'SwapFree': p.GIB})
        for values in ({'MemAvailable': 10*p.GIB-1, 'SwapFree': p.GIB},
                       {'MemAvailable': 10*p.GIB, 'SwapFree': p.GIB-1}):
            with self.assertRaises(p.Stop):
                p.safety(values)

    def test_config_preserves_production_and_is_private(self):
        before = json.dumps(CFG, sort_keys=True)
        out = p.private_config(CFG, Path('/trial/run'), {'frac': .4, 'workers': 7, 'reserve': 1536}, 18089)
        self.assertEqual(json.dumps(CFG, sort_keys=True), before)
        self.assertEqual(out['parallel'], 2)
        self.assertEqual(out['host'], '127.0.0.1')
        self.assertEqual(out['exe'], '/trial/run/strata')
        self.assertEqual(out['vision']['exe'], '/trial/run/strata-vision')
        self.assertEqual(out['cwd'], '/trial/run')
        self.assertEqual(out['log'], '/trial/run/engine.log')
        self.assertEqual(out['args'][out['args'].index('--pcie-frac')+1], '0.4')
        for k in ('tokenizer', 'model_name', 'parallel'):
            self.assertEqual(out[k], CFG[k])
        p.validate_config(out)

    def test_baseline_keeps_link_probe(self):
        out = p.private_config(CFG, Path('/trial'), {'frac': None, 'workers': 11, 'reserve': 2048}, 18089)
        self.assertNotIn('--pcie-frac', out['args'])
        self.assertEqual(out['env']['STRATA_PREFILL_NVFP4'], CFG['env']['STRATA_PREFILL_NVFP4'])
        self.assertNotIn('STRATA_VERIFY_PROFILE', out['env'])  # GPU event profiling slows speed measurements

    def test_invariants_fail_closed(self):
        for key, value in (('parallel', 1), ('model_name', 'other')):
            cfg = json.loads(json.dumps(CFG))
            cfg[key] = value
            with self.assertRaises(p.Stop):
                p.validate_config(cfg)

    def test_flag_removes_duplicates(self):
        self.assertEqual(p.flag(['--x', '1', '--y', '2', '--x', '3'], '--x', '4'), ['--y', '2', '--x', '4'])

    def test_stream_collects_actual_tokens_timings_and_first_output(self):
        rows = [event({'choices': [{'delta': {'role': 'assistant'}}]}),
                event({'choices': [{'delta': {'content': '1, 2, 3'}}]}),
                event({'choices': [{'delta': {}, 'finish_reason': 'stop'}],
                       'usage': {'prompt_tokens': 741, 'completion_tokens': 7},
                       'timings': {'predicted_per_second': 28}}), b'data: [DONE]\n']
        with patch.object(p.urllib.request, 'urlopen', return_value=FakeResponse(rows)):
            result = p.request('http://127.0.0.1:1', 'synthetic')
        self.assertEqual(result['usage']['prompt_tokens'], 741)
        self.assertEqual(result['output_preview'], '1, 2, 3')
        self.assertGreaterEqual(result['ttft_s'], 0)
        self.assertEqual(len(result['chunk_times_s']), 1)

    def test_successful_empty_unload_response(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b''
        with patch.object(p.urllib.request, 'urlopen', return_value=response):
            self.assertEqual(p.http('http://127.0.0.1:1/api/models/unload/test', {}), {})

    def test_plaintext_unload_is_opt_in_strict_json_otherwise(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'Model unloaded'
        response.__enter__.return_value.status = 200
        with patch.object(p.urllib.request, 'urlopen', return_value=response):
            self.assertEqual(p.http('http://127.0.0.1:1/api/models/unload/test', {}, json_response=False),
                             {'status': 200, 'body': 'Model unloaded'})
            with self.assertRaises(json.JSONDecodeError):
                p.http('http://127.0.0.1:1/metrics')

    def test_stream_errors_and_incomplete_stop(self):
        for rows in ([event({'error': {'message': 'OOM'}})], [b'data: [DONE]\n'],
                     [event({'choices': [{'delta': {'content': 'bad'}, 'finish_reason': 'error'}]})]):
            with patch.object(p.urllib.request, 'urlopen', return_value=FakeResponse(rows)), self.assertRaises(p.Stop):
                p.request('http://127.0.0.1:1', 'synthetic')

    def test_monitor_abort_signals_only_owned_identity(self):
        with tempfile.TemporaryDirectory() as d:
            monitor = p.Monitor(Path(d), 'http://127.0.0.1:1', 100)
            who = {'pid': 98765, 'start': '123', 'boot': 'fixture'}
            monitor.process = life.OwnedProcess(unittest.mock.Mock(), who)
            with patch.object(life, 'signal_identity') as kill, patch.object(p.os, 'killpg') as unsafe:
                monitor.abort('safety threshold')
            kill.assert_called_once_with(who, signal.SIGTERM)
            unsafe.assert_not_called()
            with self.assertRaises(p.Stop):
                monitor.check()

    def test_cleanup_after_parent_exited_still_requests_full_tree_reaper(self):
        owner = unittest.mock.Mock()
        proc = life.OwnedProcess(owner, {'pid': 76543, 'start': '123', 'boot': 'fixture'})
        with patch.object(life, 'same', return_value=False), patch.object(p.os, 'killpg') as unsafe:
            self.assertIsNotNone(proc.poll())
            p.close_process(proc)
        owner.call.assert_called_once_with({'op': 'stop'})
        unsafe.assert_not_called()
        # Real frontend-exit + detached grandchildren proof is in LifecycleTests.

    def test_real_owned_process_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            owner = life.Supervisor(Path(d) / 'journal.json', [sys.executable, '-c', 'pass'], 30)
            try:
                owner.arm({})
                proc = owner.spawn([sys.executable, '-c', 'import time; time.sleep(60)'],
                                   d, dict(os.environ), Path(d) / 'log')
                p.close_process(proc)
                self.assertIsNotNone(proc.poll())
            finally:
                owner.finish()

    def test_ready_requires_both(self):
        with self.assertRaises(p.Stop):
            p.require_ready([{'model': p.MODEL, 'state': 'ready'}])
        p.require_ready([{'model': name, 'state': 'ready'} for name in (p.MODEL, 'pocket-tts')])

    def test_p95_is_nearest_rank_not_mean(self):
        self.assertEqual(p.percentile([1, 2, 10]), 10)
        self.assertIsNone(p.percentile([]))

    def test_error_detection(self):
        for msg in ('ERR native failure', 'cudaMalloc: out of memory', 'allocation failed', 'Traceback (most recent call last):'):
            self.assertRegex(msg, p.ERROR_RE)
        self.assertIsNone(p.ERROR_RE.search('drafts accepted 22 of 24, fallback_limits=0'))

    def test_artifact_atomic_and_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'result.json'
            p.record(out, {'ok': True})
            self.assertEqual(json.loads(out.read_text()), {'ok': True})
            self.assertFalse(out.with_suffix('.json.tmp').exists())
            with self.assertRaises(p.Stop):
                p.record(out, {'value': 'x' * (8*1024*1024)})
            self.assertEqual(json.loads(out.read_text()), {'ok': True})


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.installed = self.root / 'installed'
        self.runtime = self.root / 'runtime'
        self.trial = self.root / 'trial'
        for path in (self.installed / 'engine', self.installed / 'tools', self.installed / 'serve',
                     self.installed / '.venv/bin', self.runtime):
            path.mkdir(parents=True)
        (self.installed / '.venv/bin/python').symlink_to(sys.executable)
        for name in ('strata-nibbler', 'strata-vision'):
            shutil.copy2('/usr/bin/true', self.installed / 'engine' / name)
        # Minimal frontend dependencies: these are fakes, NOT actual model/server loading.
        (self.installed / 'serve/__init__.py').write_text('')
        (self.installed / 'serve/server.py').write_text(
            'import os\nclass ChatTemplate:\n def __init__(self, path): path.read_text()\n'
            'def child_env(cfg): return dict(os.environ)\n')
        (self.installed / 'tools/strata_tokenizer.py').write_text('')
        # copytree freezes only serve/tools; PIL fake therefore lives alongside serve.
        (self.installed / 'serve/PIL.py').write_text('Image = object\n')
        pack = self.root / 'qwen-nvfp4-gptq/pack'
        token = pack / 'tokenizer'
        token.mkdir(parents=True)
        for name in ('vocab.json', 'token_type.json'):
            (token / name).write_text('{}')
        for name in ('merges.txt', 'chat_template.jinja'):
            (token / name).write_text('fixture')
        payload = self.root / 'payload.gguf'
        payload.touch()
        cfg = json.loads(json.dumps(CFG))
        for flag in ('--native', '--native-dense-gguf', '--ple-gguf', '--embd-gguf', '--mtp', '--expert-profile'):
            cfg['args'][cfg['args'].index(flag) + 1] = str(payload)
        cfg['args'][cfg['args'].index('--pack') + 1] = str(pack)
        cfg['tokenizer'] = str(token)
        cfg['lib_dirs'] = [str(self.installed / 'engine')]
        cfg['vision'].update(mmproj=str(payload), model=str(payload))
        (self.runtime / 'config.json').write_text(json.dumps(cfg))
        self.launcher = self.root / 'runuser'
        self.policy = self.root / 'policy'
        self.policy.write_text('unlimited')
        # Fake PAM boundary. Positive infinity is explicitly synthetic; degraded mode
        # uses the disposable child's REAL limit, which is finite on this CPU VM.
        self.launcher.write_text(f'#!{sys.executable}\n'
            'import os,sys,resource,runpy,json\n'
            f'policy=open({str(self.policy)!r}).read()\n'
            'assert sys.argv[1:4] == ["-u", "native-inference", "--"]\n'
            'assert os.environ["PATH"] == "/usr/local/cuda-13.3/bin:/usr/bin:/bin"\n'
            'assert os.environ["HOME"] == os.getcwd() == os.environ["TMPDIR"]\n'
            'assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"\n'
            'sys.path.insert(0, os.environ["PYTHONPATH"] + "/serve")\n'
            'sys.path.insert(0, os.environ["PYTHONPATH"] + "/tools")\n'
            'if policy == "unlimited": resource.getrlimit=lambda _: (-1,-1)\n'
            'elif policy == "finite": resource.setrlimit(resource.RLIMIT_MEMLOCK, (0,0))\n'
            'elif policy == "missing-import": sys.path=[]\n'
            'sys.argv=sys.argv[5:]\n'
            'runpy.run_path(sys.argv[0], run_name="__main__")\n')
        self.launcher.chmod(0o755)
        self.ldd = '/usr/bin/ldd'
        self.native = SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid())
        self.previous_umask = os.umask(0o077)
        self.addCleanup(os.umask, self.previous_umask)

    def fake_live(self, tag='fixture', supervisor=None, set_limit_error=None, neutral=False,
                  fractions='baseline', kernel_error=None, running=None):
        rows = [{'model': p.MODEL, 'state': 'ready'}, {'model': 'pocket-tts', 'state': 'ready'}]
        def api(url, body=None, **_kw):
            if neutral:
                self.assertIsNone(body, 'neutral mode may only GET')
                if url.endswith('/running'):
                    return {'running': [dict(row, command='DO_NOT_EXPORT') for row in (running or rows)]}
                if url.endswith('/health'):
                    return {'status': 'ok', 'backend': 'cpu', 'extra': 'DO_NOT_EXPORT'}
                if url.endswith('/slots'):
                    return [{'is_processing': False, 'prompt': 'DO_NOT_EXPORT'} for _ in range(2)]
                if url.endswith('/metrics'):
                    return {'live': {'state': 'idle'}, 'engine': {'extra': 'DO_NOT_EXPORT'}}
                raise AssertionError('neutral control/private/model endpoint forbidden: ' + url)
            if '/api/models/unload/' in url:
                raise AssertionError('production mutation before preflight refusal')
            return {'status': 'ok'}
        def tool(name):
            return str(self.launcher) if name == 'runuser' else self.ldd if name == 'ldd' else '/usr/bin/' + name
        with contextlib.ExitStack() as stack:
            forbidden = []
            for name, value in (('ROOT', self.installed), ('RUNTIME', self.runtime), ('TRIAL', self.trial)):
                stack.enter_context(patch.object(p, name, value))
            stack.enter_context(patch.object(p.os, 'geteuid', return_value=0))
            stack.enter_context(patch.object(p.socket, 'gethostname', return_value='nibbler'))
            stack.enter_context(patch.object(p.os, 'chown'))
            stack.enter_context(patch.object(p.resource, 'setrlimit', side_effect=set_limit_error))
            stack.enter_context(patch.object(p.pwd, 'getpwnam', return_value=self.native))
            stack.enter_context(patch.object(p, 'resolved_tool', side_effect=tool))
            stack.enter_context(patch.object(p, 'memory', return_value={'MemAvailable': 20*p.GIB, 'SwapFree': 2*p.GIB}))
            stack.enter_context(patch.object(p, 'gpu', return_value={'used_mib': 0, 'free_mib': 16000}))
            stack.enter_context(patch.object(p, 'identities', return_value={'fixture': {'pid': 1, 'start_ticks': '1'}}))
            stack.enter_context(patch.object(p, 'production_hashes', return_value={'fixture': 'hash'}))
            stack.enter_context(patch.object(p, 'model_identity', return_value={'fixture': True}))
            if neutral:
                for name in ('request', 'unload_original', 'restore_original', 'wait_vram', 'Monitor'):
                    forbidden.append(stack.enter_context(patch.object(p, name, side_effect=AssertionError(name + ' forbidden in neutral mode'))))
                forbidden.append(stack.enter_context(patch.object(p.os, 'execv', side_effect=AssertionError('frontend exec forbidden'))))
                original_run = subprocess.run
                def no_private_exec(argv, **kwargs):
                    self.assertEqual(argv[0], str(self.launcher))
                    self.assertIn('--launch-check', argv)
                    self.assertNotIn('--exec', argv)
                    return original_run(argv, **kwargs)
                stack.enter_context(patch.object(p.subprocess, 'run', side_effect=no_private_exec))
                if kernel_error:
                    stack.enter_context(patch.object(life, 'probe_children', side_effect=kernel_error))
            else:
                stack.enter_context(patch.object(p, 'idle', return_value={'fixture': True}))
                stack.enter_context(patch.object(p, 'running_summary', return_value=rows))
            def host_query(argv, **_kwargs):
                self.assertIn(Path(argv[0]).name, ('nvidia-smi', 'lscpu'), 'private native command forbidden')
                return 'fixture'
            stack.enter_context(patch.object(p, 'command', side_effect=host_query if neutral else None, return_value='fixture'))
            calls = stack.enter_context(patch.object(p, 'http', side_effect=api))
            owners = stack.enter_context(patch.object(p, 'Supervisor', side_effect=supervisor or AssertionError('must refuse before supervisor')))
            with contextlib.redirect_stdout(io.StringIO()):
                rc = p.main(['--run', '--tag', tag, '--fractions', fractions, '--repeats', '1'] +
                            (['--preflight-only'] if neutral else []))
            for call in forbidden:
                call.assert_not_called()  # A swallowed assertion must never make a negative fixture pass.
            if neutral:
                owners.assert_not_called()
            return rc, calls, owners

    def assert_neutral_evidence(self, ready):
        out = self.trial / 'fixture'
        result = json.loads((out / 'result.json').read_text())
        self.assertEqual(result['mode'], 'preflight-only')
        self.assertEqual(result['ready'], ready)
        self.assertFalse(result['passed'])
        self.assertEqual(result['runs'], [])
        self.assertEqual(result['production_mutations'], 'none')
        for path in out.rglob('*.json'):
            self.assertNotIn('DO_NOT_EXPORT', path.read_text(), path)
            self.assertLessEqual(path.stat().st_size, 8*1024*1024)
        for name in ('lifecycle.json', 'unload-control.json', 'restoration.json', 'unloaded.json', 'samples.jsonl'):
            self.assertFalse((out / name).exists(), name)
        self.assertEqual(list(out.rglob('launch-readiness.json')), [])
        self.assertEqual(list(out.rglob('server.log')), [])
        return result

    def test_neutral_positive_prepares_every_arm_and_stops_without_guardian(self):
        before_config = (self.runtime / 'config.json').read_bytes()
        rc, calls, owners = self.fake_live(neutral=True, fractions='baseline,0.25,0.40')
        self.assertEqual(rc, 0)
        owners.assert_not_called()
        result = self.assert_neutral_evidence(True)
        self.assertTrue(result['production_unchanged'])
        self.assertEqual(len(result['launch_checks']), 3)
        for proof in result['launch_checks']:
            self.assertEqual(proof['memlock'], [-1, -1])
            self.assertEqual((proof['uid'], proof['gid']), (os.getuid(), os.getgid()))
            self.assertEqual(proof['kernel_children'], {'supported': True})
        self.assertEqual(len(list((self.trial / 'fixture').glob('run-*/config.json'))), 3)
        self.assertEqual((self.runtime / 'config.json').read_bytes(), before_config)
        self.assertTrue(all(call.args[0].endswith(('/running', '/slots', '/metrics', '/health'))
                            for call in calls.call_args_list))

    def test_neutral_child_limits_refusal_has_no_control_and_records_real_limits(self):
        self.policy.write_text('finite')
        rc, _, owners = self.fake_live(neutral=True)
        self.assertEqual(rc, 1)
        owners.assert_not_called()
        self.assert_neutral_evidence(False)
        proof = json.loads((self.trial / 'fixture/run-00/preflight.json').read_text())
        self.assertFalse(proof['ready'])
        self.assertEqual(proof['memlock'], [0, 0])
        self.assertEqual(proof['mode'], 'preflight-only')

    def test_neutral_missing_kernel_refuses_before_launch_checks_or_guardian(self):
        rc, _, owners = self.fake_live(neutral=True, kernel_error=life.LifecycleError('kernel children unavailable'))
        self.assertEqual(rc, 1)
        owners.assert_not_called()
        result = self.assert_neutral_evidence(False)
        self.assertIn('kernel-children', result['stop']['reason'])
        self.assertEqual(list((self.trial / 'fixture').glob('run-*')), [])

    def test_neutral_not_ready_refuses_before_upstream_get_could_autoload(self):
        rows = [{'model': p.MODEL, 'state': 'unloaded'}, {'model': 'pocket-tts', 'state': 'ready'}]
        rc, calls, owners = self.fake_live(neutral=True, running=rows)
        self.assertEqual(rc, 1)
        owners.assert_not_called()
        self.assert_neutral_evidence(False)
        self.assertEqual([call.args[0] for call in calls.call_args_list], ['http://127.0.0.1:8088/running'])

    def test_child_finite_memlock_refuses_before_unload_despite_parent_infinity(self):
        self.policy.write_text('finite')
        with patch.object(p.resource, 'getrlimit', return_value=(-1, -1)):
            rc, calls, owners = self.fake_live()
        self.assertEqual(rc, 1)
        result = json.loads((self.trial / 'fixture/result.json').read_text())
        self.assertIn('post-PAM memlock degraded: (0, 0)', result['stop']['reason'])
        owners.assert_not_called()
        self.assertFalse(any('/unload/' in call.args[0] for call in calls.call_args_list))
        self.assertFalse((self.trial / 'fixture/lifecycle.json').exists())

    def test_full_exact_environment_preflight_succeeds_without_model_then_stops_at_owner(self):
        def refused(*_args, **_kw):
            raise p.Stop('fixture refuses supervisor after successful no-model preflight')
        rc, calls, owners = self.fake_live(supervisor=refused)
        self.assertEqual(rc, 1)
        owners.assert_called_once()
        proof = json.loads((self.trial / 'fixture/run-00/preflight.json').read_text())
        self.assertEqual(proof['uid'], os.getuid())
        self.assertEqual(proof['gid'], os.getgid())
        self.assertEqual(proof['memlock'], [-1, -1])
        self.assertEqual(len(proof['dependencies']), 2)
        self.assertFalse((self.trial / 'fixture/run-00/launch-readiness.json').exists())
        self.assertFalse(any('/unload/' in call.args[0] for call in calls.call_args_list))

    def test_missing_python_binary_import_and_launcher_refuse_without_mutation(self):
        cases = ('python', 'binary', 'import', 'launcher')
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                # Restore prerequisites between cases, then remove exactly one.
                python = self.installed / '.venv/bin/python'
                if not python.exists():
                    python.symlink_to(sys.executable)
                shutil.copy2('/usr/bin/true', self.installed / 'engine/strata-nibbler')
                self.policy.write_text('unlimited')
                missing = None
                if case == 'python':
                    python.unlink()
                elif case == 'binary':
                    (self.installed / 'engine/strata-nibbler').unlink()
                elif case == 'import':
                    self.policy.write_text('missing-import')
                else:
                    missing = self.launcher.with_name('hidden-launcher')
                    self.launcher.rename(missing)
                try:
                    rc, calls, owners = self.fake_live(tag=f'case-{index}')
                    self.assertEqual(rc, 1)
                    owners.assert_not_called()
                    self.assertFalse(any('/unload/' in call.args[0] for call in calls.call_args_list))
                finally:
                    if missing:
                        missing.rename(self.launcher)

    def test_absolute_launcher_resolution_ignores_restricted_child_path(self):
        with patch.object(p.shutil, 'which', return_value='/usr/sbin/runuser') as which, \
             patch.object(Path, 'resolve', return_value=Path('/usr/sbin/runuser')):
            self.assertEqual(p.resolved_tool('runuser'), '/usr/sbin/runuser')
            which.assert_called_once_with('runuser', path='/usr/sbin:/usr/bin:/sbin:/bin')
        with patch.object(p.shutil, 'which', return_value=None), self.assertRaises(p.Stop):
            p.resolved_tool('runuser')

    def test_identity_and_actual_exec_memlock_refuse_before_reading_model_or_exec(self):
        for execute in (False, True):
            with patch.object(p.resource, 'getrlimit', return_value=(0, 0)), \
                 patch.object(Path, 'read_text', side_effect=AssertionError('model/config read')), \
                 patch.object(p.os, 'execv', side_effect=AssertionError('exec frontend')):
                with self.assertRaisesRegex(p.Stop, 'memlock degraded'):
                    p.launch_check('/no-model', os.getuid(), os.getgid(), '/usr/bin/ldd', execute)
            with self.assertRaisesRegex(p.Stop, 'identity mismatch'):
                p.launch_check('/no-model', os.getuid()+1, os.getgid(), '/usr/bin/ldd', execute)

    def test_config_paths_permissions_and_elf_dependencies_refuse_without_unload(self):
        payload = self.root / 'payload.gguf'
        for index, case in enumerate(('path', 'permission', 'elf', 'config')):
            with self.subTest(case=case):
                payload.touch()
                payload.chmod(0o600)
                self.ldd = '/usr/bin/ldd'
                if case == 'path':
                    payload.unlink()
                elif case == 'permission':
                    payload.chmod(0)
                elif case == 'elf':
                    fake = self.root / 'ldd'
                    fake.write_text('#!/bin/sh\nprintf "libfixture.so => not found\\n"\n')
                    fake.chmod(0o755)
                    self.ldd = str(fake)
                else:
                    config = json.loads((self.runtime / 'config.json').read_text())
                    config['parallel'] = 1
                    (self.runtime / 'config.json').write_text(json.dumps(config))
                    # This invariant is rejected even earlier, before creating trial output.
                    with self.assertRaisesRegex(p.Stop, 'baseline mismatch'):
                        self.fake_live(tag=f'path-{index}')
                    self.assertFalse((self.trial / f'path-{index}').exists())
                    continue
                rc, calls, owners = self.fake_live(tag=f'path-{index}')
                self.assertEqual(rc, 1)
                owners.assert_not_called()
                self.assertFalse(any('/unload/' in call.args[0] for call in calls.call_args_list))

    def test_unavailable_memlock_set_and_port_collision_refuse_before_owner(self):
        rc, calls, owners = self.fake_live(tag='limits', set_limit_error=PermissionError('no CAP_SYS_RESOURCE'))
        self.assertEqual(rc, 1)
        owners.assert_not_called()
        self.assertFalse(any('/unload/' in call.args[0] for call in calls.call_args_list))
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 18089))
            rc, calls, owners = self.fake_live()
        self.assertEqual(rc, 1)
        owners.assert_not_called()
        self.assertFalse(any('/unload/' in call.args[0] for call in calls.call_args_list))


class RestorationTests(unittest.TestCase):
    def test_idempotent_actual_restoration_entry_with_api_and_process_fakes(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / 'lifecycle.json'
            baseline = {'identities': {'router': 'same'}, 'hashes': {'engine': 'unchanged'}}
            state = {'baseline': baseline, 'restore_attempt': 1}
            ready = [{'model': p.MODEL, 'state': 'ready'}, {'model': 'pocket-tts', 'state': 'ready'}]
            with patch.object(p, 'recovery_guard', return_value=state), \
                 patch.object(p, 'identities', return_value=baseline['identities']), \
                 patch.object(p, 'production_hashes', return_value=baseline['hashes']), \
                 patch.object(p, 'running_summary', return_value=ready), \
                 patch.object(p, 'wait_vram', side_effect=AssertionError('already-ready model must not wait for empty GPU')), \
                 patch.object(p, 'memory', return_value={'MemAvailable': 20*p.GIB, 'SwapFree': 2*p.GIB}), \
                 patch.object(p, 'gpu', return_value={'used_mib': 14000}), \
                 patch.object(p, 'Monitor'), \
                 patch.object(p, 'request', return_value={'output_preview': 'CALIBRATION READY'}) as request, \
                 patch.object(p, 'http', return_value={'status': 'ok'}) as api:
                self.assertEqual(p.restore_original(journal), 0)
                self.assertEqual(p.restore_original(journal), 0)  # Same state/attempt: truly re-entrant.
            self.assertEqual(request.call_count, 2)
            self.assertFalse(any('/unload/' in c.args[0] for c in api.call_args_list))
            self.assertEqual(len(list(Path(directory).glob('restore-monitor-1-*'))), 2)
            self.assertTrue(json.loads((Path(directory) / 'restoration.json').read_text())['restored'])

    def test_restoration_identity_change_refuses_before_original_model_request(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {'baseline': {'identities': {'router': 'old'}, 'hashes': {}}, 'restore_attempt': 1}
            with patch.object(p, 'recovery_guard', return_value=state), \
                 patch.object(p, 'identities', return_value={'router': 'reused'}), \
                 patch.object(p, 'request', side_effect=AssertionError('must not load')) as request, \
                 patch.object(p, 'memory', return_value={}):
                self.assertEqual(p.restore_original(Path(directory) / 'journal.json'), 1)
                request.assert_not_called()


if __name__ == '__main__':
    unittest.main()

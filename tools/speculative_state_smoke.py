"""Model-level length/EOS commit and immediate live-prefix reuse regression.

Private engine only; dry-run by default. Requires a free GPU/model-loading window.
"""
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
from tools.conversation_cache_parity import load_tokenizer, state_hashes


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    if not args.run:
        print('Dry run: caps 1..8, EOS, consumed lengths and immediate live-prefix reuse.')
        return
    args.output.mkdir(parents=False, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    tokenizer = load_tokenizer(Path(cfg['tokenizer']))
    template = ChatTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
    def prompt(text):
        return tokenizer.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
    env = child_env(cfg)
    env.update(STRATA_STATE_HASH='1', STRATA_IQ_MT_MIN='1')
    engine_args = list(cfg['args']) + ['--prompt-cache', '6', '--conversation-cache-mib', '4096',
                                     '--adapt-swaps', '0', '--pcie-frac', '0', '--spec', '4',
                                     '--mtp-max-t', '4', '--spec-min-p', '0', '--suffix-draft', '0']
    log = args.output / 'engine.log'
    engine = StrataEngine(str(args.engine.resolve()), engine_args, cwd=cfg.get('cwd'), log=str(log), env=env)
    records = []
    def generate(ids, cap, name):
        outputs = [t for t in engine.generate(ids, cap, {'temperature': 0}, threading.Event()) if t is not None]
        records.append({'name': name, 'prompt_len': len(ids), 'cap': cap, 'ids': outputs, **engine.last})
        (args.output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        require(bool(outputs) and len(outputs) <= cap, name + ': missing/oversized output')
        return outputs
    try:
        # More than one fused chunk exercises tail balancing and recurrent/PLE
        # read-ahead before output-cap tests rewind the assistant checkpoint.
        ids = prompt('Read this data, then write the numbers 1 through 100 separated by spaces, without prose.\n' +
                     'alpha beta gamma delta epsilon zeta eta theta\n' * 900)
        for cap in range(1, 9):
            outputs = generate(ids, cap, f'cap-{cap}')
            require(len(outputs) == cap and engine.last['finish'] == 'length', f'cap-{cap}: fixture ended early')
            expected_live = len(ids) + len(outputs) - 1
            # The last output has not been consumed. Feeding it back as the
            # final prompt token must mount the entire live prefix, not rewind
            # to the assistant checkpoint because of hidden speculative inputs.
            generate(ids + outputs, 1, f'resume-{cap}')
            require(engine.last.get('reused') == expected_live, f'cap-{cap}: lost live-prefix reuse')
        generate(prompt('Reply with exactly OK and nothing else.'), 64, 'eos')
        require(engine.last['finish'] == 'stop', 'EOS fixture did not stop naturally')
    finally:
        engine.close()
    hashes = state_hashes(log.read_text())
    require(len(hashes) == len(records), 'missing state fingerprints')
    for record, state in zip(records, hashes):
        record['state'] = state
        require(int(state['L']) == record['prompt_len'] + len(record['ids']) - 1,
                record['name'] + ': persistent state advanced beyond returned tokens')
    (args.output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    print('PASS: caps 1..8, EOS, consumed lengths and immediate live-prefix reuse')


if __name__ == '__main__':
    main()

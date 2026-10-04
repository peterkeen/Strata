"""Small matched quality/performance gate against an idle private Strata HTTP server.

Never starts, stops or reconfigures a service. Use the same model/template/settings
for each arm. Not a comprehensive model-quality benchmark.
"""
import argparse
import base64
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib


def red_image():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    pixels = (b'\0' + b'\xff\0\0' * 64) * 64
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def json_answer(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


def correct_answer(text, expected):
    def matches(actual, wanted):
        # Python considers True == 1; JSON booleans are not numeric answers.
        if isinstance(actual, bool) != isinstance(wanted, bool):
            return False
        if isinstance(wanted, list):
            return isinstance(actual, list) and len(actual) == len(wanted) and all(
                matches(a, b) for a, b in zip(actual, wanted))
        return actual == wanted
    try:
        data = json_answer(text)
        return isinstance(data, dict) and set(data) == {'answer'} and matches(data['answer'], expected)
    except (ValueError, IndexError):
        return False


def fixtures():
    yield 'arithmetic', 'What is 17 * 23? Return only JSON with one key "answer" and a number.', 391
    yield 'python-trace', 'Evaluate [x*x for x in range(6) if x % 2]. Return only JSON with key "answer" and the resulting array.', [1, 9, 25]
    yield 'binary-search', ('For sorted [2, 4, 4, 7, 9], what zero-based insertion index does Python bisect_left return for 4? '
                            'Return only JSON with key "answer" and a number.'), 1
    yield 'sql-null', ('In SQL, is NULL = NULL true, false, or unknown? '
                      'Return only JSON with key "answer" and that lowercase string.'), 'unknown'
    yield 'json-extraction', ('From {"rows":[{"name":"Ada","score":7},{"name":"Lin","score":11}]} '
                             'return only JSON with key "answer" and the name of the highest-scoring person.'), 'Lin'
    yield 'french', 'Combien font douze plus quinze? Retournez uniquement un objet JSON avec la cle "answer" et le nombre.', 27
    yield 'chinese', '十加七等于多少？只返回 JSON 对象，键为 "answer"，值为数字。', 17
    code = '\n'.join(f'function rule{i}(x) {{ return x === {i} ? x + {i+1} : x - {i}; }}' for i in range(900))
    yield 'long-code', ('Find rule731 in this JavaScript source and compute rule731(731). '
                        'Return only JSON with key "answer" and a number.\n' + code), 1463
    tail_code = '\n'.join(f'function rule{i}(x) {{ return x === {i} ? x + {i+1} : x - {i}; }}' for i in range(280))
    yield 'long-code-tail', ('Find rule231 in this JavaScript source and compute rule231(231). '
                             'Return only JSON with key "answer" and a number.\n' + tail_code), 463
    yield 'vision', [
        {'type': 'image_url', 'image_url': {'url': red_image()}},
        {'type': 'text', 'text': 'What is the dominant color of this image? Return only JSON with key "answer" and a lowercase English color name.'}], 'red'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:5802')
    ap.add_argument('--model', default='qwen3.8-flash-next-iq3_s')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--skip-vision', action='store_true')
    ap.add_argument('--baseline', type=Path, help='matched results: fail on new failures; report existing baseline failures explicitly')
    ap.add_argument('--fixture', action='append', help='run only named quality fixtures (repeatable)')
    ap.add_argument('--skip-performance', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('output exists; use a new results file')
    results = []

    def request(name, content, count=256):
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': content}],
                'max_tokens': count, 'temperature': 0, 'reasoning_effort': 'none'}
        req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as response:
            data = json.load(response)
        record = {'arm': args.label, 'name': name, 'wall_s': time.monotonic() - start,
                  'usage': data.get('usage'), 'timings': data.get('timings'),
                  'choice': data['choices'][0]}
        results.append(record)
        return record

    for name, content, expected in fixtures():
        if (args.skip_vision and name == 'vision') or (args.fixture and name not in args.fixture):
            continue
        record = request(name, content)
        record['passed'] = correct_answer(record['choice']['message'].get('content') or '', expected)
        record['expected'] = expected
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)

    # Alternate branches to prohibit cached same-prefix repeats. Reproduce the
    # profiled short-remainder shape, while counting engine time rather than TTFT.
    for words in (() if args.skip_performance else (900, 1800)):
        for rep in range(2):
            for branch in ('A', 'B'):
                line = ('alpha beta gamma delta epsilon zeta eta theta\n' if branch == 'A'
                        else 'copper silver iron zinc nickel gold tin lead\n')
                content = f'Stream {branch}. Read this data and respond with OK.\n' + line * words
                record = request(f'prefill-{words}-{rep}-{branch}', content, 8)
                args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
                print(json.dumps(record, ensure_ascii=False), flush=True)
    quality = {r['name']: r for r in results if 'passed' in r}
    if args.fixture and set(args.fixture) != quality.keys():
        raise SystemExit('QUALITY GATE FAILED: unknown or skipped fixture')
    failed = [name for name, r in quality.items() if not r['passed']]
    if args.baseline:
        baseline = {r['name']: r for r in json.loads(args.baseline.read_text()) if 'passed' in r}
        if quality.keys() != baseline.keys() or any(quality[n]['expected'] != baseline[n]['expected'] for n in quality):
            raise SystemExit('QUALITY GATE FAILED: baseline fixtures do not match')
        old_failures = [name for name, r in baseline.items() if not r['passed']]
        failed = [name for name in failed if baseline[name]['passed']]
        print('Existing baseline failures (not counted as passes): ' + ', '.join(old_failures), flush=True)
    if failed:
        raise SystemExit('QUALITY GATE FAILED: ' + ', '.join(failed))
    print(f'PASS: {sum(r["passed"] for r in quality.values())}/{len(quality)} correct; '
          'no new quality failures; performance recorded separately', flush=True)


if __name__ == '__main__':
    main()

"""Brick b's gate: Minerva's greedy ids, teacher-forced through bitnet.cpp.

    python tools/conformance.py [--minerva EXE] [--server EXE] [--gguf F] [--prompts DIR]
                                [--sizes 64,256,512,1024] [--new 48] [--out results.json]
                                [--lora ADAPTER.gguf] [--ids-files A.ids,B.ids]

For each bench prompt (bitnet-bench `prompts/pN.ids`: BOS + N-1 ids, so exactly N ids):

1. **Ours**: `minerva generate --ids pN.ids --new NEW`: greedy ids, EOS not a stop.
2. **The reference, teacher-forced**: bitnet.cpp's `llama-server` (pinned `765741d`, CPU),
   asked at each step of our ids for its own next-token distribution (n_probs 256: this 2024
   server softmaxes among the top n_probs only, and top-1 alone would read probability 1).

A step agrees when the reference's argmax is our token; where it is not, the gap is the
reference's probability of its argmax less that of ours. **Gate: every parting is a near-tie,
worst gap < 0.1** (rusty_sloth's TIE for I2_S). Token COUNTS must match: N in, NEW out.

With `--lora`, both engines load the same adapter (Minerva `--lora`, the server `--lora`):
the gate is then on the adapted model.

The comparison itself is rusty_sloth's (`tools/conformance_matrix.py`: `step`, `teacher_forced`,
`vocabulary`), imported, not copied, so both engines face the same judge. Only the server
launcher is ours: theirs fixes `-c 1024`, and p1024 plus NEW needs more.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

sys.path.insert(0, 'F:/coding/rusty_sloth/tools')
import conformance_matrix as cm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TIE = cm.TIE['i2_s']


class Server:
    """bitnet.cpp's server on the CPU, `-c` sized to the longest prompt plus NEW."""

    def __init__(self, exe, gguf, ctx, port, log_dir, lora=None):
        self.port = port
        self.log = open(os.path.join(log_dir, f'server-{port}.log'), 'w')
        cmd = [exe, '-m', gguf, '--port', str(port), '-ngl', '0', '-c', str(ctx), '-np', '1']
        if lora:
            cmd += ['--lora', lora]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self.log)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit(f'server exited ({self.proc.returncode}); see {self.log.name}')
            try:
                if json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2)).get('status') == 'ok':
                    return
            except Exception:
                time.sleep(0.5)
        self.close()
        raise SystemExit('server did not come up')

    def post(self, path, body):
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}{path}', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=1200))

    def close(self):
        self.proc.kill()
        self.proc.wait()
        self.log.close()


LORA = []


def ours(exe, gguf, ids_file, new, work, n):
    probs = os.path.join(work, f'ours-{n}.probs.jsonl')
    r = subprocess.run([exe, 'generate', '--gguf', gguf, '--ids', ids_file, '--new', str(new), '--probs', probs] + LORA,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'minerva failed on {n}:\n{r.stderr[-2000:]}')
    ids = [int(x) for x in r.stdout.strip().splitlines()[-1].split(',') if x]
    timing = next(l for l in r.stderr.splitlines() if l.startswith('minerva: load'))
    return ids, timing, probs


def ours_screened(exe, gguf, ids_file, new):
    """The same prompt through the head screen (brick d): the ids must be the full head's."""
    env = dict(os.environ, MINERVA_HEAD='screen')
    r = subprocess.run([exe, 'generate', '--gguf', gguf, '--ids', ids_file, '--new', str(new)] + LORA,
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit('minerva (screen) failed:\n' + r.stderr[-2000:])
    screen = next((l for l in r.stderr.splitlines() if l.startswith('screen:')), '')
    return [int(x) for x in r.stdout.strip().splitlines()[-1].split(',') if x], screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minerva', default=os.path.join(HERE, '..', 'target', 'release', 'minerva.exe'))
    ap.add_argument('--server', default='F:/coding/bitnet-bench/src/bitnet-win/build/bin/Release/llama-server.exe')
    ap.add_argument('--gguf', default='F:/coding/bitnet-golden/ours-i2_s.gguf')
    ap.add_argument('--prompts', default='F:/coding/bitnet-bench/prompts')
    ap.add_argument('--sizes', default='64,256,512,1024')
    ap.add_argument('--new', type=int, default=48)
    ap.add_argument('--port', type=int, default=8791)
    ap.add_argument('--work', default=os.path.join(HERE, '..', 'target', 'conformance'))
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'docs', 'conformance-b.json'))
    ap.add_argument('--poison', action='store_true',
                    help='negative control: rotate our ids by one before judging; the gate must FAIL')
    ap.add_argument('--lora', help='adapter GGUF loaded by both engines')
    ap.add_argument('--ids-files', help='comma-separated id files instead of --prompts/--sizes (named by their stem)')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.lora:
        LORA.extend(['--lora', a.lora])
    if a.ids_files:
        files = {os.path.splitext(os.path.basename(f))[0]: f for f in a.ids_files.split(',')}
    else:
        files = {f'p{n}': os.path.join(a.prompts, f'p{n}.ids') for n in [int(s) for s in a.sizes.split(',')]}
    sizes = list(files)

    # 1. Ours first, alone (the host's free RAM holds one 1.2 GB model comfortably, not two).
    runs = {}
    for n in sizes:
        f = files[n]
        prompt = [int(x) for x in open(f).read().replace(',', ' ').split()]
        if n[0] == 'p' and n[1:].isdigit():
            assert len(prompt) == int(n[1:]), f'{n}.ids holds {len(prompt)} ids'
        mine, timing, probs = ours(a.minerva, a.gguf, f, a.new, a.work, n)
        assert len(mine) == a.new, f'{n}: {len(mine)} ids out, want {a.new}'
        screened, screen = ours_screened(a.minerva, a.gguf, f, a.new)
        if screened != mine:
            raise SystemExit(f'{n}: the head screen chose other ids than the full head')
        print(f'{n}: head screen ids == full head ids | {screen}', flush=True)
        if a.poison:
            mine = mine[1:] + mine[:1]
        print(f'{n}: ours {len(mine)} ids | {timing}', flush=True)
        runs[n] = (prompt, mine, probs)

    # 2. The reference, teacher-forced along our ids.
    piece = cm.vocabulary(a.gguf)
    longest = max(len(runs[n][0]) for n in sizes)
    server = Server(a.server, a.gguf, longest + a.new + 16, a.port, a.work, a.lora)
    rows, ok = [], True
    try:
        for n in sizes:
            prompt, mine, probs = runs[n]
            t0 = time.time()
            agree, worst, parts = cm.teacher_forced(server, prompt, mine, True, piece)
            passed = worst < TIE
            ok &= passed
            ours_top = [json.loads(l)['top'] for l in open(probs)]
            for p in parts:
                p['ours_top2'] = ours_top[p['step']][:2]
            rows.append({'prompt': f'{n}', 'in': len(prompt), 'out': len(mine), 'agree': agree, 'steps': len(mine),
                         'worst_gap': worst, 'pass': passed, 'parts': parts, 'ours': mine})
            print(f'{n}: {agree}/{len(mine)} agree, worst gap {worst:.4f} -> {"PASS" if passed else "FAIL"} '
                  f'({time.time() - t0:.0f}s)', flush=True)
            for p in parts:
                print(f'    step {p["step"]}: ours {p["ours"]} (cpp p={p["p_ours"]}) vs cpp {p["runtime"]!r} '
                      f'p={p["p_runtime"]:.4f}; our top-2 {p["ours_top2"]}')
    finally:
        server.close()

    json.dump({'gate': f'every parting a near-tie: worst gap < {TIE}', 'pass': ok, 'model': a.gguf,
               'reference': f'bitnet.cpp 765741d llama-server {a.server} (CPU), n_probs 256', 'lora': a.lora,
               'new': a.new, 'rows': rows}, open(a.out, 'w'), indent=1)
    print(('PASS' if ok else 'FAIL') + f' -> {a.out}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

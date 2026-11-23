"""Cold-cache load A/B: a memory map against a streamed read, each on a file the OS has not cached.

    python tools/cold_ab.py [--rounds 3] [--arms map,read] [--ids F] [--new 16]

Every run gets a FRESH copy of the model made by `robocopy /J` (unbuffered I/O: the copy's pages
never enter the OS file cache), so its first run is cold. The same copy is then run again
(warm): the control that proves the copy really was cold. Arms alternate (ABBA) round to round.
The copy lives under target/cold (gitignored) on the same drive as the model.

Reported per run, from the `minerva:` line: load seconds and prefill+first seconds (the first
token is where a map's page faults land), plus decode tok/s.
"""
import argparse, os, re, shutil, statistics, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r'load ([\d.]+)s \| prefill \d+ tokens \+ first: ([\d.]+)s \| decode \d+ tokens: [\d.]+s \(([\d.]+) tok/s\)')


def fresh_copy(src, dst_dir):
    shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir)
    r = subprocess.run(['robocopy', os.path.dirname(src), dst_dir, os.path.basename(src), '/J', '/NJH', '/NJS', '/NP'],
                       capture_output=True, text=True)
    if r.returncode >= 8:
        raise SystemExit(f'robocopy failed ({r.returncode}): {r.stdout[-500:]}')
    return os.path.join(dst_dir, os.path.basename(src))


def run(exe, gguf, ids, new, arm):
    env = dict(os.environ, MINERVA_LOAD=arm, MINERVA_HEAD='screen')
    r = subprocess.run([exe, 'generate', '--gguf', gguf, '--ids', ids, '--new', str(new)], capture_output=True, text=True, env=env)
    m = LINE.search(r.stderr)
    if r.returncode != 0 or not m:
        raise SystemExit(f'{arm}: failed\n{r.stderr[-800:]}')
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), r.stdout.strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=os.path.join(HERE, '..', 'target', 'release', 'minerva.exe'))
    ap.add_argument('--gguf', default='F:/coding/bitnet-golden/ours-i2_s.gguf')
    ap.add_argument('--ids', default='F:/coding/bitnet-bench/prompts/p64.ids')
    ap.add_argument('--new', type=int, default=16)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--arms', default='map,read')
    a = ap.parse_args()
    arms = a.arms.split(',')
    cold_dir = os.path.join(HERE, '..', 'target', 'cold')
    res = {arm: {'cold': [], 'warm': []} for arm in arms}
    ids_seen = set()
    for r in range(a.rounds):
        order = arms if r % 2 == 0 else arms[::-1]
        for arm in order:
            g = fresh_copy(a.gguf, cold_dir)
            for kind in ('cold', 'warm'):
                load, first, tps, ids = run(a.exe, g, a.ids, a.new, arm)
                res[arm][kind].append((load, first, tps))
                ids_seen.add(ids)
                print(f'round {r + 1} {arm:<5} {kind}: load {load:6.2f}s  prefill+first {first:7.2f}s  decode {tps:6.2f} tok/s', flush=True)
    shutil.rmtree(cold_dir, ignore_errors=True)
    print(f'\nmethod: fresh robocopy /J copy per run (cold), then the same copy again (warm); {a.rounds} rounds, '
          f'arms alternate; prompt {os.path.basename(a.ids)} + {a.new}; work parity: {len(ids_seen)} distinct id sequence(s)')
    print(f'{"arm":<6} {"kind":<5} {"load s med":>11} {"prefill+first s med":>20} {"load+first med":>15} {"decode tok/s med":>17}')
    for arm in arms:
        for kind in ('cold', 'warm'):
            v = res[arm][kind]
            med = lambda i: statistics.median(x[i] for x in v)
            tot = statistics.median(x[0] + x[1] for x in v)
            print(f'{arm:<6} {kind:<5} {med(0):>11.2f} {med(1):>20.2f} {tot:>15.2f} {med(2):>17.2f}')


if __name__ == '__main__':
    main()

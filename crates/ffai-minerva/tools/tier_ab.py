"""Minerva's A/B instrument: the same binary with one env var varied per arm, interleaved.

    python tools/tier_ab.py [--rounds 10] [--ids F] [--new 64] [--arms scalar,ssse3,avx2,avx2]
    python tools/tier_ab.py --var MINERVA_HEAD --arms full,screen,screen

Method (codec-measurement):
- ONE binary; the arm is an env var (`--var`, default `MINERVA_ISA`), read once per process. A
  repeated arm name is the NULL arm (the same code twice), which sets the noise floor.
- Arms rotate every round (round r runs the list rotated by r, reversed on odd r), so no arm is
  always first or last.
- The clock is each run's own internal timer (decode tok/s and prefill seconds from the
  `minerva:` line): no process launch, no model load.
- Work parity: every run must print the same token ids, and runs of one arm the same
  `logits-fnv` (a full head hashes every logit, a screened head its tokens); else the table is VOID.
- Paired ratios per round against the first arm, with the win count and z = (wins - N/2) / (0.5 sqrt N).
"""
import argparse, math, os, re, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r'prefill (\d+) tokens \+ first: ([\d.]+)s \| decode (\d+) tokens: ([\d.]+)s \(([\d.]+) tok/s\) '
                  r'\| threads (\d+) \| isa (\w+) \| head (\w+) \| logits-fnv ([0-9a-f]+)')


def run(exe, ids, new, var, value, threads):
    env = dict(os.environ, **{var: value})
    cmd = [exe, 'generate', '--ids', ids, '--new', str(new)] + (['--threads', str(threads)] if threads else [])
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f'{value}: exit {r.returncode}\n{r.stderr[-1500:]}')
    m = LINE.search(r.stderr)
    ran = {'MINERVA_ISA': m.group(7), 'MINERVA_HEAD': m.group(8).lower()}.get(var, value) if m else None
    if ran != value:
        raise SystemExit(f'{value}: no timing line, or another arm ran ({ran}):\n{r.stderr[-800:]}')
    return {'prefill_s': float(m.group(2)), 'decode_tps': float(m.group(5)), 'fnv': m.group(9),
            'threads': int(m.group(6)), 'ids': r.stdout.strip().splitlines()[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=os.path.join(HERE, '..', 'target', 'release', 'minerva.exe'))
    ap.add_argument('--ids', default='F:/coding/bitnet-bench/prompts/p64.ids')
    ap.add_argument('--new', type=int, default=64)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--var', default='MINERVA_ISA')
    ap.add_argument('--threads', type=int, default=0, help='Minerva --threads (0: the pool default)')
    ap.add_argument('--arms', default='scalar,ssse3,avx2,avx2')
    a = ap.parse_args()
    names = a.arms.split(',')
    labels = [n if names[:i].count(n) == 0 else f'{n}#{names[:i].count(n) + 1}' for i, n in enumerate(names)]
    res = {l: [] for l in labels}
    for r in range(a.rounds):
        order = list(range(len(names)))
        order = order[r % len(order):] + order[:r % len(order)]
        if r % 2:
            order.reverse()
        for i in order:
            res[labels[i]].append(run(a.exe, a.ids, a.new, a.var, names[i], a.threads))
        print(f'round {r + 1}/{a.rounds}: ' + '  '.join(f'{labels[i]} {res[labels[i]][-1]["decode_tps"]:.2f}' for i in order),
              flush=True)

    fnv_ok = all(len({x['fnv'] for l in labels if l.split('#')[0] == n for x in res[l]}) == 1 for n in set(names))
    idss = {x['ids'] for l in labels for x in res[l]}
    ok = fnv_ok and len(idss) == 1
    print(f'\nmethod: one binary, {a.var} per arm, {a.rounds} rotated rounds, internal clock, '
          f'threads {res[labels[0]][0]["threads"]}, prompt {os.path.basename(a.ids)} + {a.new} new')
    print(f'work parity: logits-fnv {"equal within every arm" if fnv_ok else "DIFFERS within an arm"}, '
          f'{len(idss)} distinct id sequence(s) -> {"OK" if ok else "VOID"}')
    if not ok:
        sys.exit(1)
    base = labels[0]
    print(f'\n{"arm":<10} {"decode tok/s median":>20} {"min":>7} {"max":>7} {"prefill s med":>14}  vs {base}: ratio med  wins  z')
    for l in labels:
        d = [x['decode_tps'] for x in res[l]]
        p = [x['prefill_s'] for x in res[l]]
        ratios = [x / y for x, y in zip(d, (x['decode_tps'] for x in res[base]))]
        wins = sum(q > 1 for q in ratios)
        z = (wins - len(ratios) / 2) / (0.5 * math.sqrt(len(ratios)))
        print(f'{l:<10} {statistics.median(d):>20.2f} {min(d):>7.2f} {max(d):>7.2f} {statistics.median(p):>14.2f}  '
              f'{statistics.median(ratios):>12.3f}x {wins:>3}/{len(ratios)} {z:+.2f}')
    print(f'\nprefill+first, paired: speedup = {base} seconds / arm seconds (> 1: the arm is faster)')
    for l in labels:
        sp = [y['prefill_s'] / x['prefill_s'] for x, y in zip(res[l], res[base])]
        wins = sum(q > 1 for q in sp)
        z = (wins - len(sp) / 2) / (0.5 * math.sqrt(len(sp)))
        print(f'{l:<10} speedup median {statistics.median(sp):.3f}x  range {min(sp):.3f}-{max(sp):.3f}  wins {wins}/{len(sp)}  z {z:+.2f}')
    for n in [l for l in labels if '#' in l]:
        first = n.split('#')[0]
        q = [x['decode_tps'] / y['decode_tps'] for x, y in zip(res[n], res[first])]
        print(f'null arm {n} / {first}: median {statistics.median(q):.3f}, range {min(q):.3f}-{max(q):.3f} '
              f'(the floor any difference must clear)')


if __name__ == '__main__':
    main()

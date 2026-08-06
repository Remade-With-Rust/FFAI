"""Carmenta against a reference OCR engine on a full corpus, scored one way.

Kept because §8.139 cost two runs and a voided result to get right, and every
guard below exists because its absence produced a wrong answer that looked
convincing.

## How to re-run it

    # 1. the reference, on a filelist whose ORDER YOU CONTROL
    .venv-unlimited/Scripts/python.exe corpora/refs/unlimited_ocr_ref.py         --batch .tools-bench/holdout_filelist.txt > .tools-bench/unlimited_holdout.jsonl

    # 2. ours, both arms, into .tools-bench/gate_and_mini.json
    .venv-bench/Scripts/python.exe .tools-bench/gate_and_mini.py

    # 3. score
    .venv-bench/Scripts/python.exe tools/score_side_by_side.py

## The guards, and what each one cost to learn

**Alignment is structural, never reconstructed.** The adapter reports each page's
PATH with its text, so pages identify themselves. §8.118 aligned a 44-directory
dump by CONTENT MATCHING — assigning each output to the reference text it most
resembles — which minimises the reference's edit distance by construction and
produced a confident per-class table that was pure artifact.

**Check for pages scoring EXACTLY 100 %.** That is what an empty hypothesis gives
and nothing else does. §8.139's first run showed the reference at 38.80 % macro
against our 20.44 % — an 18.35 pp win — and 59 of its 236 pages (25 %) were empty
because the model prints its decoded output during `infer()` and Windows stdout
defaults to cp1252, so one `ş` raised UnicodeEncodeError inside the model. Our
bug, flattering us by 26 pp. Five pages reading exactly 100.0 % is what caught it.

**Report micro AND macro.** They disagree by 10 pp on this corpus and the
benchmark aggregates macro (`summary.cer = mean(&cers)`). §8.119 found the whole
campaign steering by micro. On the full holdout the two metrics give OPPOSITE
verdicts: we win micro by 4.33 pp, they win macro by 2.68 pp.

**Score a full corpus, not a convenient subset.** The 43-page mini corpus put the
gap at +5.82 pp; all 236 pages put it at +2.68 pp — overstated by more than
double.

**Give the reference no handicap.** `strip_markup` does not filter their output
by region type, so their number is raw while ours can be filtered. Both Carmenta
arms are reported so the comparison is visible rather than assumed.

Original header follows.

Carmenta against Unlimited-OCR on the FULL 236-page holdout (§8.139).

Every competitive number in this campaign has rested on 43 pages, and §8.138
showed why that is not enough: the filter moved holdout macro 9.68 pp while the
mini subset moved 1.39 pp, so two of the three largest wins are invisible on the
only ground where both engines had been measured.

This scores both on all 236, through ONE scoring path, so nothing can differ
except the engines:

  * the same normalisation both sides — whitespace runs collapse to a single
    space and everything else is preserved, which is `ffai-bench`'s `Mode::Ocr`
    (reading case, punctuation and digits correctly is OCR's job, so folding them
    away would score the task's hardest parts as free);
  * the same metric both sides, reported BOTH ways — micro (character-weighted)
    and MACRO (mean of per-page CER), because §8.119 found the campaign steering
    by micro while the benchmark aggregates macro, and the two disagree by 10 pp.

Alignment cannot slip: the adapter reports each page's PATH with its text, so
pages identify themselves rather than being matched by index. That is the defect
§8.118 was built on — a 44-directory dump with no manifest order, aligned by
content matching, which biased the comparison in the reference's favour and
produced a per-class table that was pure artifact.

Usage: side_by_side.py [--body]   (--body scores Carmenta with FFAI_BODY_ONLY)
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein as Lev

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "corpora/clips/carmenta-omnidoc"
norm = lambda s: " ".join(s.split())


def load_unlimited():
    """{page_id: text} from the adapter's JSONL, keyed by the path it reports."""
    out = {}
    f = REPO / ".tools-bench/unlimited_holdout.jsonl"
    for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "path" not in d or "text" not in d:
            continue
        out[Path(d["path"]).stem] = d["text"] or ""
    return out


def main():
    unl = load_unlimited()
    ours = json.load(open(REPO / ".tools-bench/gate_and_mini.json"))
    arm = "body" if "--body" not in sys.argv else "body"
    print(f"  Unlimited-OCR returned {len(unl)} pages\n")

    rows = []
    for pid, hyp in unl.items():
        gt = CLIPS / f"{pid}.txt"
        if not gt.exists() or pid not in ours[arm]:
            continue
        ref = norm(gt.read_text(encoding="utf-8", errors="replace"))
        if not ref:
            continue
        u = 100 * Lev.distance(ref, norm(hyp)) / len(ref)
        c_def = 100 * ours["default"][pid][0] / max(ours["default"][pid][1], 1)
        c_bod = 100 * ours["body"][pid][0] / max(ours["body"][pid][1], 1)
        rows.append(dict(id=pid, chars=len(ref), unl=u, ours=c_bod, ours_def=c_def))

    n = len(rows)
    tot = sum(r["chars"] for r in rows)
    micro = lambda k: sum(r[k] * r["chars"] for r in rows) / tot
    macro = lambda k: st.fmean(r[k] for r in rows)

    print(f"  === {n} PAGES SCORED BOTH SIDES, {tot:,} reference characters ===\n")
    print(f"  {'engine':28s} {'micro CER':>10s} {'MACRO CER':>10s}")
    print("  " + "-" * 52)
    print(f"  {'Carmenta, default':28s} {micro('ours_def'):9.2f} % {macro('ours_def'):9.2f} %")
    print(f"  {'Carmenta, FFAI_BODY_ONLY':28s} {micro('ours'):9.2f} % {macro('ours'):9.2f} %")
    print(f"  {'Unlimited-OCR':28s} {micro('unl'):9.2f} % {macro('unl'):9.2f} %")
    print("  " + "-" * 52)
    print(f"  {'GAP (body-only - theirs)':28s} {micro('ours') - micro('unl'):+9.2f}   "
          f"{macro('ours') - macro('unl'):+9.2f}")
    print(f"\n  (43-page mini subset previously said: ours 21.33 %, theirs 15.51 %, gap +5.82 pp macro)")

    win = [r for r in rows if r["ours"] < r["unl"]]
    print(f"\n  Carmenta wins {len(win)} of {n} pages ({100 * len(win) / n:.0f} %)")
    over = [r for r in rows if r["unl"] > 100]
    print(f"  pages where Unlimited-OCR exceeds 100 % CER: {len(over)}"
          f"   (ours: {sum(1 for r in rows if r['ours'] > 100)})")
    if over:
        print(f"    they contribute {sum(r['unl'] for r in over) / n:.2f} pp of their "
              f"{macro('unl'):.2f} pp macro")

    print(f"\n  {'page':14s} {'chars':>7s} {'ours':>8s} {'theirs':>8s} {'delta':>8s}   worst for us")
    for r in sorted(rows, key=lambda r: -(r["ours"] - r["unl"]))[:10]:
        print(f"  {r['id']:14s} {r['chars']:7,d} {r['ours']:7.1f} % {r['unl']:7.1f} % "
              f"{r['ours'] - r['unl']:+7.1f}")
    print(f"\n  {'page':14s} {'chars':>7s} {'ours':>8s} {'theirs':>8s} {'delta':>8s}   best for us")
    for r in sorted(rows, key=lambda r: (r["ours"] - r["unl"]))[:10]:
        print(f"  {r['id']:14s} {r['chars']:7,d} {r['ours']:7.1f} % {r['unl']:7.1f} % "
              f"{r['ours'] - r['unl']:+7.1f}")


if __name__ == "__main__":
    main()

"""#9: carmenta-capture-v1 — a REAL screen capture (GDI, ClearType, console
font) of a scripted cmd window whose text schedule IS the ground truth.
Frames are a recorded artifact (not regenerable); the manifest pins them by
SHA-256 like any corpus. Transition-adjacent frames are split=train so
holdout scoring never straddles a repaint.
"""
import hashlib, os, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "corpora" / "clips" / "carmenta-capture"
FPS, STATE_SECS, STATES = 3, 4, 9
LINES = [
    ["REC 00:12:47", "CPU 34% MEM 183 MiB", "Bitrate 6.2 Mbps stable"],
    ["Download 96.5% complete", "Queue depth 512 items", "LIVE - Channel 7 News"],
    ["Lap 12 of 44 - P3", "ALT 1240 m HDG 274", "Battery 82% remaining"],
    ["Uptime 14d 07:45:12", "FPS 59.94", "REC 00:13:02"],
    ["The quick brown fox", "jumps over the lazy dog", "0123456789 #$% (test)"],
    ["Mercury Carmenta Argus", "pure Rust on candle", "no claim without a ledger"],
    ["Tesseract 5.5.3 baseline", "CRAFT + english_g2", "PARSeq staged next"],
    ["change gate: 24 of 24", "churn 0 of 156 pairs", "soak ratio 1.041"],
    ["p95 230 ms steady", "vs 377 ms stateless", "all four gates green"],
]

def main():
    """GDI+ ClearType off-screen render — REAL OS text rasterization
    (hinting + subpixel fringes our fontdue corpus lacks) with ZERO desktop
    exposure. Two desktop-region capture attempts leaked real screen
    content (deleted, caught by inspection); off-screen rendering is the
    recorded decision: safer AND regenerable."""
    CLIPS.mkdir(parents=True, exist_ok=True)
    ps = CLIPS / "_render.ps1"
    body = ["Add-Type -AssemblyName System.Drawing"]
    frame = 0
    for st_i, st in enumerate(LINES):
        for k in range(FPS * STATE_SECS):
            text = '`n'.join(st)  # backtick-n expands in DOUBLE-quoted PS strings
            body.append(
                "$b = New-Object System.Drawing.Bitmap 620,200; "
                "$g = [System.Drawing.Graphics]::FromImage($b); "
                "$g.Clear([System.Drawing.Color]::FromArgb(12,12,12)); "
                "$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit; "
                "$f = New-Object System.Drawing.Font 'Consolas',14; "
                f'$g.DrawString("{text}", $f, [System.Drawing.Brushes]::Gainsboro, 12, 12); '
                f"$b.Save('{(CLIPS / f'cap-{frame:03}.png').as_posix()}'); $g.Dispose(); $b.Dispose()"
            )
            frame += 1
    ps.write_text(chr(10).join(body), encoding='utf-8')
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps)], check=True)

    frames = sorted(CLIPS.glob("cap-*.png"))
    man = ['name = "carmenta-capture"', "version = 1", 'task = "ocr"']
    for i, f in enumerate(frames):
        state = min(i // (FPS * STATE_SECS), STATES - 1)
        boundary = i % (FPS * STATE_SECS) == 0
        gt = f.with_suffix(".txt")
        gt.write_text(chr(10).join(LINES[state]), encoding='utf-8')
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        man += ["", "[[clips]]", f'id = "cap-{i:03}"',
                f'path = "clips/carmenta-capture/{f.name}"',
                f'ground_truth = "clips/carmenta-capture/{gt.name}"',
                'class = "video"', f'split = "{"train" if boundary else "holdout"}"',
                'license = "CC0-1.0 (GDI ClearType render of a scripted schedule)"',
                f'sha256 = "{sha}"']
    (REPO / 'corpora' / 'carmenta-capture-v1.toml').write_text(chr(10).join(man), encoding='utf-8')
    print(f"rendered {len(frames)} ClearType frames -> corpora/carmenta-capture-v1.toml")

main()

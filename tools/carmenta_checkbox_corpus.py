"""carmenta-checkbox-v1: form rows with checkboxes, ground truth by construction.

Built for docs/plans/commercial-gaps.md gaps 4 and 14: a reader returned
checkbox marks as characters ("@", "0", "D", "B") at text confidence, so a
form whose value is a set of ticks could not be read.

Each clip is a form fragment: question rows ending in "[ ] Yes  [ ] No"
(and some three-way rows), plus DISTRACTORS a naive square-finder would take
for boxes: the letters O, D, 0, 8 and the ideograph 口 in running text, a
small table, and an underline rule.

Boxes are drawn two ways (a stroked rectangle; the font's own ballot-box
glyph) and marked four ways (X strokes; a tick that overflows the box; a
solid fill; the font's ballot-box-with-X glyph), or left empty. Every clip is
rendered anti-aliased gray and 1-bit (a CCITT-style scan). Ground truth, in a
.json sidecar, is every box's frame and whether it is marked.

All text is invented. Renders are CC0-1.0.
"""
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
NAME = "carmenta-checkbox"
CLIPS = REPO / "corpora" / "clips" / NAME
TEXT_FONTS = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/cour.ttf"]
SYMBOL_FONT = "C:/Windows/Fonts/seguisym.ttf"
CJK_FONT = "C:/Windows/Fonts/msyh.ttc"
W, H = 2100, 640
QUESTIONS = [
    "Did you hold a position with a business?",
    "Is the income over $500,000?",
    "Was any gift received from a lobbyist?",
    "Does your spouse hold stock in the company?",
    "Real estate owned in this county?",
    "Any creditor owed more than $10,000?",
    "Interest held jointly with spouse?",
    "Was the loan made on ordinary terms?",
]


def stroke_box(d, x, y, s, t):
    d.rectangle([x, y, x + s - 1, y + s - 1], outline=0, width=t)


def mark(d, kind, x, y, s, t):
    """Draw a mark in the box at (x, y) of side s. Returns nothing."""
    if kind == "x":
        m = s // 5
        d.line([x + m, y + m, x + s - m, y + s - m], fill=0, width=t + 1)
        d.line([x + s - m, y + m, x + m, y + s - m], fill=0, width=t + 1)
    elif kind == "tick":
        # starts inside, ends above and right of the box: an overflowing tick
        d.line([x + s // 5, y + s // 2, x + s * 2 // 5, y + s * 4 // 5], fill=0, width=t + 2)
        d.line([x + s * 2 // 5, y + s * 4 // 5, x + s * 6 // 5, y - s // 3], fill=0, width=t + 2)
    elif kind == "fill":
        d.rectangle([x + t + 2, y + t + 2, x + s - t - 3, y + s - t - 3], fill=0)


def render(seed, ink):
    rnd = random.Random(seed)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    font_path = TEXT_FONTS[seed % len(TEXT_FONTS)]
    em = rnd.choice([34, 40, 46])
    font = ImageFont.truetype(font_path, em)
    sym = ImageFont.truetype(SYMBOL_FONT, int(em * 1.1))
    boxes = []
    y = 30
    for row in range(5):
        q = rnd.choice(QUESTIONS)
        d.text((30, y), q, font=font, fill=0)
        options = ["Yes", "No"] if row % 3 else ["Self", "Spouse", "Both"]
        # Boxes start after the question: a long line in a wide font must
        # never put text inside a box that the ground truth calls empty.
        x = max(1000, 30 + int(d.textlength(q, font=font)) + 60)
        chosen = rnd.randrange(len(options) + 1)  # len(options): none marked
        for i, opt in enumerate(options):
            s = int(em * rnd.choice([0.8, 0.9, 1.0]))
            t = max(2, s // 16)
            glyph = rnd.random() < 0.3
            by = y + (em - s) // 2 + 4
            marked = i == chosen
            kind = rnd.choice(["x", "tick", "fill", "glyph"]) if marked else "none"
            if glyph:
                ch = "\u2612" if kind in ("glyph", "x") else "\u2610"
                d.text((x, by), ch, font=sym, fill=0)
                gx0, gy0, gx1, gy1 = d.textbbox((x, by), ch, font=sym)
                bx, by2, bs = gx0, gy0, max(gx1 - gx0, gy1 - gy0)
                if kind in ("tick", "fill"):
                    mark(d, kind, gx0, gy0, gx1 - gx0, t)
                frame = [gx0, gy0, gx1 - gx0, gy1 - gy0]
            else:
                stroke_box(d, x, by, s, t)
                if marked:
                    mark(d, "x" if kind == "glyph" else kind, x, by, s, t)
                frame = [x, by, s, s]
            boxes.append({"bbox": frame, "checked": marked, "mark": kind, "glyph": glyph})
            d.text((frame[0] + frame[2] + 12, y), opt, font=font, fill=0)
            x = frame[0] + frame[2] + 30 + int(d.textlength(opt, font=font)) + 40
        y += int(em * 2.1)
    # Distractors: square-ish letters, a CJK ideograph, a table, a rule.
    d.text((30, y), "O D 0 8 QOD 080 ODD", font=font, fill=0)
    cjk = ImageFont.truetype(CJK_FONT, em)
    d.text((700, y), "\u53e3\u53e3 \u56de", font=cjk, fill=0)
    ty = y + int(em * 1.6)
    for c in range(4):
        for r in range(2):
            d.rectangle([30 + c * 220, ty + r * 70, 30 + (c + 1) * 220, ty + (r + 1) * 70], outline=0, width=2)
    d.rectangle([1000, ty + 60, 1650, ty + 62], fill=0)
    if ink == "bw":
        img = img.point(lambda v: 0 if v < 128 else 255, mode="L")
    return img, boxes


def main():
    CLIPS.mkdir(parents=True, exist_ok=True)
    man = [f'name = "{NAME}"', "version = 1", 'task = "ocr"']
    n = 0
    for seed in range(40):
        for ink in ["gray", "bw"]:
            img, boxes = render(seed, ink)
            cid = f"cb-{seed:02}-{ink}"
            png = CLIPS / f"{cid}.png"
            img.save(png, optimize=True)
            (CLIPS / f"{cid}.json").write_text(json.dumps({"ink": ink, "seed": seed, "boxes": boxes}), encoding="utf-8")
            sha = hashlib.sha256(png.read_bytes()).hexdigest()
            man += ["", "[[clips]]", f'id = "{cid}"', f'path = "clips/{NAME}/{png.name}"',
                    f'class = "checkbox-{ink}"', f'split = "{"train" if seed % 2 == 0 else "holdout"}"',
                    'license = "CC0-1.0 (render of invented form rows)"', f'sha256 = "{sha}"']
            n += 1
    (REPO / "corpora" / f"{NAME}-v1.toml").write_text("\n".join(man) + "\n", encoding="utf-8")
    print(f"{n} clips -> {CLIPS}")


if __name__ == "__main__":
    main()

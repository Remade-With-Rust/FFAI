"""carmenta-form-v1: typed form values on ruled lines, ground truth by construction.

Built for docs/plans/commercial-gaps.md gap 12. A caller reading scanned
financial-disclosure forms found values on ruled lines read as look-alike
letters ("1117" -> "TTT7", "$10,000" -> "810,000") at confidences that passed
a 0.85 gate. The filings themselves are public records full of personal data
and the caller does not keep them, so this corpus reproduces the CONDITION
instead: the same fictional values rendered with no rule (the control) and
with three kinds of rule, so every error the rule causes is attributable to
it by a paired comparison.

Axes:
  rule   none | under (touching the baseline) | desc (through the descender
         zone) | strike (through the middle of the glyphs)
  ink    gray (anti-aliased) | bw (1-bit, the way a CCITT Group 4 scan arrives)
  font   Courier New, Arial, Times New Roman, cycled per value
  size   40 / 46 / 52 px em, i.e. 10-12 pt at 300 ppi

Values with an even index are split=train (tune thresholds there), odd are
split=holdout (claim there). Each clip is one form line: a printed label and
a typed value; ground truth is "label value". A sidecar .json records the
value and condition so a scorer can measure the VALUE exactly.

All values are invented. Fonts are the Windows system fonts; the renders are
released CC0-1.0.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
NAME = "carmenta-form"
CLIPS = REPO / "corpora" / "clips" / NAME
FONTS = [
    ("courier", "C:/Windows/Fonts/cour.ttf"),
    ("arial", "C:/Windows/Fonts/arial.ttf"),
    ("times", "C:/Windows/Fonts/times.ttf"),
]
SIZES = [40, 46, 52]
W, H = 1500, 150

# (label, value, kind). Invented, and chosen for the confusions seen in the
# field: 1/I/l/T, 0/O/U, 5/S, 8/$, and names with descenders.
VALUES = [
    ("Amount:", "$10,000", "amount"),
    ("Amount:", "$1,000", "amount"),
    ("Amount:", "$150,000 to $249,999", "amount"),
    ("Amount:", "$5,001 - $10,000", "amount"),
    ("Amount:", "$500,000", "amount"),
    ("Amount:", "$1,117.00", "amount"),
    ("Amount:", "$25,000", "amount"),
    ("Amount:", "$100", "amount"),
    ("Amount:", "$2,500", "amount"),
    ("Amount:", "$75,000", "amount"),
    ("Address:", "1117 Oak Street", "number"),
    ("Address:", "1000 Park Avenue", "number"),
    ("Address:", "1121 River Road", "number"),
    ("Unit:", "#305", "number"),
    ("Address:", "Suite 1414", "number"),
    ("Address:", "PO Box 1810", "number"),
    ("Plan:", "401K Plan", "number"),
    ("Date:", "2/11/26", "number"),
    ("Date:", "12/31/2025", "number"),
    ("Address:", "Unit 7", "number"),
    ("Location:", "Gulf Shores", "name"),
    ("Location:", "Irish Hill", "name"),
    ("Name of Creditor:", "Merrill Lynch", "name"),
    ("Name of Creditor:", "Tri-State Bank", "name"),
    ("Location:", "Cabell County", "name"),
    ("Source of Income:", "Kentucky Utilities", "name"),
    ("Name of Creditor:", "First Federal Savings", "name"),
    ("Source of Income:", "Blue Grass Energy", "name"),
    ("Source of Income:", "Lexington Clinic", "name"),
    ("Business:", "Hillcrest Farms", "name"),
]
RULES = ["none", "under", "desc", "strike"]
INKS = ["gray", "bw"]


def render(label, value, font_path, em, rule, ink):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, em)
    ascent, descent = font.getmetrics()
    baseline = 30 + ascent
    x_label, x_value = 40, 560
    d.text((x_label, baseline), label, font=font, fill=0, anchor="ls")
    d.text((x_value, baseline), value, font=font, fill=0, anchor="ls")
    x0, _, x1, _ = d.textbbox((x_value, baseline), value, font=font, anchor="ls")
    t = max(2, round(em / 16))  # rule thickness: 2-3 px at 300 ppi
    if rule == "under":
        # a form's field line: spans the field, touching the glyph bottoms
        d.rectangle([x_value - 20, baseline, W - 40, baseline + t - 1], fill=0)
    elif rule == "desc":
        y = baseline + round(descent * 0.45)
        d.rectangle([x_value - 20, y, W - 40, y + t - 1], fill=0)
    elif rule == "strike":
        # a filer striking through an entry: only across the typed text
        y = baseline - round(ascent * 0.32)
        d.rectangle([x0 - 6, y, x1 + 6, y + t - 1], fill=0)
    if ink == "bw":
        img = img.point(lambda v: 0 if v < 128 else 255, mode="L")
    return img


def main():
    CLIPS.mkdir(parents=True, exist_ok=True)
    man = [f'name = "{NAME}"', "version = 1", 'task = "ocr"']
    n = 0
    for i, (label, value, kind) in enumerate(VALUES):
        font_name, font_path = FONTS[i % len(FONTS)]
        em = SIZES[(i // len(FONTS)) % len(SIZES)]
        split = "train" if i % 2 == 0 else "holdout"
        for rule in RULES:
            for ink in INKS:
                cid = f"form-{i:02}-{rule}-{ink}"
                png = CLIPS / f"{cid}.png"
                render(label, value, font_path, em, rule, ink).save(png, optimize=True)
                (CLIPS / f"{cid}.txt").write_text(f"{label} {value}", encoding="utf-8")
                meta = {"value": value, "label": label, "kind": kind, "rule": rule,
                        "ink": ink, "font": font_name, "em": em, "index": i}
                (CLIPS / f"{cid}.json").write_text(json.dumps(meta), encoding="utf-8")
                sha = hashlib.sha256(png.read_bytes()).hexdigest()
                man += ["", "[[clips]]", f'id = "{cid}"',
                        f'path = "clips/{NAME}/{png.name}"',
                        f'ground_truth = "clips/{NAME}/{cid}.txt"',
                        f'class = "form-{rule}-{ink}"', f'split = "{split}"',
                        'license = "CC0-1.0 (render of invented values in Windows system fonts)"',
                        f'sha256 = "{sha}"']
                n += 1
    (REPO / "corpora" / f"{NAME}-v1.toml").write_text("\n".join(man) + "\n", encoding="utf-8")
    print(f"{n} clips -> {CLIPS}")


if __name__ == "__main__":
    main()

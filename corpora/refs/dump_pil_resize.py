"""Single-variable instrument: PIL's Lanczos resize, as raw uint8, at several shapes.

Everything else in the content path (tiling, rescale, normalize, CHW) is
arithmetic verifiable by inspection. The resampler is the only stage with a
CONVENTION we have to match rather than derive, so it gets isolated here and
gated bit-exactly by `crates/ffai-argus/tests/resize_oracle.rs`.

# Why more than one shape

The content path itself only ever does 512 -> 2048 and 2048 -> 512, and both
are exact 4x ratios. That is a weak gate for a resampler:

* an exact integer ratio puts every output centre at a tidy offset, so a
  half-pixel convention error can partly cancel;
* the tap count is constant, so an off-by-one in the window width never shows;
* upscaling leaves `filter_scale` at 1, so the DOWNSCALE kernel widening
  (`filter_scale = max(1, scale)`) is exercised by exactly one case.

The odd and non-square shapes below cost nothing to dump and cover all three.

Run: python corpora/refs/dump_pil_resize.py
"""
import pathlib

import numpy as np
from PIL import Image

N = 512


def formula_image(n: int = N) -> np.ndarray:
    """The same deterministic pattern every Argus oracle uses."""
    y, x = np.mgrid[0:n, 0:n]
    fx, fy = x / n, y / n
    r = 0.5 + 0.5 * np.sin(6.0 * np.pi * fx)
    g = 0.5 + 0.5 * np.sin(6.0 * np.pi * fy + 1.0)
    b = 0.5 + 0.5 * np.sin(6.0 * np.pi * (fx + fy) + 2.0)
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def main() -> None:
    out = pathlib.Path(".oracle/pil-resize")
    out.mkdir(parents=True, exist_ok=True)

    px = formula_image()
    (out / "src_512.rgb8").write_bytes(px.tobytes())
    img = Image.fromarray(px, "RGB")

    # The two the content path actually performs.
    big = img.resize((2048, 2048), Image.LANCZOS)
    (out / "up_2048.rgb8").write_bytes(np.asarray(big).tobytes())
    (out / "down_512.rgb8").write_bytes(
        np.asarray(big.resize((512, 512), Image.LANCZOS)).tobytes()
    )

    # And the shapes that actually stress the kernel: non-integer ratios,
    # non-square targets, both directions, and one that shrinks hard enough to
    # widen the filter several times over.
    cases = [
        ("odd_up_777x333", img, (777, 333)),
        ("odd_down_301x419", img, (301, 419)),
        ("tiny_down_37x53", img, (37, 53)),
        ("wide_from_big_1000x250", big, (1000, 250)),
    ]
    manifest = []
    for name, src, (dw, dh) in cases:
        arr = np.asarray(src.resize((dw, dh), Image.LANCZOS))
        (out / f"{name}.rgb8").write_bytes(arr.tobytes())
        sw, sh = src.size
        manifest.append(f"{name} {sw} {sh} {dw} {dh}")
    (out / "cases.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print("PIL", getattr(Image, "__version__", "?"), "dumped ->", out)
    for line in manifest:
        print("  ", line)


if __name__ == "__main__":
    main()

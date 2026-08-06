# ffai — Python bindings

Diana (YOLO26, pure Rust) over numpy, PIL and torch arrays.

```python
import ffai, numpy as np, cv2

d = ffai.Detector("n")                       # tier; rect geometry, ./models
r = d.detect(cv2.imread("street.jpg"), from_bgr=True)

print(r)                                     # Detections(6 persons)
r.xyxy   # (N, 4) float32, SOURCE pixel coords - letterbox already inverted
r.conf   # (N,)  float32
r.cls    # (N,)  uint32
r.names  # 80 class names
```

Accepts anything numpy can view as `uint8`: numpy arrays, PIL images
(`np.asarray(img)`), torch tensors (`t.numpy()`). `HxWx3` RGB, `HxWx4` RGBA or
`HxW` grayscale.

**RGB is assumed.** OpenCV hands out BGR, so pass `from_bgr=True` rather than
letting a colour swap shift every detection silently — the same requirement
Ultralytics has, made explicit.

Verified against Ultralytics 8.4.113 on the same image: **6 detections each,
max box delta 0.0002 px.** All four input paths (numpy / PIL / torch / BGR)
produce byte-identical boxes.

The GIL is released for the duration of `detect`, so threading over frames
actually parallelises.

## Building

```
cargo build --release -p ffai-py
# then copy target/release/ffai.dll -> site-packages/ffai.pyd   (Windows)
#              target/release/libffai.so -> site-packages/ffai.so (Linux)
```

`maturin develop -m crates/ffai-py/Cargo.toml` does the same thing if you have it.

## Weights

None are bundled. YOLO26 checkpoints are AGPL-3.0; convert your own with
`tools/diana_convert.py`, which writes the `models/` directory `Detector` reads.

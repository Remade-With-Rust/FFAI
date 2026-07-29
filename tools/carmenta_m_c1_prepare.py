"""M-C1 one-shot preparation: CRAFT weights + oracle fixtures + TrOCR tokenizer.

Run from the repo root with the bench venv:
    .venv-bench/Scripts/python tools/carmenta_m_c1_prepare.py

Produces, into the FFai model cache (%LOCALAPPDATA%/ffai/models/...):
  craft-mlt/craft.safetensors        -- converted from craft_mlt_25k.pth (MIT),
                                        "module." DataParallel prefix stripped
  trocr-small-printed/tokenizer.json -- generated from the repo's sentencepiece
                                        model via transformers' slow->fast
                                        conversion (the HF repo ships no
                                        tokenizer.json; we generate, we don't
                                        trust a third-party re-upload)
and, into corpora/refs/fixtures/ (gitignored, regenerable):
  craft_fixture.npz  -- 640x640 crop of page-00 + the PyTorch CRAFT
                        region/affinity maps for it (the detection oracle)
  trocr_fixture.json -- a line-crop file + transformers' generated text for it
                        (the recognition oracle)
  trocr_line.png     -- the line crop itself

The torch CRAFT definition below is a faithful transcription of
clovaai/CRAFT-pytorch (MIT) craft.py + basenet/vgg16_bn.py, kept inline so
the oracle needs no extra pip package.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "ffai" / "models"
FIXTURES = REPO / "corpora" / "refs" / "fixtures"
PTH = Path.home() / ".EasyOCR" / "model" / "craft_mlt_25k.pth"


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Vgg16BN(nn.Module):
    def __init__(self):
        super().__init__()
        # Module NAMES must match the checkpoint exactly: clovaai's vgg16_bn
        # add_module()s each torchvision layer under its ORIGINAL feature
        # index (slice2.13, slice2.14, ...), so a renumbered Sequential loads
        # nothing under strict=True. Same computation, same keys.
        features = torchvision.models.vgg16_bn(weights=None).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for i in range(12):
            self.slice1.add_module(str(i), features[i])
        for i in range(12, 19):
            self.slice2.add_module(str(i), features[i])
        for i in range(19, 29):
            self.slice3.add_module(str(i), features[i])
        for i in range(29, 39):
            self.slice4.add_module(str(i), features[i])
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

    def forward(self, x):
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        return h, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2


class Craft(nn.Module):
    def __init__(self):
        super().__init__()
        self.basenet = Vgg16BN()
        # Constructor args derived from the CHECKPOINT shapes, not the paper:
        # conv0 is (in+mid) -> mid 1x1, conv3 is mid -> out 3x3. The cat
        # inputs are (fc7+relu5_3)=1536, (up+relu4_3)=768, (up+relu3_2)=384,
        # (up+relu2_2)=192 — all verified against craft_mlt_25k.pth.
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x):
        fc7, relu5_3, relu4_3, relu3_2, relu2_2 = self.basenet(x)
        y = torch.cat([fc7, relu5_3], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=relu4_3.shape[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, relu4_3], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(y, size=relu3_2.shape[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, relu3_2], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(y, size=relu2_2.shape[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, relu2_2], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)
        return y.permute(0, 2, 3, 1), feature


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_rgb(img):
    """CRAFT's normalizeMeanVariance: (img/255 - mean)/std per RGB channel."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img.astype(np.float32) / 255.0 - mean) / std


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)

    # ---- 1. CRAFT: load, strip DataParallel prefix, verify forward, save ----
    print(f"loading {PTH} ...", flush=True)
    state = torch.load(PTH, map_location="cpu")
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    model = Craft()
    model.load_state_dict(state, strict=True)  # strict: a mismatch is a bug, not a warning
    model.eval()
    print("state dict loaded strict=True — architecture transcription is exact", flush=True)

    from safetensors.torch import save_file
    out_dir = CACHE / "craft-mlt"
    out_dir.mkdir(parents=True, exist_ok=True)
    st_path = out_dir / "craft.safetensors"
    save_file(state, str(st_path))
    print(f"wrote {st_path} sha256={sha256(st_path)}", flush=True)

    # ---- 2. Detection oracle fixture: 640x640 crop of page-00, no resize ----
    from PIL import Image
    page = np.array(Image.open(REPO / "corpora/clips/carmenta-render/page-00.png").convert("RGB"))
    crop = page[:640, :640, :]
    x = torch.from_numpy(normalize_rgb(crop)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        t0 = time.perf_counter()
        maps, _ = model(x)
        print(f"torch forward {time.perf_counter()-t0:.2f}s", flush=True)
    maps = maps[0].numpy()  # (320, 320, 2): region, affinity
    np.savez_compressed(FIXTURES / "craft_fixture.npz", crop=crop, maps=maps)
    # A raw little-endian dump the Rust test can read without an npz parser.
    maps.astype("<f4").tofile(FIXTURES / "craft_maps_320x320x2_f32.bin")
    Image.fromarray(crop).save(FIXTURES / "craft_crop.png")
    print(f"fixture maps: region max {maps[...,0].max():.3f}, affinity max {maps[...,1].max():.3f}", flush=True)

    # ---- 3. TrOCR tokenizer.json (slow sentencepiece -> fast conversion) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/trocr-small-printed", use_fast=True)
    tdir = CACHE / "trocr-small-printed"
    tdir.mkdir(parents=True, exist_ok=True)
    tok.backend_tokenizer.save(str(tdir / "tokenizer.json"))
    print(f"wrote {tdir/'tokenizer.json'} sha256={sha256(tdir/'tokenizer.json')}", flush=True)

    # ---- 4. Recognition oracle: transformers' own output on one line crop ----
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    ved = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    ved.eval()
    # A clean line crop out of page-00: first text line region.
    line = page[72:110, 64:740, :]
    Image.fromarray(line).save(FIXTURES / "trocr_line.png")
    pixel_values = processor(images=Image.fromarray(line), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = ved.generate(pixel_values, max_new_tokens=64, num_beams=1, do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    (FIXTURES / "trocr_fixture.json").write_text(
        json.dumps({"file": "trocr_line.png", "greedy_text": text, "ids": ids[0].tolist()}),
        encoding="utf-8",
    )
    print(f"trocr oracle text: {text!r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

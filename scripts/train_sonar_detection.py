"""Milestone M2: a pipe detector for side-scan sonar images, tested on held-out minutes of the survey.

Usage: .venv/bin/python -I scripts/train_sonar_detection.py [DATA_DIR]   (EPOCHS=1 in the environment for a smoke run)
Writes docs/results/m2_sonar_detection.json, docs/figures/m2_sonar_test.jpg and models/m2_sonar_detection.pt (not in git).
FROM_SCRATCH=1 trains without the COCO and ImageNet starting weights and writes the same files with a "_scratch" suffix.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

root = Path(__file__).resolve().parent.parent
data = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/SubPipeMini/DATA"
BANDS = ("HF", "LF")
WIDTH = 1333  # sonar images are 5000 (HF) or 2500 (LF) pixels across; this is the width the detector would resize them to
BLOCK_S = 60
GAP_S = 25  # each sonar image is a rolling window over the last 25 s of pings, so images closer than this share pixels
TEST_BLOCKS = [1, 6, 7]  # 60-120 s and 360-480 s: the M1 test minutes, plus minute 7, where the pipe drops out of view
EPOCHS, BATCH, LR, SEED = int(os.environ.get("EPOCHS", 20)), 4, 0.005, 0
SCRATCH = os.environ.get("FROM_SCRATCH") == "1"  # no COCO or ImageNet starting weights: the licence question in docs/LICENCES.md
TAG = "_scratch" if SCRATCH else ""  # a from-scratch run writes its own files and leaves the M2 results alone
SCORE = 0.5  # a detection at or above this score counts as "pipe seen"
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.hub.set_dir(str(root / "models/torch"))  # keep downloaded weights inside the project

# Sonar windows from both bands: seconds since the survey start, image, and pipe boxes (x0, y0, x1, y1 in scaled pixels).
t0 = float(pd.read_csv(data / "EstimatedState.csv", nrows=1).rename(columns=str.strip)["timestamp"].iloc[0])
band, t, X, B, paths, centre = [], [], [], [], [], {name: [] for name in BANDS}
for name in BANDS:
    d = data / f"SSS_{name}_images"
    for p in sorted((d / "Image").glob("*.pbm"), key=lambda p: float(p.stem)):
        img = Image.open(p).convert("RGB")
        w, h = WIDTH, round(img.height * WIDTH / img.width)
        f = d / "YOLO_Annotation" / f"{p.stem}.txt"
        yolo = np.array([line.split()[1:] for line in f.read_text().splitlines() if line.strip()] if f.exists() else [], float).reshape(-1, 4)
        cx, cy, bw, bh = yolo.T
        centre[name] += list(cx * img.width)
        B.append(np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1).astype(np.float32))
        X.append(np.array(img.resize((w, h), Image.BILINEAR)))
        band.append(name)
        t.append(float(p.stem) - t0)
        paths.append(p)
band, t = np.array(band), np.array(t)
present = np.array([len(b) > 0 for b in B])

# The gap rests on this: one second later, the same picture has moved down 20 rows (500 rows = 25 s).
a, b = (np.asarray(Image.open(paths[k]).convert("L"), float) for k in np.flatnonzero(band == "HF")[100:102])
assert np.abs(a[:-20] - b[20:]).mean() < 1, "sonar images should be rolling windows that move 20 rows a second"

# Split by time; drop training windows within GAP_S of a test window.
held = np.isin((t // BLOCK_S).astype(int), TEST_BLOCKS)
role = np.where(held, "test", np.where(np.abs(t[:, None] - t[held][None, :]).min(axis=1) < GAP_S, "gap", "train"))
train, test = np.flatnonzero(role == "train"), np.flatnonzero(role == "test")
assert np.abs(t[train][:, None] - t[test][None, :]).min() >= GAP_S
for name in BANDS:  # each band needs windows with and without the pipe, in training and in test
    for ids in (train, test):
        assert 0 < present[ids][band[ids] == name].sum() < (band[ids] == name).sum()


def ap(dets, gts, thr):
    """Average precision for one class: all-point interpolation (VOC 2010 onwards), tied scores taken as one threshold.
    dets: per image, (boxes, scores); gts: per image, boxes."""
    flat = sorted(((s, i, b) for i, (boxes, scores) in enumerate(dets) for b, s in zip(boxes, scores)), key=lambda d: -d[0])
    used = [np.zeros(len(g), bool) for g in gts]
    hit = np.zeros(len(flat), bool)
    for k, (_, i, b) in enumerate(flat):
        if len(gts[i]):
            ious = box_iou(torch.from_numpy(b[None]), torch.from_numpy(gts[i]))[0].numpy()
            j = int(ious.argmax())
            hit[k] = ious[j] >= thr and not used[i][j]
            used[i][j] |= hit[k]
    scores = np.array([d[0] for d in flat])
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1] if len(flat) else np.zeros(0, int)  # end of each tie
    tp = np.cumsum(hit)[last]
    recall, precision = tp / sum(len(g) for g in gts), tp / (last + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # best precision at this recall or beyond
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


g = [np.array([[0, 0, 10, 10]], np.float32), np.zeros((0, 4), np.float32)]
on, off = np.array([[0, 0, 10, 10]], np.float32), np.array([[20, 20, 30, 30]], np.float32)
assert ap([(on, np.array([0.9])), (off, np.array([0.8]))], g, 0.5) == 1.0  # the false alarm ranks below the hit
assert ap([(on, np.array([0.8])), (off, np.array([0.9]))], g, 0.5) == 0.5  # the false alarm ranks above it
assert ap([(on, np.array([0.9])), (off, np.array([0.9]))], g, 0.5) == 0.5  # a tie is one threshold: precision 0.5


def evaluate(dets, gts):
    seen = np.array([len(s) > 0 and s.max() >= SCORE for _, s in dets])
    has = np.array([len(g) > 0 for g in gts])
    return {
        "ap50": round(ap(dets, gts, 0.5), 3),
        "ap50_95": round(float(np.mean([ap(dets, gts, thr) for thr in np.arange(0.5, 0.96, 0.05)])), 3),
        "pipe_windows_found": [int((seen & has).sum()), int(has.sum())],
        "no_pipe_windows_left_empty": [int((~seen & ~has).sum()), int((~has).sum())],
    }


def tensor(j):
    return torch.from_numpy(X[j]).permute(2, 0, 1).float().div(255)


# One model for both bands: COCO-trained Faster R-CNN with a new two-class head (background, pipe).
if SCRATCH:
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=2)
else:
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * (len(train) // BATCH), pct_start=0.1)
curve, start = [], time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    order = rng.permutation(train)
    for i in range(0, len(order) - BATCH + 1, BATCH):
        images, targets = [], []
        for j in order[i : i + BATCH]:
            x, b = tensor(j), torch.from_numpy(B[j])
            if rng.random() < 0.5:  # mirror port and starboard, so the model can't lean on which side the pipe is
                x, b = x.flip(-1), torch.stack([x.shape[-1] - b[:, 2], b[:, 1], x.shape[-1] - b[:, 0], b[:, 3]], 1)
            images.append(x.to(device))
            targets.append({"boxes": b.to(device), "labels": torch.ones(len(b), dtype=torch.int64, device=device)})
        loss = sum(model(images, targets).values())
        assert torch.isfinite(loss), "training diverged"
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4)})
    print(curve[-1], flush=True)
train_minutes = (time.time() - start) / 60
torch.save(model.state_dict(), root / f"models/m2_sonar_detection{TAG}.pt")


@torch.no_grad()
def detect(ids):
    model.eval()
    return [(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()) for i in range(0, len(ids), 8) for r in model([tensor(j).to(device) for j in ids[i : i + 8]])]


# Test once. The baseline predicts the median training box in every window, with no learning at all.
start = time.time()
dets = detect(test)
speed = len(test) / (time.time() - start)
scores_by_band = {}
for name in BANDS:
    keep = band[test] == name
    gts = [B[j] for j in test[keep]]
    fixed = np.median(np.concatenate([B[j] for j in train if band[j] == name]), axis=0).astype(np.float32)
    scores_by_band[name] = {
        "detector": evaluate([d for d, k in zip(dets, keep) if k], gts),
        "fixed_box_baseline": evaluate([(fixed[None], np.ones(1, np.float32))] * len(gts), gts),
    }


def stretches(name):
    """Runs of consecutive windows with a pipe box, as [first, last] seconds."""
    tt, has = t[band == name], present[band == name].astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, has, 0]))
    return [[round(float(tt[a]), 1), round(float(tt[b - 1]), 1)] for a, b in zip(edges[::2], edges[1::2])]


def count(mask):
    return {"windows": int(mask.sum()), "with_pipe": int((mask & present).sum())}


results = {
    "model": "torchvision Faster R-CNN, MobileNetV3-Large FPN, " + ("random starting weights" if SCRATCH else "COCO weights, new two-class head") + "; one model for both bands",
    "input_width": WIDTH,
    "data": {
        name: {"windows": count(band == name), "seconds_with_pipe_box": stretches(name),
               "box_centre_px": {"median": round(float(np.median(centre[name]))), "std": round(float(np.std(centre[name])))}}
        for name in BANDS
    },
    "split": {
        "block_s": BLOCK_S, "gap_s": GAP_S, "test_blocks": TEST_BLOCKS,
        "windows": {name: {r: count((role == r) & (band == name)) for r in ("train", "test", "gap")} for name in BANDS},
    },
    "training": {
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "schedule": "one-cycle", "seed": SEED,
        "augmentation": "port-starboard mirror", "minutes": round(train_minutes, 1), "device": device,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
    },
    "curve": curve,
    "test": {"score_threshold_for_seen": SCORE, **scores_by_band},
    "inference_windows_per_s": round(speed, 1),
    # Every test window: where the best detection sits ("left" is the labelled pipe's side of the vehicle track).
    "test_windows": [
        {"band": str(band[j]), "t": round(float(t[j]), 1), "labelled": bool(present[j]), "best_score": round(float(s.max()), 3) if len(s) else 0.0,
         "best_box_side": ("left" if bx[s.argmax(), [0, 2]].mean() < X[j].shape[1] / 2 else "right") if len(s) else None}
        for j, (bx, s) in zip(test, dets)
    ],
}
(root / "docs/results").mkdir(parents=True, exist_ok=True)
(root / f"docs/results/m2_sonar_detection{TAG}.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results["test"], indent=2))

# Figure: four test windows per band, spread over the test seconds. Green: labelled pipe box. Magenta: best detection.
font = ImageFont.load_default(size=16)
strips = []
for name in BANDS:
    mine = np.flatnonzero(band[test] == name)
    for k in mine[np.linspace(0, len(mine) - 1, 4).astype(int)]:
        j, (boxes, scores) = test[k], dets[k]
        img = Image.fromarray(X[j]).resize((1200, 150))
        sx, sy = 1200 / X[j].shape[1], 150 / X[j].shape[0]
        draw = ImageDraw.Draw(img)
        for x0, y0, x1, y1 in B[j]:
            draw.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(0, 230, 0), width=3)
        note = "no detection"
        if len(scores):
            x0, y0, x1, y1 = boxes[scores.argmax()]
            draw.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 0, 255), width=2)
            note = f"best detection {scores.max():.2f}"
        label = "pipe labelled" if len(B[j]) else "no pipe label"
        draw.text((6, 4), f"{name}  t+{t[j]:.0f}s  {label}, {note}", fill="white", font=font, stroke_width=2, stroke_fill="black")
        strips.append(img)
sheet = Image.new("RGB", (1200, len(strips) * 156), "white")
for k, s in enumerate(strips):
    sheet.paste(s, (0, k * 156))
sheet.save(root / f"docs/figures/m2_sonar_test{TAG}.jpg", quality=85)

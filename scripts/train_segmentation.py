"""Milestone M1: a pipe segmentation baseline on SubPipe camera frames, tested on held-out minutes of the survey.

Usage: .venv/bin/python -I scripts/train_segmentation.py [DATA_DIR]   (EPOCHS=1 in the environment for a smoke run)
Writes docs/results/m1_segmentation.json, docs/figures/m1_predictions.jpg and models/m1_segmentation.pt (not in git).
FROM_SCRATCH=1 trains without the ImageNet starting weights and writes the same files with a "_scratch" suffix.
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
from torch.nn import functional as F
from torchvision import tv_tensors
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import v2

root = Path(__file__).resolve().parent.parent
data = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/SubPipeMini/DATA"
seg = data / "Segmentation"
W, H = 640, 360  # frames are 2704 x 1520; train and score at this size
BLOCK_S = 60  # one-minute blocks; each block goes wholly to training, validation or test
GAP_S = 5  # the camera sees much the same seabed for a few seconds, so drop training frames this close to val or test
TEST_BLOCKS = [1, 6]  # 60-120 s (blue water) and 360-420 s (green water)
VAL_BLOCKS = [3]  # 180-240 s (both colours), used only to pick the epoch
EPOCHS, BATCH, LR, SEED = int(os.environ.get("EPOCHS", 40)), 8, 3e-4, 0
SCRATCH = os.environ.get("FROM_SCRATCH") == "1"  # no ImageNet starting weights: the licence question in docs/LICENCES.md
TAG = "_scratch" if SCRATCH else ""  # a from-scratch run writes its own files and leaves the M1 results alone
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.hub.set_dir(str(root / "models/torch"))  # keep downloaded weights inside the project

# Labelled frames, in time order, as seconds since the start of the survey.
t0 = float(pd.read_csv(data / "EstimatedState.csv", nrows=1).rename(columns=str.strip)["timestamp"].iloc[0])
masks = sorted(seg.glob("*_label.png"), key=lambda m: float(m.name[: -len("_label.png")]))
t = np.array([float(m.name[: -len("_label.png")]) for m in masks]) - t0


def load(mask):
    """The frame resized to W x H, and its pipe mask. Any non-zero label pixel is pipe (labels come in three formats)."""
    stem = mask.name[: -len("_label.png")]
    img = Image.open(next(p for p in (seg / f"{stem}.jpg", seg / f"{stem}.png") if p.exists()))
    img.draft("RGB", (W, H))  # JPEGs decode at reduced scale, much faster
    a = np.array(Image.open(mask))
    pipe = a[..., :3].any(axis=-1) if a.ndim == 3 else a > 0
    return np.array(img.convert("RGB").resize((W, H), Image.BILINEAR)), np.array(Image.fromarray(pipe).resize((W, H), Image.NEAREST))


X, Y = map(np.stack, zip(*(load(m) for m in masks)))
colour = X[..., 2].mean(axis=(1, 2)) - X[..., 1].mean(axis=(1, 2))  # mean blue minus mean green: > 0 blue water, < 0 green

# Split by time, then drop training frames that sit within GAP_S of a validation or test frame.
block = (t // BLOCK_S).astype(int)
role = np.where(np.isin(block, TEST_BLOCKS), "test", np.where(np.isin(block, VAL_BLOCKS), "val", "train"))
held = t[role != "train"]
role[(role == "train") & (np.abs(t[:, None] - held[None, :]).min(axis=1) < GAP_S)] = "gap"
idx = {r: np.flatnonzero(role == r) for r in ("train", "val", "test")}
assert np.abs(t[idx["train"]][:, None] - held[None, :]).min() >= GAP_S
assert min(colour[block == b].mean() for b in TEST_BLOCKS) < 0 < max(colour[block == b].mean() for b in TEST_BLOCKS), "test should cover blue and green water"

augment = v2.Compose([v2.RandomHorizontalFlip(), v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)])
normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics, which the encoder expects


def batches(ids, train=False):
    if train:  # shuffle, and keep only full batches (batch norm fails on a batch of one)
        ids = rng.permutation(ids)[: len(ids) // BATCH * BATCH]
    for i in range(0, len(ids), BATCH):
        pairs = [(tv_tensors.Image(torch.from_numpy(X[j]).permute(2, 0, 1)), tv_tensors.Mask(torch.from_numpy(Y[j])[None])) for j in ids[i : i + BATCH]]
        if train:
            pairs = [augment(x, y) for x, y in pairs]  # the flip applies to frame and mask together; colour to the frame only
        x, y = map(torch.stack, zip(*pairs))
        yield normalize(x.float() / 255).to(device), y.float().to(device)


@torch.no_grad()
def predict(ids):
    model.eval()
    return np.concatenate([(model(x)["out"][:, 0] > 0).cpu().numpy() for x, _ in batches(ids)])


def iou(p, y):
    """Pipe IoU and background IoU, counting pixels over all the frames given."""
    return float((p & y).sum() / (p | y).sum()), float((~p & ~y).sum() / (~p | ~y).sum())


assert iou(np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 0], bool)) == (0.5, 2 / 3)


def scores(p, y):
    pipe, background = iou(p, y)
    return {"pipe_iou": round(pipe, 3), "background_iou": round(background, 3), "mean_iou": round((pipe + background) / 2, 3)}


# Train; keep the weights from the epoch with the best validation pipe IoU.
model = deeplabv3_mobilenet_v3_large(num_classes=1, weights_backbone=None if SCRATCH else "IMAGENET1K_V1").to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * (len(idx["train"]) // BATCH))
curve, best, start = [], (-1.0, 0, None), time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for x, y in batches(idx["train"], train=True):
        loss = F.binary_cross_entropy_with_logits(model(x)["out"], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    val_iou = iou(predict(idx["val"]), Y[idx["val"]])[0]
    curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val_pipe_iou": round(val_iou, 4)})
    print(curve[-1], flush=True)
    if val_iou > best[0]:
        best = (val_iou, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
train_minutes = (time.time() - start) / 60
model.load_state_dict(best[2])
torch.save(best[2], root / f"models/m1_segmentation{TAG}.pt")

# Test once, on the held-out minutes.
start = time.time()
pred = predict(idx["test"])
fps = len(pred) / (time.time() - start)
truth, test_block = Y[idx["test"]], block[idx["test"]]
inter, union = (pred & truth).sum(axis=(1, 2)), (pred | truth).sum(axis=(1, 2))
per_frame = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
results = {
    "model": f"torchvision DeepLabV3, MobileNetV3-Large encoder initialised from {'random weights' if SCRATCH else 'ImageNet weights'}, new one-class head",
    "input_size": [W, H],
    "split": {
        "block_s": BLOCK_S,
        "gap_s": GAP_S,
        "test_blocks": TEST_BLOCKS,
        "val_blocks": VAL_BLOCKS,
        "frames": {r: int((role == r).sum()) for r in ("train", "val", "test", "gap")},
        "labelled_span_s": [round(float(t.min()), 1), round(float(t.max()), 1)],
        "blocks": [
            {"seconds": [int(b) * BLOCK_S, (int(b) + 1) * BLOCK_S], "frames": int((block == b).sum()),
             "roles": sorted(set(role[block == b].tolist())), "blue_minus_green": round(float(colour[block == b].mean()), 1)}
            for b in np.unique(block)
        ],
    },
    "training": {
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED, "loss": "binary cross-entropy",
        "augmentation": "horizontal flip, colour jitter", "best_epoch": best[1], "minutes": round(train_minutes, 1),
        "device": device, "torch": torch.__version__, "torchvision": torchvision.__version__,
    },
    "curve": curve,
    "test": {
        **scores(pred, truth),
        "per_block": {f"{b * BLOCK_S}-{(b + 1) * BLOCK_S} s": scores(pred[test_block == b], truth[test_block == b]) for b in TEST_BLOCKS},
        "per_frame_pipe_iou": {"median": round(float(np.median(per_frame)), 3), "p10": round(float(np.percentile(per_frame, 10)), 3),
                               "min": round(float(per_frame.min()), 3), "frames_below_0.5": int((per_frame < 0.5).sum())},
    },
    "inference_frames_per_s": round(fps, 1),
}
(root / "docs/results").mkdir(parents=True, exist_ok=True)
(root / f"docs/results/m1_segmentation{TAG}.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results["test"], indent=2))

# Figures: eight test frames spread over the two test minutes, and the eight worst.
# Yellow = pipe found, magenta = pipe missed, cyan = false alarm.
font = ImageFont.load_default(size=18)
tw, th = 448, 252
for name, picks in (("m1_predictions", np.linspace(0, len(pred) - 1, 8).astype(int)), ("m1_worst_frames", np.argsort(per_frame)[:8])):
    sheet = Image.new("RGB", (tw * 4, th * 2), "white")
    for k, i in enumerate(picks):
        frame, p, y = X[idx["test"][i]].copy(), pred[i], truth[i]
        for region, rgb in ((p & y, (255, 220, 0)), (y & ~p, (255, 0, 255)), (p & ~y, (0, 255, 255))):
            frame[region] = (0.5 * frame[region] + 0.5 * np.array(rgb)).astype("uint8")
        tile = Image.fromarray(frame).resize((tw, th))
        ImageDraw.Draw(tile).text((8, 6), f"t+{t[idx['test'][i]]:.0f}s  pipe IoU {per_frame[i]:.2f}", fill="white", font=font, stroke_width=2, stroke_fill="black")
        sheet.paste(tile, ((k % 4) * tw, (k // 4) * th))
    sheet.save(root / f"docs/figures/{name}{TAG}.jpg", quality=85)

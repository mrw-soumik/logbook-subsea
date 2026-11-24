"""M6 follow-up: does training the sonar detector on more surveys close the gap to NFR-02's 0.90?

Trains the M2 detector again, with the same settings, on Chunks 0, 1 and 2 instead of Chunk 2 alone. Chunk 2 gives
every window; Chunks 0 and 1 give the every-5th windows fetched in M6. Chunks 3 and 4 stay unseen for the test, which
is run by scripts/evaluate_chunks.py:
  SONAR_MODEL=models/m6_sonar_multichunk.pt OUT=docs/results/m6_sonar_multichunk_test.json \
    .venv/bin/python -I scripts/evaluate_chunks.py Chunk3 Chunk4
No setting was changed after seeing Chunks 3 and 4.

Usage: .venv/bin/python -I scripts/train_sonar_multichunk.py   (EPOCHS=1 in the environment for a smoke run)
Writes models/m6_sonar_multichunk.pt (not in git) and docs/results/m6_sonar_multichunk_training.json.
FROM_SCRATCH=1 trains without the COCO and ImageNet starting weights and writes the same files with a "_scratch" suffix.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

root = Path(__file__).resolve().parent.parent
SOURCES = {"Chunk2": root / "data/SubPipeMini/DATA", "Chunk0": root / "data/SubPipe/DATA/Chunk0", "Chunk1": root / "data/SubPipe/DATA/Chunk1"}
WIDTH = 1333
EPOCHS, BATCH, LR, SEED = int(os.environ.get("EPOCHS", 20)), 4, 0.005, 0  # the M2 settings, unchanged
SCRATCH = os.environ.get("FROM_SCRATCH") == "1"  # no COCO or ImageNet starting weights: the licence question in docs/LICENCES.md
TAG = "_scratch" if SCRATCH else ""
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.hub.set_dir(str(root / "models/torch"))

X, B, counts, meta = [], [], {}, []  # meta: (chunk, band, stem) of each window, to name any batch that misbehaves
for chunk, d in SOURCES.items():
    for band in ("HF", "LF"):
        folder = d / f"SSS_{band}_images"
        paths = sorted((folder / "Image").glob("*.pbm"), key=lambda p: float(p.stem))  # skips the misspelt .bpm files
        with_pipe = 0
        for p in paths:
            img = Image.open(p).convert("RGB")
            w, h = WIDTH, round(img.height * WIDTH / img.width)
            f = folder / "YOLO_Annotation" / f"{p.stem}.txt"
            yolo = np.array([line.split()[1:] for line in f.read_text().splitlines() if line.strip()] if f.exists() else [], float).reshape(-1, 4)
            cx, cy, bw, bh = yolo.T
            B.append(np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1).astype(np.float32))
            X.append(np.array(img.resize((w, h), Image.BILINEAR)))
            meta.append((chunk, band, p.stem))
            with_pipe += len(yolo) > 0
        counts[f"{chunk} {band}"] = {"windows": len(paths), "with_pipe": with_pipe}
        print(chunk, band, counts[f"{chunk} {band}"], flush=True)
assert all(c["windows"] and 0 < c["with_pipe"] < c["windows"] for c in counts.values()), "each source needs pipe and no-pipe windows"


def tensor(j):
    return torch.from_numpy(X[j]).permute(2, 0, 1).float().div(255)


if SCRATCH:
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=2)
else:
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * (len(X) // BATCH), pct_start=0.1)
curve, skipped, pipe_free_batches, start = [], [], 0, time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    order = rng.permutation(len(X))
    for i in range(0, len(order) - BATCH + 1, BATCH):
        # A batch with no pipe in any window can leave the second stage with nothing to score, and its loss turns NaN
        # (three runs hit this from epoch 6). Such batches are about 0.5% of all; pipe-free windows still train in mixed ones.
        if not any(len(B[j]) for j in order[i : i + BATCH]):
            pipe_free_batches += 1
            continue
        images, targets = [], []
        for j in order[i : i + BATCH]:
            x, b = tensor(j), torch.from_numpy(B[j])
            if rng.random() < 0.5:  # mirror port and starboard, as in M2
                x, b = x.flip(-1), torch.stack([x.shape[-1] - b[:, 2], b[:, 1], x.shape[-1] - b[:, 0], b[:, 3]], 1)
            images.append(x.to(device))
            targets.append({"boxes": b.to(device), "labels": torch.ones(len(b), dtype=torch.int64, device=device)})
        parts = model(images, targets)
        loss = sum(parts.values())
        if not torch.isfinite(loss):  # skip the step and record which windows caused it; many such batches is a real failure
            skipped.append({"epoch": epoch, "windows": [meta[j] for j in order[i : i + BATCH]], "losses": {k: float(v) for k, v in parts.items()}})
            print("skipped a non-finite batch:", skipped[-1], flush=True)
            assert len(skipped) <= 5, "training diverged"
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4)})
    print(curve[-1], flush=True)
torch.save(model.state_dict(), root / f"models/m6_sonar_multichunk{TAG}.pt")
(root / f"docs/results/m6_sonar_multichunk_training{TAG}.json").write_text(json.dumps({
    "starting_weights": "random" if SCRATCH else "COCO (torchvision), with an ImageNet backbone",
    "trained_on": counts, "unseen_for_test": ["Chunk3", "Chunk4"],
    "settings": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "schedule": "one-cycle", "seed": SEED, "augmentation": "port-starboard mirror",
                 "pipe_free_batches_skipped": "all batches whose four windows had no pipe box (see pipe_free_batches below)",
                 "note": "three earlier runs hit NaN losses from epoch 6, every time on a batch of four pipe-free windows; a run keeping every proposal instead was far too slow",
                 "device": device, "torch": torch.__version__, "torchvision": torchvision.__version__},
    "minutes": round((time.time() - start) / 60, 1), "pipe_free_batches": pipe_free_batches, "skipped_non_finite_batches": skipped, "curve": curve}, indent=2) + "\n")

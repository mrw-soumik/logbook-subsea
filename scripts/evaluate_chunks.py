"""M6: test the M1 and M2 models on survey chunks they never saw (Chunks 0, 1, 3 and 4 of the full SubPipe archive).

Sonar has labels in every chunk, so the detector is scored exactly as in M2, against a fixed box taken from Chunk 2.
The camera has pipe labels only in Chunk 2, so for the camera this reports how often the model finds the pipe in
sampled frames when the sonar labels put the pipe beside the vehicle, plus a figure to judge by eye. That is a rough
check, not IoU.

Usage: .venv/bin/python -I scripts/evaluate_chunks.py   (after scripts/fetch_chunks.py; see the README)
Writes docs/results/m6_held_out_chunks.json and docs/figures/m6_held_out_camera.jpg.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.ops import box_iou
from torchvision.transforms import v2

root = Path(__file__).resolve().parent.parent
DEFAULT_OUT = root / "docs/results/m6_held_out_chunks.json"
SONAR_MODEL = Path(os.environ.get("SONAR_MODEL", root / "models/m2_sonar_detection.pt"))  # to score another detector
OUT = Path(os.environ.get("OUT", DEFAULT_OUT))
full, mini = root / "data/SubPipe/DATA", root / "data/SubPipeMini/DATA"
CHUNKS = sys.argv[1:] or ["Chunk0", "Chunk1", "Chunk3", "Chunk4"]  # name chunks to check a subset
WIDTH, SCORE, HIDDEN_SHARE = 1333, 0.5, 0.02  # the M2 input width, the M2 "pipe seen" score, the M3 "pipe in view" share
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.hub.set_dir(str(root / "models/torch"))


def ap(dets, gts, thr):
    """Average precision for one class, exactly as in scripts/train_sonar_detection.py (all-point, ties as one)."""
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
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1] if len(flat) else np.zeros(0, int)
    tp = np.cumsum(hit)[last]
    recall, precision = tp / max(sum(len(g) for g in gts), 1), tp / (last + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


g = [np.array([[0, 0, 10, 10]], np.float32), np.zeros((0, 4), np.float32)]
on, off = np.array([[0, 0, 10, 10]], np.float32), np.array([[20, 20, 30, 30]], np.float32)
assert ap([(on, np.array([0.9])), (off, np.array([0.8]))], g, 0.5) == 1.0
assert ap([(on, np.array([0.8])), (off, np.array([0.9]))], g, 0.5) == 0.5


def scores(dets, gts):
    seen = np.array([len(s) > 0 and s.max() >= SCORE for _, s in dets])
    has = np.array([len(x) > 0 for x in gts])
    found, empty = (seen & has).sum(), (~seen & ~has).sum()
    return {"ap50": round(ap(dets, gts, 0.5), 3),
            "ap50_95": round(float(np.mean([ap(dets, gts, t) for t in np.arange(0.5, 0.96, 0.05)])), 3),
            "pipe_windows_found": [int(found), int(has.sum())], "no_pipe_windows_left_empty": [int(empty), int((~has).sum())],
            "balanced_accuracy": round(float((found / has.sum() + empty / (~has).sum()) / 2), 3) if has.any() and (~has).any() else None}


def yolo(folder, stem):
    f = folder / f"{stem}.txt"
    return np.array([line.split()[1:] for line in f.read_text().splitlines() if line.strip()] if f.exists() else [], float).reshape(-1, 4)


def t0_of(chunk_dir):
    return float(pd.read_csv(chunk_dir / "EstimatedState.csv", nrows=1).rename(columns=str.strip)["timestamp"].iloc[0])


state = torch.load(SONAR_MODEL)
if any(k.endswith("num_batches_tracked") for k in state):  # trained from scratch: plain batch norm, no starting weights
    det = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=2)
else:
    det = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    det.roi_heads.box_predictor = FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features, 2)
det.load_state_dict(state)
det.eval().to(device)
seg = deeplabv3_mobilenet_v3_large(num_classes=1, weights_backbone=None)
seg.load_state_dict(torch.load(root / "models/m1_segmentation.pt"))
seg.eval().to(device)
normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# The fixed box: the median Chunk 2 pipe box per band, in image-relative units (Chunk 2 is the training survey).
fixed = {b: np.median(np.concatenate([yolo(p.parent, p.stem) for p in (mini / f"SSS_{b}_images/YOLO_Annotation").glob("*.txt")
                                      if p.name != "classes.txt"]), axis=0) for b in ("HF", "LF")}

results, tiles, camera_found, ref_sums = {}, [], {}, []  # ref_sums: pixel counts against reference outlines, where they exist
for chunk in CHUNKS:
    d = full / chunk
    if not (d / "EstimatedState.csv").exists():
        results[chunk] = {"status": "not fetched"}
        continue
    t0, out = t0_of(d), {}
    for band in ("HF", "LF"):
        folder = d / f"SSS_{band}_images"
        paths = sorted((folder / "Image").glob("*.pbm"), key=lambda p: float(p.stem))
        dets, gts, fixed_dets = [], [], []
        with torch.no_grad():
            for i in range(0, len(paths), 8):
                imgs = [Image.open(p).convert("RGB") for p in paths[i : i + 8]]
                xs = [torch.from_numpy(np.array(im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.BILINEAR))).permute(2, 0, 1).float().div(255)
                      for im in imgs]
                for p, x, r in zip(paths[i : i + 8], xs, det([x.to(device) for x in xs])):
                    h, w = x.shape[1:]
                    cx, cy, bw, bh = yolo(folder / "YOLO_Annotation", p.stem).T
                    gts.append(np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1).astype(np.float32))
                    dets.append((r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()))
                    fx, fy, fw, fh = fixed[band]
                    fixed_dets.append((np.array([[(fx - fw / 2) * w, (fy - fh / 2) * h, (fx + fw / 2) * w, (fy + fh / 2) * h]], np.float32), np.ones(1, np.float32)))
        centre = [float((b[0] + b[2]) / 2 / WIDTH) for b in gts if len(b) for b in b]
        out[f"sonar_{band}"] = {"windows": len(paths), "skipped_bpm_files": len(list((folder / "Image").glob("*.bpm"))),  # misspelt extension in the archive
                                "detector": scores(dets, gts), "fixed_box_baseline": scores(fixed_dets, gts),
                                "pipe_box_centre_relative": {"median": round(float(np.median(centre)), 3) if centre else None,
                                                             "chunk2": round(float(fixed[band][0]), 3)}}
    # Camera: sampled frames; the reference is whether the high-frequency sonar label puts the pipe beside the vehicle.
    frames = sorted((d / "Cam0_images").glob("*.jpg"), key=lambda p: float(p.stem))
    if frames:
        labelled = {round(float(p.stem)): True for p in (d / "SSS_HF_images/YOLO_Annotation").glob("*.txt")
                    if p.stem.replace(".", "", 1).isdigit()}  # skips classes.txt and a stray "d.860.txt" in Chunk 0
        shares = []
        with torch.no_grad():
            for p in frames:
                im = Image.open(p)
                im.draft("RGB", (640, 360))
                a = np.array(im.convert("RGB").resize((640, 360), Image.BILINEAR))
                m = (seg(normalize(torch.from_numpy(a).permute(2, 0, 1).float()[None] / 255).to(device))["out"][0, 0] > 0).cpu().numpy()
                shares.append(float(m.mean()))
                ref_file = root / "labels/camera_masks_unseen" / f"{chunk}_{p.stem}.png"
                if ref_file.exists():
                    ref = np.array(Image.open(ref_file)) > 0
                    ref_sums.append((chunk, (m & ref).sum(), (m | ref).sum(), (~m & ~ref).sum(), (~m | ~ref).sum()))
                sonar_says = any(labelled.get(round(float(p.stem)) + k) for k in (-1, 0, 1))
                tiles.append((chunk, float(p.stem) - t0, a, m, sonar_says))
        found = np.array(shares) >= HIDDEN_SHARE
        says = np.array([s for c, *_, s in tiles if c == chunk])
        out["camera"] = {"frames": len(frames), "pipe_found": int(found.sum()), "frames_where_sonar_label_has_pipe": int(says.sum()),
                         "pipe_found_when_sonar_label_has_pipe": int((found & says).sum()),
                         "pipe_found_when_sonar_label_has_none": int((found & ~says).sum())}
        camera_found[chunk] = dict(zip((p.stem for p in frames), found))
    results[chunk] = out
    print(chunk, json.dumps(out), flush=True)

pooled = {}  # sonar windows of all evaluated chunks together (mAP doesn't pool, so per chunk only)
for band in ("HF", "LF"):
    s = [v[f"sonar_{band}"]["detector"] for v in results.values() if f"sonar_{band}" in v]
    found, pipe = (sum(x["pipe_windows_found"][i] for x in s) for i in (0, 1))
    empty, none = (sum(x["no_pipe_windows_left_empty"][i] for x in s) for i in (0, 1))
    pooled[f"sonar_{band}"] = {"pipe_windows_found": [found, pipe], "no_pipe_windows_left_empty": [empty, none],
                               "balanced_accuracy": round((found / pipe + empty / none) / 2, 3) if pipe and none else None}


def presence(rows):
    """Camera "pipe found" against frame labels: rows of (chunk, label, found)."""
    yes, no = [f for _, v, f in rows if v == "yes"], [f for _, v, f in rows if v == "no"]
    return {"visible_frames_found": [int(sum(yes)), len(yes)], "not_visible_frames_marked": [int(sum(no)), len(no)],
            "unsure_frames_left_out": sum(v == "unsure" for _, v, _ in rows),
            "balanced_accuracy": round((sum(yes) / len(yes) + 1 - sum(no) / len(no)) / 2, 3) if yes and no else None}


assert presence([("c", "yes", True), ("c", "no", True), ("c", "no", False), ("c", "unsure", True)])["balanced_accuracy"] == 0.75
labels_file = root / "labels/camera_presence_unseen.csv"  # frame-level labels for chunks with no camera masks
def ious(rows):
    """Pipe and mean IoU, counting pixels over all the frames given: rows of (chunk, pipe and, pipe or, bg and, bg or)."""
    pi, pu, bi, bu = (int(sum(r[k] for r in rows)) for k in (1, 2, 3, 4))
    return {"frames": len(rows), "pipe_iou": round(pi / pu, 3), "mean_iou": round((pi / pu + bi / bu) / 2, 3)}


assert ious([("c", 1, 2, 2, 3)]) == {"frames": 1, "pipe_iou": 0.5, "mean_iou": round((0.5 + 2 / 3) / 2, 3)}
reference_iou = {"outlines": "labels/camera_masks_unseen: Segment Anything inside boxes drawn around points placed by Claude, candidates chosen by Claude (provisional)",
                 "pooled": ious(ref_sums), **{c: ious([r for r in ref_sums if r[0] == c]) for c in sorted({r[0] for r in ref_sums})}} if ref_sums else None
camera_check = None
if labels_file.exists() and camera_found:
    lab = pd.read_csv(labels_file, comment="#", dtype={"stem": str})
    assert set(lab.pipe_visible) <= {"yes", "no", "unsure"}, f"{labels_file.name}: pipe_visible must be yes, no or unsure"
    rows = [(r.chunk, r.pipe_visible, bool(camera_found[r.chunk][r.stem])) for r in lab.itertuples() if r.stem in camera_found.get(r.chunk, {})]
    camera_check = {"labels": str(labels_file.relative_to(root)), "pooled": presence(rows),
                    **{c: presence([x for x in rows if x[0] == c]) for c in camera_found}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "pooled": pooled, "sonar_model": SONAR_MODEL.name, "camera_frame_check": camera_check, "camera_reference_iou": reference_iou,
    "note": "Sonar: every 5th window, all labels. Camera: one frame every 5 s; no camera labels exist outside Chunk 2.",
    "score_threshold_for_seen": SCORE, "camera_pipe_share_for_found": HIDDEN_SHARE, "chunks": results}, indent=2) + "\n")

# Figure: every sampled frame where the camera model marked a pipe (M1 mask in yellow), to judge by eye.
if OUT == DEFAULT_OUT and len(CHUNKS) == 4:  # a run on a subset or another detector leaves the full figure alone
    font = ImageFont.load_default(size=14)
    tw, th = 320, 180
    marked = [t for t in tiles if t[3].mean() >= HIDDEN_SHARE]
    sheet = Image.new("RGB", (6 * tw, max((len(marked) + 5) // 6, 1) * th), "white")
    for k, (chunk, t, a, m, says) in enumerate(marked):
        f = a.copy()
        f[m] = (0.55 * f[m] + 0.45 * np.array([255, 220, 0])).astype("uint8")
        tile = Image.fromarray(f).resize((tw, th))
        ImageDraw.Draw(tile).text((5, 3), f"{chunk} t+{t:.0f}s  pipe {m.mean() * 100:.0f}%  sonar label: {'pipe' if says else 'none'}",
                                  fill="white", font=font, stroke_width=2, stroke_fill="black")
        sheet.paste(tile, ((k % 6) * tw, (k // 6) * th))
    sheet.save(root / "docs/figures/m6_held_out_camera.jpg", quality=85)

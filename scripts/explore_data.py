"""Summarise a SubPipe chunk (milestone M0) and draw the figures used in docs/DATA.md.

Usage: .venv/bin/python -I scripts/explore_data.py [DATA_DIR] [FIGURE_DIR]
Defaults: data/SubPipeMini/DATA and docs/figures. Prints a JSON summary; every number in docs/DATA.md comes from it.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent.parent
data = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/SubPipeMini/DATA"
figs = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "docs/figures"
figs.mkdir(parents=True, exist_ok=True)


def stamps(folder, suffix):
    """Timestamps (seconds) taken from file names such as 1693573934.247.jpg or 1693573935.081_label.png."""
    return np.array(sorted(float(p.name[: -len(suffix)]) for p in (data / folder).glob(f"*{suffix}")))


def rate(t):
    return round(float((len(t) - 1) / (t[-1] - t[0])), 2)


summary = {}

# Navigation: one row per Cam0 frame.
nav = pd.read_csv(data / "EstimatedState.csv")
nav.columns = [c.strip() for c in nav.columns]
step = np.hypot(np.diff(nav["x (m)"]), np.diff(nav["y (m)"]))
speed = np.hypot(nav["u (m/s)"], nav["v (m/s)"])
summary["navigation"] = {
    "rows": len(nav),
    "start_utc": str(pd.to_datetime(nav["timestamp"].iloc[0], unit="s")),
    "duration_s": round(float(nav["timestamp"].iloc[-1] - nav["timestamp"].iloc[0]), 1),
    "track_length_m": round(float(step.sum()), 1),
    "speed_m_s_median": round(float(speed.median()), 2),
    "depth_m_range": [round(float(nav["depth (m)"].min()), 1), round(float(nav["depth (m)"].max()), 1)],
    "altitude_m_range": [round(float(nav["alt (m)"].min()), 1), round(float(nav["alt (m)"].max()), 1)],
    "other_logs": sorted(p.stem for p in data.glob("*.csv") if p.stem != "EstimatedState"),
}

# Cameras.
cam0, cam1 = stamps("Cam0_images", ".jpg"), stamps("Cam1_images", ".jpg")
summary["cameras"] = {
    "cam0_frames": len(cam0),
    "cam0_fps": rate(cam0),
    "cam0_size": Image.open(next((data / "Cam0_images").glob("*.jpg"))).size,
    "cam1_frames": len(cam1),
    "cam1_fps": rate(cam1),
    "cam1_size": Image.open(next((data / "Cam1_images").glob("*.jpg"))).size,
    "cam0_frames_with_navigation_row": int(np.isin(np.round(cam0, 3), np.round(nav["timestamp"], 3)).sum()),
}

# Segmentation: images plus label masks.
seg = data / "Segmentation"
masks = sorted(seg.glob("*_label.png"))
# Labelled frames are saved as .jpg or .png; their masks as <timestamp>_label.png.
seg_files = [p for p in seg.iterdir() if p.suffix in (".jpg", ".png") and not p.name.endswith("_label.png")]
seg_imgs = np.array(sorted(float(p.stem) for p in seg_files))
image_of = lambda mask: next(seg / (mask.name[: -len("_label.png")] + ext) for ext in (".jpg", ".png") if (seg / (mask.name[: -len("_label.png")] + ext)).exists())
pixel_values = Counter()
pipe_share = []
for m in masks[:: max(1, len(masks) // 60)]:  # about 60 masks, spread across the chunk
    a = np.array(Image.open(m))
    flat = a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1, 1)
    pixel_values.update(tuple(int(x) for x in v) for v in np.unique(flat, axis=0))
    pipe_share.append(float(flat.any(axis=1).mean()))
summary["segmentation"] = {
    "images": len(seg_imgs),
    "images_by_type": dict(Counter(p.suffix for p in seg_files)),
    "masks": len(masks),
    "masks_without_image": sum(1 for m in masks if not any((seg / (m.name[: -len("_label.png")] + e)).exists() for e in (".jpg", ".png"))),
    "mask_modes": dict(Counter(Image.open(m).mode for m in masks)),
    "mask_size": Image.open(masks[0]).size,
    "mask_values_seen": sorted(pixel_values),
    "nonzero_pixel_share_median": round(float(np.median(pipe_share)), 3),
    "images_matching_cam0_frames": int(np.isin(np.round(seg_imgs, 3), np.round(cam0, 3)).sum()),
}

# Side-scan sonar: images plus pipeline boxes.
for band in ("HF", "LF"):
    d = data / f"SSS_{band}_images"
    imgs = sorted((d / "Image").glob("*.pbm"))
    yolo = [p for p in (d / "YOLO_Annotation").glob("*.txt") if p.name != "classes.txt"]
    coco = json.loads((d / "COCO_Annotation/coco_format.json").read_text())
    first = Image.open(imgs[0])
    summary[f"sonar_{band}"] = {
        "images": len(imgs),
        "image_mode": first.mode,
        "image_size": first.size,
        "yolo_label_files": len(yolo),
        "classes": (d / "YOLO_Annotation/classes.txt").read_text().split(),
        "coco_images": len(coco.get("images", [])),
        "coco_boxes": len(coco.get("annotations", [])),
        "coco_categories": [c["name"] for c in coco.get("categories", [])],
        "images_per_second": rate(stamps(f"SSS_{band}_images/Image", ".pbm")),
    }

print(json.dumps(summary, indent=2, default=str))

# Figure 1: the vehicle's track, with labelled camera frames and sonar images marked.
t0 = nav["timestamp"].iloc[0]
fig, ax = plt.subplots(figsize=(10, 3.6))
sc = ax.scatter(nav["y (m)"], nav["x (m)"], c=nav["timestamp"] - t0, s=4, cmap="viridis")
near = lambda ts: np.searchsorted(nav["timestamp"].values, ts).clip(0, len(nav) - 1)
i = near(seg_imgs)
ax.scatter(nav["y (m)"].iloc[i], nav["x (m)"].iloc[i] + 6, s=4, c="crimson", marker="|", label="camera frames with pipe labels (drawn just above the track)")
ax.set_xlabel("east (m)")
ax.set_ylabel("north (m)")
ax.legend(loc="upper left", fontsize=8)
fig.colorbar(sc, label="seconds since start", orientation="vertical", fraction=0.03, pad=0.01)
ax.set_title(f"SubPipe chunk: vehicle track, {summary['navigation']['track_length_m']} m in {summary['navigation']['duration_s']} s")
fig.tight_layout()
fig.savefig(figs / "track.png", dpi=110)

# Figure 2: one labelled camera frame with its mask, and one sonar image with its box.
# A frame with a clearly visible pipe: the 90th percentile of labelled pipe area.
share = {m: (np.array(Image.open(m).convert("L").resize((338, 190))) > 0).mean() for m in masks[::5]}
pick = sorted(share, key=share.get)[int(0.9 * (len(share) - 1))]
img = Image.open(image_of(pick))
mask = np.array(Image.open(pick).convert("L")) > 0
sonar_img = sorted((data / "SSS_HF_images/Image").glob("*.pbm"))
labelled = [p for p in sonar_img if (data / "SSS_HF_images/YOLO_Annotation" / (p.stem + ".txt")).exists()]
s_img = labelled[len(labelled) // 2]
s = Image.open(s_img)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(img)
axes[0].set_title(f"camera frame {pick.name.replace('_label.png', '')}")
axes[1].imshow(img)
axes[1].imshow(np.ma.masked_where(~mask, mask), alpha=0.5, cmap="autumn")
axes[1].set_title("pipe label overlaid")
axes[2].imshow(s, cmap="gray", aspect="auto")
for line in (data / "SSS_HF_images/YOLO_Annotation" / (s_img.stem + ".txt")).read_text().splitlines():
    _, cx, cy, w, h = map(float, line.split())
    W, H = s.size
    axes[2].add_patch(plt.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False, ec="red", lw=1.5))
axes[2].set_title(f"side-scan sonar (HF) {s_img.stem}, pipeline box in red")
for a in axes:
    a.axis("off")
fig.tight_layout()
fig.savefig(figs / "samples.png", dpi=90)

# Figure 3: twelve labelled frames spread across the chunk (pipe label tinted red), to show collars, cover and the
# change in water colour.
by_time = sorted(masks, key=lambda m: float(m.name[: -len("_label.png")]))
W, H = 452, 254
sheet = Image.new("RGB", (W * 4, H * 3), "white")
for k, m in enumerate(by_time[int(i)] for i in np.linspace(0, len(by_time) - 1, 12)):
    frame = np.array(Image.open(image_of(m)).convert("RGB").resize((W, H)))
    pipe = np.array(Image.open(m).convert("L").resize((W, H))) > 0
    frame[pipe] = (0.55 * frame[pipe] + 0.45 * np.array([255, 40, 40])).astype("uint8")
    tile = Image.fromarray(frame)
    t = float(m.name[: -len("_label.png")]) - t0
    ImageDraw.Draw(tile).text((6, 6), f"t+{t:.0f}s, pipe {pipe.mean() * 100:.0f}% of frame", fill="white")
    sheet.paste(tile, ((k % 4) * W, (k // 4) * H))
sheet.save(figs / "labelled_frames.jpg", quality=85)

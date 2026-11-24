"""Milestone M3: turn the M1 camera model and the M2 sonar detector into an event listing along the route, with a map.

Usage: .venv/bin/python -I scripts/build_events.py [DATA_DIR]
Needs models/m1_segmentation.pt and models/m2_sonar_detection.pt, so run the M1 and M2 scripts first. Writes
docs/results/m3_events.csv, docs/results/m3_events.json and docs/figures/m3_route.png. When
labels/hand_events_360-420s.csv has rows (see docs/LABELLING.md), it also scores the listing against them (NFR-03).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import v2

root = Path(__file__).resolve().parent.parent
data = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/SubPipeMini/DATA"
hand_file = root / "labels/hand_events_360-420s.csv"
HAND = (360, 420)  # the hand-marked minute: an M1 and M2 test minute, so neither model trained on it
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.hub.set_dir(str(root / "models/torch"))
# Event rules, set by looking at training minutes only (120-180 s, 240-360 s), before any hand labels existed.
CAM_STEP = 6  # every 6th Camera 0 frame: 5 a second
HIDDEN_SHARE, HIDDEN_MIN_S = 0.02, 1.0  # camera "pipe not visible": pipe under 2% of the frame for at least 1 s
COLLAR_RATIO, COLLAR_GAP_S = 1.2, 3.0  # collar: pipe 20% wider than its 12 s running median, and widest within 3 s
SONAR_SCORE, SONAR_MIN_S = 0.5, 2.0  # sonar "pipe not visible": most windows covering a second have no pipe, for 2 s
WINDOW_S = 25  # a sonar image spans the last 25 s of pings; its top row is the newest
MATCH_S = 2.0  # scoring: a listed collar matches a hand-marked one within 2 s


def runs(t, flag, min_s):
    """(start, end) seconds of each run where flag holds for at least min_s."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(flag, int), 0]))
    return [(float(t[a]), float(t[b - 1])) for a, b in zip(edges[::2], edges[1::2]) if t[b - 1] - t[a] >= min_s]


def find_collars(t, width, share, per_s):
    """(time, width / usual width) where the pipe is COLLAR_RATIO wider than its running median and widest nearby."""
    width, visible = pd.Series(np.asarray(width, float)), np.asarray(share) >= HIDDEN_SHARE
    base = width.where(visible).rolling(int(12 * per_s), center=True, min_periods=1).median()  # usual width, pipe in view
    top = width == width.rolling(int(2 * COLLAR_GAP_S * per_s) + 1, center=True, min_periods=1).max()
    hit = np.flatnonzero(top & (width >= COLLAR_RATIO * base) & visible)
    hit = hit[np.r_[True, np.diff(np.asarray(t)[hit]) > COLLAR_GAP_S]]  # one per flat-topped peak
    return [(float(t[i]), float(width[i] / base[i])) for i in hit]


def matched(hand, listed, tol):
    """How many hand-marked times pair one-to-one with listed times within tol, nearest pairs first."""
    pairs = sorted((abs(h - x), i, j) for i, h in enumerate(hand) for j, x in enumerate(listed) if abs(h - x) <= tol)
    used_h, used_l = set(), set()
    for _, i, j in pairs:
        if i not in used_h and j not in used_l:
            used_h.add(i)
            used_l.add(j)
    return len(used_h)


def overlap(a, b, spans):
    """Share of [a, b] covered by spans; a single moment counts as covered if it falls inside one."""
    if b <= a:
        return float(any(lo <= a <= hi for lo, hi in spans))
    return sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in spans) / (b - a)


def score_marks(hand, collars, cam_hidden):
    """NFR-03: hand marks (time_s, event) against the listed collars and camera "pipe not visible" stretches."""
    unknown = set(hand.event) - {"collar", "hidden_start", "hidden_end"}
    assert not unknown, f"{hand_file.name}: unknown events {unknown}"
    assert pd.to_numeric(hand.time_s).between(*HAND).all(), f"{hand_file.name}: times must be within {HAND} s"
    marks = hand[hand.event != "collar"].sort_values("time_s")
    assert marks.event.tolist() == ["hidden_start", "hidden_end"] * (len(marks) // 2), f"{hand_file.name}: each hidden_start needs a hidden_end after it"
    hand_hidden = [tuple(p) for p in marks.time_s.to_numpy(float).reshape(-1, 2)]
    hand_collars = hand[hand.event == "collar"].time_s.astype(float).tolist()
    listed_collars = [c for c, _ in collars if HAND[0] <= c <= HAND[1]]
    listed_hidden = [(a, b) for a, b in cam_hidden if a <= HAND[1] and b >= HAND[0]]
    n_collars = matched(hand_collars, listed_collars, MATCH_S)
    n_hidden = sum(overlap(a, b, cam_hidden) >= 0.5 for a, b in hand_hidden)
    total = len(hand_collars) + len(hand_hidden)
    return {
        "status": "scored",
        "collars": {"hand_marked": len(hand_collars), "listed": len(listed_collars), "matched_within_s": MATCH_S, "matched": n_collars},
        "pipe_not_visible_camera": {"hand_marked": len(hand_hidden), "found_half_covered": int(n_hidden), "listed": len(listed_hidden),
                                    "listed_with_no_hand_mark": int(sum(overlap(a, b, hand_hidden) == 0 for a, b in listed_hidden))},
        "event_recall": round((n_collars + n_hidden) / total, 3) if total else None,
        "collar_precision": round(n_collars / len(listed_collars), 3) if listed_collars else None,
    }


demo = pd.DataFrame({"time_s": [365.0, 371.0, 375.0, 390.0], "event": ["collar", "hidden_start", "hidden_end", "collar"]})
demo_score = score_marks(demo, [(365.5, 1.3), (380.0, 1.4)], [(370.0, 374.0)])  # one collar matched, the stretch 3/4 covered
assert demo_score["event_recall"] == round(2 / 3, 3) and demo_score["collar_precision"] == 0.5
assert runs(np.arange(10.0), [0, 1, 1, 1, 0, 0, 1, 0, 1, 1], 1) == [(1.0, 3.0), (8.0, 9.0)]
tt = np.arange(0, 30, 0.2)
bumps = 0.25 + 0.15 * np.clip(1 - np.abs(tt - 10), 0, None) + 0.15 * np.clip(1 - np.abs(tt - 20), 0, None)
assert [round(c) for c, _ in find_collars(tt, bumps, np.full_like(tt, 0.2), 5)] == [10, 20]
assert matched([10, 20], [10.5, 19, 30], 2) == 2 and matched([10, 11], [10.5], 2) == 1
assert overlap(0, 10, [(5, 20)]) == 0.5 and overlap(3, 3, [(0, 5)]) == 1.0

started = time.time()  # NFR-06: processing time, from loading the models to the finished listing

# Navigation: position and distance along the vehicle's track at any moment.
nav = pd.read_csv(data / "EstimatedState.csv").rename(columns=str.strip)
t0 = float(nav["timestamp"].iloc[0])
nav_t, east, north = nav["timestamp"].to_numpy() - t0, nav["y (m)"].to_numpy(), nav["x (m)"].to_numpy()
along = np.r_[0, np.cumsum(np.hypot(np.diff(east), np.diff(north)))]

# Camera: the M1 model's pipe mask, 5 frames a second. Width = 90th percentile of the pipe's share of each image row.
seg = deeplabv3_mobilenet_v3_large(num_classes=1, weights_backbone="IMAGENET1K_V1")
seg.load_state_dict(torch.load(root / "models/m1_segmentation.pt"))
seg.eval().to(device)
normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
frames = sorted((data / "Cam0_images").glob("*.jpg"), key=lambda p: float(p.stem))[::CAM_STEP]
cam = {"t": [], "share": [], "width": []}
with torch.no_grad():
    for i in range(0, len(frames), 16):
        batch = []
        for p in frames[i : i + 16]:
            img = Image.open(p)
            img.draft("RGB", (640, 360))
            batch.append(np.array(img.convert("RGB").resize((640, 360), Image.BILINEAR)))
        x = normalize(torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255).to(device)
        masks = (seg(x)["out"][:, 0] > 0).cpu().numpy()
        cam["t"] += [float(p.stem) - t0 for p in frames[i : i + 16]]
        cam["share"] += list(masks.mean(axis=(1, 2)))
        cam["width"] += list(np.percentile(masks.mean(axis=2), 90, axis=1))
cam = pd.DataFrame(cam)

# Sonar: the M2 detector on every window. Its best box's rows map back to the ping times they show, and each second is
# called "pipe present" when most of the windows that contain it box the pipe there.
det = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
det.roi_heads.box_predictor = FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features, 2)
det.load_state_dict(torch.load(root / "models/m2_sonar_detection.pt"))
det.eval().to(device)
sonar = {}
for name in ("HF", "LF"):
    paths = sorted((data / f"SSS_{name}_images/Image").glob("*.pbm"), key=lambda p: float(p.stem))
    win_t, spans = [], []
    with torch.no_grad():
        for i in range(0, len(paths), 8):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + 8]]
            xs = [torch.from_numpy(np.array(im.resize((1333, round(im.height * 1333 / im.width)), Image.BILINEAR))).permute(2, 0, 1).float().div(255) for im in imgs]
            for p, x, r in zip(paths[i : i + 8], xs, det([x.to(device) for x in xs])):
                t = float(p.stem) - t0
                win_t.append(t)
                if len(r["scores"]) and r["scores"][0] >= SONAR_SCORE:  # boxes come sorted by score
                    y0, y1 = r["boxes"][0, [1, 3]].cpu().numpy() * WINDOW_S / x.shape[1]
                    spans.append((t - y1, t - y0))
    grid, win_t, spans = np.arange(0.0, np.floor(max(win_t)) + 1), np.array(win_t), np.array(spans).reshape(-1, 2)
    boxed = ((spans[None, :, 0] <= grid[:, None]) & (grid[:, None] <= spans[None, :, 1])).sum(axis=1)
    covering = ((win_t[None, :] - WINDOW_S <= grid[:, None]) & (grid[:, None] <= win_t[None, :])).sum(axis=1)
    sonar[name] = pd.DataFrame({"t": grid, "present": boxed / np.maximum(covering, 1)})

# Events.
per_s = 1 / float(np.median(np.diff(cam.t)))
cam_hidden = runs(cam.t.to_numpy(), cam.share.rolling(5, center=True, min_periods=1).median() < HIDDEN_SHARE, HIDDEN_MIN_S)
collars = find_collars(cam.t.to_numpy(), cam.width, cam.share, per_s)
rows = [("pipe not visible", "camera", a, b, f"{b - a:.1f} s") for a, b in cam_hidden]
rows += [("collar or joint", "camera", c, c, f"{r:.2f} x usual width") for c, r in collars]
sonar_hidden = {name: runs(s.t.to_numpy(), s.present < 0.5, SONAR_MIN_S) for name, s in sonar.items()}
for name, spans in sonar_hidden.items():
    rows += [("pipe not visible", f"sonar {name}", a, b, f"{b - a:.0f} s") for a, b in spans]
ev = pd.DataFrame(rows, columns=["event", "source", "start_s", "end_s", "detail"]).sort_values(["start_s", "source"], ignore_index=True)
ev.insert(0, "id", [f"E{k:03d}" for k in range(1, len(ev) + 1)])
ev["start_utc"] = pd.to_datetime(t0 + ev.start_s, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-5]
ev["along_track_start_m"], ev["along_track_end_m"] = np.interp(ev.start_s, nav_t, along), np.interp(ev.end_s, nav_t, along)
ev["east_m"], ev["north_m"] = np.interp(ev.start_s, nav_t, east), np.interp(ev.start_s, nav_t, north)
ev = ev[["id", "event", "source", "start_s", "end_s", "start_utc", "along_track_start_m", "along_track_end_m", "east_m", "north_m", "detail"]].round(1)
(root / "docs/results").mkdir(parents=True, exist_ok=True)
ev.to_csv(root / "docs/results/m3_events.csv", index=False)

# Score against the hand-marked minute, if it has been marked.
hand = pd.read_csv(hand_file, comment="#") if hand_file.exists() else pd.DataFrame(columns=["time_s", "event", "note"])
score = score_marks(hand, collars, cam_hidden) if len(hand) else {"status": f"waiting for hand marks in {hand_file.relative_to(root)} (see docs/LABELLING.md)"}

elapsed, footage = time.time() - started, float(cam.t.max())
collar_t = [c for c, _ in collars]
summary = {
    "rules": {"camera_frames_per_s": round(per_s, 1), "hidden_share": HIDDEN_SHARE, "hidden_min_s": HIDDEN_MIN_S,
              "collar_ratio": COLLAR_RATIO, "collar_gap_s": COLLAR_GAP_S, "sonar_score": SONAR_SCORE, "sonar_min_s": SONAR_MIN_S},
    "events": {f"{e} ({s})": int(n) for (e, s), n in ev.groupby(["event", "source"]).size().items()},
    "collar_spacing_median": {"s": round(float(np.median(np.diff(collar_t))), 1), "m": round(float(np.median(np.diff(np.interp(collar_t, nav_t, along)))), 1)},
    "track_m": round(float(along[-1]), 1),
    "models": {name: hashlib.sha256((root / f"models/{name}.pt").read_bytes()).hexdigest()[:12]  # which models made this listing
               for name in ("m1_segmentation", "m2_sonar_detection")},
    "processing": {"footage_s": round(footage, 1), "seconds": round(elapsed, 1), "device": device,
                   "minutes_per_hour_of_footage": round(elapsed / footage * 60, 1)},
    "hand_marked_minute_s": list(HAND),
    "score": score,
}
(root / "docs/results/m3_events.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

# Figure: the route with events, and the signals behind them over time.
fig, (top, low) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.2]})
top.plot(east, north, color="0.6", lw=1, label="vehicle track")
top.scatter(np.interp(collar_t, nav_t, east), np.interp(collar_t, nav_t, north), s=6, c="k", zorder=3, label="collar or joint (camera)")
for label, spans, colour, dy in (("pipe not visible: camera", cam_hidden, "crimson", 4), ("pipe not visible: sonar HF", sonar_hidden["HF"], "darkorange", 8),
                                 ("pipe not visible: sonar LF", sonar_hidden["LF"], "purple", 12)):
    for k, (a, b) in enumerate(spans):
        m = (nav_t >= a) & (nav_t <= b)
        top.plot(east[m], north[m] + dy, "-", color=colour, lw=4, solid_capstyle="butt", label=label if k == 0 else None)
top.set_xlabel("east (m, local grid)")
top.set_ylabel("north (m)")
top.set_title(f"Events along the route ({len(collars)} collars or joints; stretches drawn just north of the track)")
top.legend(fontsize=8, loc="best")
low.axvspan(*HAND, color="0.9", zorder=0, label="hand-marked minute")
low.plot(cam.t, cam.width, lw=0.7, color="C0", label="camera: pipe width (share of a row)")
low.plot(collar_t, np.interp(collar_t, cam.t, cam.width), "k.", ms=5, label="collar or joint")
for k, (a, b) in enumerate(cam_hidden):
    low.axvspan(a, b, color="crimson", alpha=0.3, label="pipe not visible: camera" if k == 0 else None)
for name, colour in (("HF", "darkorange"), ("LF", "purple")):
    low.plot(sonar[name].t, sonar[name].present, color=colour, lw=1, label=f"sonar {name}: share of windows boxing the pipe")
low.set_xlim(0, cam.t.max())
low.set_ylim(0, 1.05)
low.set_xlabel("seconds since the start of the survey")
low.legend(fontsize=7, loc="lower left", ncol=2)
fig.tight_layout()
fig.savefig(root / "docs/figures/m3_route.png", dpi=110)

"""Milestone M4: draft a short description of sampled camera events with a local vision-language model (MLX).

Each event gets two drafts: one from the plain frame, and one from the frame with the M1 model's pipe outlined in
yellow, so reviewers can see whether pointing the model at the pipe helps.

Usage: .venv/bin/python -I scripts/describe_events.py [DATA_DIR]
Needs docs/results/m3_events.csv (run scripts/build_events.py first) and models/m1_segmentation.pt. The model,
Qwen3-VL-8B-Instruct (Apache-2.0) in 4-bit MLX form, downloads once into models/hf. Writes
docs/results/m4_descriptions.csv, docs/results/m4_summary.json, the frames in docs/figures/m4_frames/ and the review
sheet docs/M4_REVIEW.md. When labels/m4_ratings.csv holds ratings from two reviewers, it also scores them (NFR-04).
"""

import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(root / "models/hf"))  # keep the model inside the project, outside git; set before import
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from mlx_vlm import generate, load  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from mlx_vlm.utils import load_config  # noqa: E402
from PIL import Image, ImageChops, ImageFilter  # noqa: E402
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

data = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/SubPipeMini/DATA"
ratings_file = root / "labels/m4_ratings.csv"
MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
N_COLLARS = 9  # sampled evenly along the route, plus every camera "pipe not visible" stretch
PROMPT = ("This is a frame of underwater video from a vehicle surveying a subsea pipeline from about 3 m above the seabed. "
          "The automatic system flagged it as: {event}. Write one or two short sentences for an inspection log describing "
          "what is visible: whether the pipe can be seen, whether it is exposed or partly covered by sand, any marine growth, "
          "debris or damage, and any collar or joint. Say if something is unclear rather than guessing.")
OUTLINE_NOTE = (" A separate pipe-detection model has outlined in yellow where it found the pipe; if there is no yellow "
                "outline, it found none. Treat the outline as a hint that may be wrong, and describe what you can see.")
VERSIONS = {"plain": PROMPT, "outlined": PROMPT + OUTLINE_NOTE}
RATINGS = {"correct", "partly", "wrong"}


def score_ratings(r, keys):
    """NFR-04 per version: share of drafts that both reviewers rated correct, over drafts both have rated."""
    assert not set(r.rating) - RATINGS, f"{ratings_file.name}: ratings must be one of {sorted(RATINGS)}"
    unknown = set(zip(r.id, r.version)) - set(keys)
    assert not unknown, f"{ratings_file.name}: unknown (id, version) pairs {sorted(unknown)}"
    out = {}
    for version, g in r.groupby("version"):
        reviewers = sorted(set(g.reviewer))
        out[version] = {"by_reviewer": {name: {"rated": int((g.reviewer == name).sum()), "share_correct": round(float((g[g.reviewer == name].rating == "correct").mean()), 3)}
                                        for name in reviewers}}
        if len(reviewers) < 2:
            out[version]["status"] = f"waiting for two reviewers (so far: {reviewers})"
            continue
        # shortcut: a third reviewer is ignored; average all pairs if that ever happens
        two = g.pivot_table(index="id", columns="reviewer", values="rating", aggfunc="first")[reviewers[:2]].dropna()
        out[version].update({"status": "scored", "reviewers": reviewers[:2], "rated_by_both": len(two),
                             "correct_by_both": round(float((two == "correct").all(axis=1).mean()), 3),
                             "reviewers_agree": round(float((two.iloc[:, 0] == two.iloc[:, 1]).mean()), 3)})
    return out


demo = pd.DataFrame({"id": ["E1", "E1", "E2", "E2", "E3"], "version": ["plain"] * 5, "reviewer": ["a", "b", "a", "b", "a"],
                     "rating": ["correct", "correct", "correct", "partly", "wrong"]})
assert score_ratings(demo, [(i, "plain") for i in ("E1", "E2", "E3")])["plain"]["correct_by_both"] == 0.5  # E3 rated once


def outline(img, seg):
    """The frame with the M1 model's pipe mask drawn as a yellow outline."""
    x = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        torch.from_numpy(np.array(img.resize((640, 360), Image.BILINEAR))).permute(2, 0, 1).float()[None] / 255)
    with torch.no_grad():
        mask = (seg(x)["out"][0, 0] > 0).numpy()
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)
    edge = np.array(ImageChops.difference(m.filter(ImageFilter.MaxFilter(5)), m.filter(ImageFilter.MinFilter(5)))) > 0
    a = np.array(img)
    a[edge] = (255, 230, 0)
    return Image.fromarray(a)


if "--ratings-only" in sys.argv:  # re-score the ratings without running the model again
    drafts = pd.read_csv(root / "docs/results/m4_descriptions.csv")
    summary = json.loads((root / "docs/results/m4_summary.json").read_text())
    summary["ratings"] = score_ratings(pd.read_csv(ratings_file, comment="#"), list(zip(drafts.id, drafts.version)))
    (root / "docs/results/m4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["ratings"], indent=2))
    sys.exit(0)

# The sample: every camera "pipe not visible" stretch, plus collars spread evenly along the route.
ev = pd.read_csv(root / "docs/results/m3_events.csv")
cam = ev[ev.source == "camera"]
collars = cam[cam.event == "collar or joint"]
sample = pd.concat([cam[cam.event == "pipe not visible"], collars.iloc[np.linspace(0, len(collars) - 1, N_COLLARS).astype(int)]]).sort_values("start_s")
t0 = float(pd.read_csv(data / "EstimatedState.csv", nrows=1).rename(columns=str.strip)["timestamp"].iloc[0])
frames = sorted((data / "Cam0_images").glob("*.jpg"), key=lambda p: float(p.stem))
frame_t = np.array([float(p.stem) for p in frames]) - t0
out_dir = root / "docs/figures/m4_frames"
out_dir.mkdir(parents=True, exist_ok=True)

seg = deeplabv3_mobilenet_v3_large(num_classes=1, weights_backbone=None)
seg.load_state_dict(torch.load(root / "models/m1_segmentation.pt"))
seg.eval()
model, processor = load(MODEL)
config = load_config(MODEL)
rows, peak = [], 0.0
for e in sample.itertuples():
    t = (e.start_s + e.end_s) / 2  # a collar's moment, or the middle of a stretch
    img = Image.open(frames[int(np.abs(frame_t - t).argmin())])
    img.draft("RGB", (1014, 570))
    img = img.convert("RGB").resize((960, 540))
    for version, image in (("plain", img), ("outlined", outline(img, seg))):
        path = out_dir / (f"{e.id}.jpg" if version == "plain" else f"{e.id}_outlined.jpg")
        image.save(path, quality=85)
        prompt = apply_chat_template(processor, config, VERSIONS[version].format(event=e.event), num_images=1)
        start = time.time()
        r = generate(model, processor, prompt, image=str(path), max_tokens=120, temperature=0.0, verbose=False)
        rows.append({"id": e.id, "version": version, "event": e.event, "time_s": round(t, 1), "frame": f"figures/m4_frames/{path.name}",
                     "description": " ".join(r.text.split()), "seconds": round(time.time() - start, 1), "finish": r.finish_reason})
        peak = max(peak, float(r.peak_memory))
        print(e.id, version, rows[-1]["seconds"], "s:", rows[-1]["description"], flush=True)
out = pd.DataFrame(rows)
(root / "docs/results").mkdir(parents=True, exist_ok=True)
out.to_csv(root / "docs/results/m4_descriptions.csv", index=False)

if not ratings_file.exists():
    ratings_file.write_text("id,version,reviewer,rating,note\n")
ratings = pd.read_csv(ratings_file, comment="#")
not_visible = out.description.str.contains("not visible|not clearly visible", case=False)  # a crude text check, not a rating
summary = {
    "model": MODEL, "model_licence": "Apache-2.0", "runtime": "mlx-vlm (MIT), on the Mac's graphics chip",
    "prompts": VERSIONS, "settings": {"temperature": 0.0, "max_tokens": 120, "frame_size": [960, 540]},
    "events": int(out.id.nunique()), "by_event": sample.event.value_counts().to_dict(),
    "drafts_saying_pipe_not_visible": {f"{v} / {e}": f"{int(n.sum())} of {len(n)}" for (v, e), n in not_visible.groupby([out.version, out.event])},
    "seconds_per_draft": {"median": float(out.seconds.median()), "max": float(out.seconds.max())},
    "peak_memory_gb": round(peak, 1), "cost": "$0: runs locally, no API",
    "ratings": score_ratings(ratings, list(zip(out.id, out.version))) if len(ratings) else {"status": f"waiting for ratings in {ratings_file.relative_to(root)} (see docs/M4_REVIEW.md)"},
}
(root / "docs/results/m4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ("events", "drafts_saying_pipe_not_visible", "seconds_per_draft", "peak_memory_gb", "ratings")}, indent=2))

# Review sheet: each frame next to its draft, both versions, for two reviewers to rate independently.
lines = [
    "# M4 review sheet: draft event descriptions",
    "",
    f"Generated by `scripts/describe_events.py` with {MODEL} (Apache-2.0), running locally. Every event has two drafts:",
    "one from the plain frame, and one from the frame with the pipe-detection model's outline in yellow. Each draft sits",
    "under the frame the model saw. Two reviewers rate every draft on their own, without comparing notes, and add one",
    "row per draft to [`labels/m4_ratings.csv`](../labels/m4_ratings.csv): `id,version,reviewer,rating,note`, where",
    "`version` is `plain` or `outlined`. The file already holds one provisional set by Claude (reviewer",
    "`claude-provisional`); add your own rows under your name without looking at those first.",
    "",
    "| Rating | Meaning |",
    "|---|---|",
    "| `correct` | Everything it says is visible in the frame, and it misses nothing important |",
    "| `partly` | Mostly right, but misses something important or adds a detail the frame doesn't support |",
    "| `wrong` | Says something the frame doesn't show (damage, debris, a pipe that isn't there), or misses that the pipe isn't visible |",
    "",
    "NFR-04 is the share of drafts that both reviewers rate `correct` (target 80%), reported for each version. Judge the",
    "plain frame for what is really there: the outline can be wrong. Re-run the script to score.",
]
for (event_id, event, t), g in out.groupby(["id", "event", "time_s"], sort=False):
    lines += ["", f"## {event_id}: {event}, {t} s"]
    for r in g.itertuples():
        lines += ["", f"**{r.version}**", "", f"![{event_id} {r.version}]({r.frame})", "", f"> {r.description}"]
(root / "docs/M4_REVIEW.md").write_text("\n".join(lines) + "\n")

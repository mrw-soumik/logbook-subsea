"""Reference pipe outlines for camera frames of the unseen chunks, so the M1 camera model can be scored there with IoU.

Segment Anything (facebook/sam-vit-base, Apache-2.0), a model unrelated to M1, proposes three outlines inside a box
drawn around the pipe from the points in labels/camera_points_unseen.csv. A person then checks every candidate on the
sheet and records the one that follows the pipe, or "reject", in labels/camera_outline_choice.csv. Only accepted
outlines are written. Points alone don't work here: on these low-contrast frames they select almost the whole image.

Usage: .venv/bin/python -I scripts/outline_pipe_sam.py   (run, check the sheet, fill in the choice file, run again)
Writes docs/figures/m6_outline_candidates.jpg and, once choices exist, labels/camera_masks_unseen/<chunk>_<stem>.png.
"""

import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(root / "models/hf"))  # keep the model inside the project, outside git; set before import
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from transformers import SamModel, SamProcessor  # noqa: E402

MODEL, MARGIN = "facebook/sam-vit-base", 70  # the box reaches MARGIN pixels either side of the pipe's centre points
choice_file, out_dir = root / "labels/camera_outline_choice.csv", root / "labels/camera_masks_unseen"
points = pd.read_csv(root / "labels/camera_points_unseen.csv", comment="#", dtype={"stem": str})
choices = pd.read_csv(choice_file, comment="#", dtype={"stem": str, "choice": str}).set_index("stem").choice.to_dict() if choice_file.exists() else {}
processor, model = SamProcessor.from_pretrained(MODEL), SamModel.from_pretrained(MODEL).eval()

rows, font = [], ImageFont.load_default(size=15)
for r in points.itertuples():
    img = Image.open(root / "data/SubPipe/DATA" / r.chunk / "Cam0_images" / f"{r.stem}.jpg")
    img.draft("RGB", (676, 380))
    img = img.convert("RGB").resize((640, 360), Image.BILINEAR)  # the size M1 sees
    xs = [float(p.split()[0]) for p in r.points.split(";")]
    box = [max(min(xs) - MARGIN, 0), 0, min(max(xs) + MARGIN, 640), 360]
    inputs = processor(images=img, input_boxes=[[box]], return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = processor.image_processor.post_process_masks(out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0][0].numpy()
    if r.stem in choices and choices[r.stem] != "reject":
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(masks[int(choices[r.stem])].astype(np.uint8) * 255).save(out_dir / f"{r.chunk}_{r.stem}.png")
    row = [img.resize((320, 180))]
    for k in range(3):
        a = np.array(img)
        a[masks[k]] = (0.55 * a[masks[k]] + 0.45 * np.array([255, 220, 0])).astype("uint8")
        row.append(Image.fromarray(a).resize((320, 180)))
    for k, tile in enumerate(row):
        label = f"{r.chunk} t+{r.t:.0f}s" if k == 0 else f"candidate {k - 1}: {masks[k - 1].mean() * 100:.0f}%"
        ImageDraw.Draw(tile).text((5, 3), label, fill="white", font=font, stroke_width=2, stroke_fill="black")
    rows.append(row)
    print(r.chunk, r.stem, "choice:", choices.get(r.stem, "-"), flush=True)

sheet = Image.new("RGB", (4 * 320, len(rows) * 180), "white")
for i, row in enumerate(rows):
    for k, tile in enumerate(row):
        sheet.paste(tile, (k * 320, i * 180))
sheet.save(root / "docs/figures/m6_outline_candidates.jpg", quality=85)

"""Write data/labelling/camera_360-420s.mp4: Camera 0 from 360 to 420 s at 15 frames a second, with the survey time in
the corner, for marking events by hand (docs/LABELLING.md). Needs ffmpeg.

Usage: .venv/bin/python -I scripts/make_labelling_clip.py [DATA_DIR]
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent.parent
data = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/SubPipeMini/DATA"
START, END = 360, 420
out = root / f"data/labelling/camera_{START}-{END}s.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
t0 = float(pd.read_csv(data / "EstimatedState.csv", nrows=1).rename(columns=str.strip)["timestamp"].iloc[0])
frames = [p for p in sorted((data / "Cam0_images").glob("*.jpg"), key=lambda p: float(p.stem)) if START <= float(p.stem) - t0 < END][::2]
font = ImageFont.load_default(size=40)
ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "15",
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
for p in frames:
    img = Image.open(p)
    img.draft("RGB", (1352, 760))
    img = img.convert("RGB").resize((1280, 720))
    ImageDraw.Draw(img).text((20, 16), f"t = {float(p.stem) - t0:.1f} s", fill="white", font=font, stroke_width=3, stroke_fill="black")
    ffmpeg.stdin.write(img.tobytes())
ffmpeg.stdin.close()
assert ffmpeg.wait() == 0, "ffmpeg failed"
print(f"{out} ({len(frames)} frames)")

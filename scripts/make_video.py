"""M6: a two-minute captioned video built from the repo's own figures, plus a narration script for an optional voiceover.

Usage: .venv/bin/python -I scripts/make_video.py   (needs ffmpeg)
Writes docs/video/logbook_subsea_2min.mp4 and docs/VIDEO_SCRIPT.md. Silent with captions, so it works where video
autoplays muted. Every number on screen comes from docs/results.
"""

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent.parent
figs, res = root / "docs/figures", root / "docs/results"
m1, m2, m3 = (json.loads((res / f).read_text()) for f in ("m1_segmentation.json", "m2_sonar_detection.json", "m3_events.json"))
W, H = 1280, 720
NAVY, INK, MUTED, BLUE = (13, 27, 46), (240, 244, 248), (160, 174, 192), (37, 99, 235)
recall = m3["score"].get("event_recall")
fixed = [m2["test"][b]["fixed_box_baseline"]["ap50"] for b in ("HF", "LF")]
collars = m3["events"]["collar or joint (camera)"]
SLIDES = [  # (seconds, title, image, caption)
    (8, "Logbook Subsea", None, "From subsea survey footage to a reviewed event listing, with a person approving every item. "
     "A demonstration on public data."),
    (10, "The problem", "labelled_frames.jpg", "Inspection teams watch hours of survey video and log every finding by hand: "
     "the time, the position and what they saw."),
    (10, "The data", "samples.png", "SubPipe, a public survey (Oceanscan-MST, CC BY 4.0): 9 minutes along 465 m of pipe, with "
     "camera, side-scan sonar and navigation."),
    (10, "Honest testing", "track.png", "Models are scored on minutes they never saw, with gaps around them. Neighbouring "
     "frames, and sonar images that share 96% of their pixels, would otherwise leak the answers."),
    (10, "Finding the pipe on camera", "m1_predictions.jpg", f"Mean IoU {m1['test']['mean_iou']} on held-out minutes "
     f"(the pipe alone: {m1['test']['pipe_iou']}). Yellow: found. Magenta: missed. Cyan: false alarm."),
    (10, "Where it fails", "m1_worst_frames.jpg", "Buried pipe and cloudy water. The failures are published next to the "
     "average, because they are the cases an inspector cares about."),
    (10, "Sonar", "m2_sonar_test.jpg", f"The vehicle follows the pipe, so a box that never moves already scores "
     f"{min(fixed):.2f} to {max(fixed):.2f} mAP. The real test is knowing when the pipe isn't there."),
    (12, "Events along the route", "m3_route.png", f"{collars} collars about 5 m apart, and the stretches where each sensor "
     "loses the pipe. Near the end the sonar loses it while the camera still sees it."),
    (8, f"Event recall {recall}", None, "Against one minute marked by hand. Provisional: the marks were made by the "
     "assistant that built the system, and an independent re-mark is pending."),
    (10, "A general AI model isn't enough", "m4_frames/E085_outlined.jpg", "A local vision-language model calls the pipe "
     "'not visible' in 17 of 18 collar drafts, even with the pipe outlined. So drafts come from measurements."),
    (12, "A person approves every item", "m5_review_screen.jpg", "Accept, edit or reject. Every decision is logged with the "
     "model versions. Exports: the event listing and a draft report."),
    (10, "Runs offline on a laptop", None, f"About {round(m3['processing']['minutes_per_hour_of_footage'])} minutes per hour "
     "of footage, $0 per survey, and the data never leaves the machine. Not for integrity decisions: a qualified person "
     "decides, and a licensed engineer signs any engineering conclusion."),
]
assert sum(s for s, *_ in SLIDES) == 120


def font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1 if bold else 0)
    except OSError:  # not on a Mac
        return ImageFont.load_default(size=size)


def wrap(draw, text, f, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


frames = root / "runs/video"
frames.mkdir(parents=True, exist_ok=True)
listing = []
for k, (secs, title, image, caption) in enumerate(SLIDES, 1):
    slide = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(slide)
    d.rectangle([0, 0, 10, H], fill=BLUE)
    if image:
        d.text((48, 28), title, font=font(40, True), fill=INK)
        lines = wrap(d, caption, font(25), W - 96)
        box_top, box_bottom = 92, H - 40 - 34 * len(lines) - 16
        img = Image.open(figs / image).convert("RGB")
        img.thumbnail((W - 96, box_bottom - box_top))
        slide.paste(img, ((W - img.width) // 2, box_top + (box_bottom - box_top - img.height) // 2))
        for i, line in enumerate(lines):
            d.text((48, box_bottom + 16 + 34 * i), line, font=font(25), fill=INK)
    else:
        d.text((80, 250), title, font=font(64, True), fill=INK)
        for i, line in enumerate(wrap(d, caption, font(30), W - 160)):
            d.text((80, 350 + 42 * i), line, font=font(30), fill=MUTED if k > 1 else INK)
    d.text((48, H - 32), "Logbook Subsea · demonstration on public data", font=font(16), fill=MUTED)
    slide.save(frames / f"slide{k:02d}.png")
    listing += [f"file 'slide{k:02d}.png'", f"duration {secs}"]
(frames / "list.txt").write_text("\n".join(listing) + "\n")
out = root / "docs/video/logbook_subsea_2min.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(frames / "list.txt"),
                "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "22", "-movflags", "+faststart", str(out)], check=True)

lines = ["# Two-minute video: narration script", "",
         "The video ([`video/logbook_subsea_2min.mp4`](video/logbook_subsea_2min.mp4)) is silent with captions. To record a",
         "voiceover, read each line while its slide shows. Made by `scripts/make_video.py` from the committed results.", "",
         "| Time | On screen | Say |", "|---|---|---|"]
t = 0
for secs, title, image, caption in SLIDES:
    lines.append(f"| {t // 60}:{t % 60:02d} | {title}{f' ({image})' if image else ''} | {caption} |")
    t += secs
(root / "docs/VIDEO_SCRIPT.md").write_text("\n".join(lines) + "\n")
print(out, f"{out.stat().st_size / 1e6:.1f} MB")

# Marking events by hand (M3, NFR-03)

> **Current marks are provisional.** Claude marked this minute on 2026-11-23 at Mostafa's request, after it had seen
> the system's output, so the current score is not independent. To replace them, delete the rows (keep the header)
> and mark the minute as below.

The event listing is scored against events a person marks by hand on one minute of camera video, 360–420 s. Neither
model trained on that minute. Marking it takes about half an hour, ideally by someone who hasn't worked on the
models.

## Before you start
- **Don't look at the system's output first:** `docs/results/m3_events.csv`, `docs/results/m3_events.json` and
  `docs/figures/m3_route.png`. Otherwise your marks lean towards what the system found.
- **Make the clip** (once): `.venv/bin/python -I scripts/make_labelling_clip.py`. It writes
  `data/labelling/camera_360-420s.mp4`, the camera at 15 frames a second with the survey time (for example
  `t = 372.4 s`) in the top-left corner.
- **Open it in QuickTime Player.** Space pauses, and the arrow keys step one frame at a time.

## What to mark
Add one row per event to [`labels/hand_events_360-420s.csv`](../labels/hand_events_360-420s.csv), using the time
shown on the frame.

| `event` | When |
|---|---|
| `collar` | Each collar or joint (a wider band around the pipe), at the moment it starts to leave the bottom of the frame |
| `hidden_start` | You can no longer see any of the pipe: buried, covered, or the camera points away. Ignore anything shorter than about a second |
| `hidden_end` | The pipe is back in view. Every `hidden_start` needs a `hidden_end` after it |

- If the pipe is already hidden when the clip starts, use `360.0` for `hidden_start`. If it is still hidden at the
  end, use `420.0` for `hidden_end`.
- Use `note` for anything you're unsure about.

Example rows (made-up times):
```
time_s,event,note
365.2,collar,
371.8,hidden_start,pipe under sand
373.0,hidden_end,
```

## How it is scored
These rules were fixed before any marks existed.
- **Collar:** found if the listing has a collar within 2 s of the mark. Each listed collar can match only one mark.
- **Hidden stretch:** found if the listing's camera "pipe not visible" stretches cover at least half of it.
- **Event recall** (NFR-03, target 90%): found marks divided by all marks. The score also reports listed collars and
  stretches that match no mark.

Run `.venv/bin/python -I scripts/build_events.py`. The score lands in `docs/results/m3_events.json` under `score`.
Commit the filled-in label file so the score can be reproduced.

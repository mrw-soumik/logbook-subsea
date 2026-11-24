"""Milestone M5: the review screen.

A reviewer goes through the M3 event listing, sees each event's camera frame and a draft written from what the system
measured, and accepts, edits or rejects it. Every decision is appended to a log with the suggestion and the model
version it answered. Exports give the reviewed event listing (CSV) and a draft inspection report (Markdown).

Run: .venv/bin/uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000, then open http://127.0.0.1:8000
Decisions live in runs/review.sqlite, outside git.
"""

import hashlib
import io
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from PIL import Image
from pydantic import BaseModel, StringConstraints

ROOT = Path(__file__).resolve().parent.parent
BAND = {"HF": "high-frequency", "LF": "low-frequency"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, event TEXT, source TEXT, start_s REAL, end_s REAL, start_utc TEXT,
    along_track_start_m REAL, along_track_end_m REAL, east_m REAL, north_m REAL, detail TEXT,
    draft TEXT, model_note TEXT, model_version TEXT);
CREATE TABLE IF NOT EXISTS decisions (
    n INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL REFERENCES events(id),
    decision TEXT NOT NULL CHECK (decision IN ('accept', 'edit', 'reject')), final_text TEXT, reason TEXT,
    suggestion TEXT NOT NULL, model_version TEXT NOT NULL, reviewer TEXT NOT NULL, at_utc TEXT NOT NULL);
"""
LATEST = """
SELECT e.*, d.decision, d.final_text, d.reason, d.reviewer, d.at_utc FROM events e
LEFT JOIN decisions d ON d.n = (SELECT MAX(n) FROM decisions WHERE event_id = e.id)
ORDER BY e.start_s, e.id
"""


class Decision(BaseModel):
    decision: Literal["accept", "edit", "reject"]
    reviewer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]
    text: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] = ""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    """Short SHA-256 of a file, so every logged suggestion traces back to the exact listing."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def drafts(ev):
    """A plain-language draft for each event, written only from what the system measured."""
    hidden = {s: list(zip(g.start_s, g.end_s)) for s, g in ev[ev.event == "pipe not visible"].groupby("source")}

    def lost(source, a, b):  # share of [a, b] in which `source` also lost the pipe
        return sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in hidden.get(source, [])) / max(b - a, 1e-9)

    out = []
    for e in ev.itertuples():
        where = f"{e.along_track_start_m:.1f} to {e.along_track_end_m:.1f} m along the track"
        if e.event == "collar or joint":
            out.append(f"Collar or joint at {e.start_s:.1f} s, {e.along_track_start_m:.1f} m along the track: on camera the "
                       f"pipe looks {float(e.detail.split()[0]):.2f} times its usual width.")
        elif e.source == "camera":
            sonar = " ".join(f"The {BAND[b]} sonar {'also lost it' if f >= 0.5 else 'still saw the pipe' if f == 0 else 'lost it for part of that time'}."
                             for b, f in ((b, lost(f"sonar {b}", e.start_s, e.end_s)) for b in BAND))
            out.append(f"Pipe not visible on camera from {e.start_s:.1f} to {e.end_s:.1f} s ({where}). {sonar}")
        else:
            cam = lost("camera", e.start_s, e.end_s) * (e.end_s - e.start_s)
            out.append(f"Pipe not seen by the {BAND[e.source.split()[1]]} sonar from {e.start_s:.1f} to {e.end_s:.1f} s ({where}). "
                       f"The camera lost it for {cam:.0f} s of that time.")
    return out


def safe_cell(v):
    """Stop spreadsheet apps reading reviewer text as a formula."""
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


def create_app(db_path=ROOT / "runs/review.sqlite", events_csv=ROOT / "docs/results/m3_events.csv",
               notes_csv=ROOT / "docs/results/m4_descriptions.csv", data_dir=ROOT / "data/SubPipeMini/DATA",
               summary_json=ROOT / "docs/results/m3_events.json"):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def db():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    ev = pd.read_csv(events_csv)
    notes = pd.read_csv(notes_csv) if Path(notes_csv).exists() else pd.DataFrame(columns=["id", "version", "description"])
    notes = notes[notes.version == "outlined"]
    note = dict(zip(notes.id, notes.description))
    listing = sha(events_csv)
    models = json.loads(Path(summary_json).read_text()).get("models", {}) if Path(summary_json).exists() else {}
    version = f"events {listing}, models " + (", ".join(f"{k} {v}" for k, v in models.items()) or "unknown")
    with closing(db()) as con, con:
        con.executescript(SCHEMA)
        stored = con.execute("SELECT value FROM meta WHERE key = 'listing'").fetchone()
        if stored and stored[0] != listing:  # never mix decisions about one listing with another listing's ids
            raise RuntimeError(f"{events_csv} changed since this review began. Move {db_path} aside to start a new review.")
        if not stored:
            con.execute("INSERT INTO meta VALUES ('listing', ?)", (listing,))
            con.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                (e.id, e.event, e.source, e.start_s, e.end_s, e.start_utc, e.along_track_start_m, e.along_track_end_m,
                 e.east_m, e.north_m, e.detail, d, note.get(e.id), version) for e, d in zip(ev.itertuples(), drafts(ev))])

    frames, frame_t = [], np.array([])
    if (Path(data_dir) / "Cam0_images").exists():
        t0 = float(pd.read_csv(Path(data_dir) / "EstimatedState.csv", nrows=1).rename(columns=str.strip)["timestamp"].iloc[0])
        frames = sorted((Path(data_dir) / "Cam0_images").glob("*.jpg"), key=lambda p: float(p.stem))
        frame_t = np.array([float(p.stem) for p in frames]) - t0

    def latest():
        with closing(db()) as con:
            return [dict(r) for r in con.execute(LATEST)]

    app = FastAPI(title="Logbook Subsea review")

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/api/events")
    def events():
        return latest()

    @app.get("/api/events/{event_id}/frame.jpg")
    def frame(event_id: str):
        with closing(db()) as con:
            e = con.execute("SELECT start_s, end_s FROM events WHERE id = ?", (event_id,)).fetchone()
        if not e or not frames:
            raise HTTPException(404, "no camera frame for this event")
        img = Image.open(frames[int(np.abs(frame_t - (e["start_s"] + e["end_s"]) / 2).argmin())])
        img.draft("RGB", (1014, 570))
        buf = io.BytesIO()
        img.convert("RGB").resize((960, 540)).save(buf, "JPEG", quality=85)
        return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/events/{event_id}/decision")
    def decide(event_id: str, d: Decision):
        with closing(db()) as con, con:
            e = con.execute("SELECT draft, model_version FROM events WHERE id = ?", (event_id,)).fetchone()
            if not e:
                raise HTTPException(404, "no such event")
            if d.decision == "edit" and d.text in ("", e["draft"]):
                raise HTTPException(422, "an edit needs changed text; use accept to keep the draft")
            final = {"accept": e["draft"], "edit": d.text, "reject": None}[d.decision]
            con.execute("INSERT INTO decisions (event_id, decision, final_text, reason, suggestion, model_version, reviewer, at_utc) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (event_id, d.decision, final, d.text or None if d.decision == "reject" else None, e["draft"],
                         e["model_version"], d.reviewer, now()))
        return {"saved": True}

    @app.get("/api/log")
    def log():
        with closing(db()) as con:
            return [dict(r) for r in con.execute("SELECT * FROM decisions ORDER BY n")]

    @app.get("/export/events.csv")
    def export_events():
        cols = ["id", "event", "source", "start_s", "end_s", "start_utc", "along_track_start_m", "along_track_end_m",
                "east_m", "north_m", "draft", "decision", "final_text", "reason", "reviewer", "at_utc", "model_version"]
        table = pd.DataFrame(latest(), columns=cols).map(safe_cell)
        return Response(table.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="reviewed_events.csv"'})

    @app.get("/export/report.md")
    def report():
        rows = latest()
        reviewed = [r for r in rows if r["decision"]]
        kept = [r for r in reviewed if r["decision"] != "reject"]
        n = {k: sum(r["decision"] == k for r in reviewed) for k in ("accept", "edit", "reject")}
        changed = f"{(n['edit'] + n['reject']) / len(reviewed):.0%}" if reviewed else "no reviews yet"
        lines = [
            "# Draft inspection report: SubPipe chunk 2",
            "",
            f"Generated {now()} from the reviewed event listing ({version}).",
            "",
            "**Draft for review.** This is a demonstration on public data, not an integrity assessment. A qualified person",
            "must check it, and any engineering conclusion needs a licensed engineer.",
            "",
            "## Review status",
            "",
            f"- Events the system listed: {len(rows)}",
            f"- Reviewed: {len(reviewed)} (accepted {n['accept']}, edited {n['edit']}, rejected {n['reject']})",
            f"- Not yet reviewed: {len(rows) - len(reviewed)}",
            f"- Share of reviewed drafts edited or rejected (NFR-05): {changed}",
        ]
        for title, kind in (("Pipe not visible", "pipe not visible"), ("Collars and joints", "collar or joint")):
            group = [r for r in kept if r["event"] == kind]
            lines += ["", f"## {title} ({len(group)} kept after review)", ""]
            if not group:
                lines.append("None reviewed and kept yet.")
                continue
            lines += ["| Event | Source | Time (s) | Along track (m) | Finding | Reviewer |", "|---|---|---|---|---|---|"]
            lines += [f"| {r['id']} | {r['source']} | {r['start_s']:.1f} to {r['end_s']:.1f} | {r['along_track_start_m']:.1f} to "
                      f"{r['along_track_end_m']:.1f} | {r['final_text'].replace('|', '/').replace(chr(10), ' ')} | {r['reviewer']} ({r['decision']}) |"
                      for r in group]
        lines += ["", "Data: SubPipe, property of Oceanscan-MST, CC BY 4.0. Positions are on the survey's local grid."]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/markdown")

    return app

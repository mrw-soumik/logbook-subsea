"""M5: the review screen logs every decision, keeps the log append-only, and exports what reviewers kept."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

EVENTS = pd.DataFrame([
    ["E001", "collar or joint", "camera", 10.0, 10.0, "2023-09-01 13:12:24.2", 8.6, 8.6, -555.0, 133.0, "1.40 x usual width"],
    ["E002", "pipe not visible", "camera", 20.0, 24.0, "2023-09-01 13:12:34.2", 17.2, 20.6, -562.0, 132.0, "4.0 s"],
    ["E003", "pipe not visible", "sonar HF", 21.0, 30.0, "2023-09-01 13:12:35.2", 18.1, 25.8, -563.0, 132.0, "9 s"],
], columns=["id", "event", "source", "start_s", "end_s", "start_utc", "along_track_start_m", "along_track_end_m", "east_m", "north_m", "detail"])


@pytest.fixture
def paths(tmp_path):
    EVENTS.to_csv(tmp_path / "events.csv", index=False)
    pd.DataFrame({"id": ["E001"], "version": ["outlined"], "description": ["The pipe is not clearly visible."]}).to_csv(tmp_path / "notes.csv", index=False)
    return {"db_path": tmp_path / "review.sqlite", "events_csv": tmp_path / "events.csv", "notes_csv": tmp_path / "notes.csv",
            "data_dir": tmp_path / "no-data", "summary_json": tmp_path / "no-summary.json"}


def test_review_flow(paths):
    c = TestClient(create_app(**paths))
    events = c.get("/api/events").json()
    assert [e["id"] for e in events] == ["E001", "E002", "E003"] and not any(e["decision"] for e in events)
    assert events[0]["draft"].startswith("Collar or joint at 10.0 s") and events[0]["model_note"] == "The pipe is not clearly visible."
    assert "high-frequency sonar also lost it" in events[1]["draft"]  # its stretch covers 3 of the 4 s, over half
    assert "low-frequency sonar still saw the pipe" in events[1]["draft"]

    assert c.post("/api/events/E001/decision", json={"decision": "accept", "reviewer": "Mostafa"}).status_code == 200
    assert c.post("/api/events/E002/decision", json={"decision": "edit", "reviewer": "Reviewer B", "text": "Pipe buried, 4 s."}).status_code == 200
    assert c.post("/api/events/E003/decision", json={"decision": "reject", "reviewer": "Reviewer B", "text": "vehicle over the pipe"}).status_code == 200
    assert c.post("/api/events/E001/decision", json={"decision": "accept", "reviewer": "Reviewer B"}).status_code == 200

    log = c.get("/api/log").json()  # every decision kept, each with the suggestion and model version it answered
    assert len(log) == 4 and all(r["suggestion"] and r["model_version"] for r in log)
    latest = {e["id"]: e for e in c.get("/api/events").json()}
    assert latest["E001"]["reviewer"] == "Reviewer B" and latest["E002"]["final_text"] == "Pipe buried, 4 s."

    report = c.get("/export/report.md").text
    assert "Reviewed: 3 (accepted 1, edited 1, rejected 1)" in report and "(NFR-05): 67%" in report
    assert "Pipe buried, 4 s." in report and "E003" not in report and "licensed engineer" in report
    assert "reviewed_events.csv" in c.get("/export/events.csv").headers["content-disposition"]


def test_bad_input_is_refused(paths):
    c = TestClient(create_app(**paths))
    assert c.post("/api/events/E999/decision", json={"decision": "accept", "reviewer": "a"}).status_code == 404
    assert c.post("/api/events/E001/decision", json={"decision": "approve", "reviewer": "a"}).status_code == 422
    assert c.post("/api/events/E001/decision", json={"decision": "accept", "reviewer": "   "}).status_code == 422
    assert c.post("/api/events/E001/decision", json={"decision": "edit", "reviewer": "a", "text": ""}).status_code == 422
    assert c.get("/api/events/E001/frame.jpg").status_code == 404  # no survey data in this test
    assert c.get("/api/log").json() == []


def test_changed_listing_never_mixes_with_old_decisions(paths):
    c = TestClient(create_app(**paths))
    c.post("/api/events/E001/decision", json={"decision": "accept", "reviewer": "a"})
    assert len(TestClient(create_app(**paths)).get("/api/log").json()) == 1  # a restart keeps decisions
    EVENTS.assign(start_s=EVENTS.start_s + 1).to_csv(paths["events_csv"], index=False)
    with pytest.raises(RuntimeError, match="changed since this review began"):
        create_app(**paths)


def test_export_neutralises_formulas(paths):
    c = TestClient(create_app(**paths))
    c.post("/api/events/E002/decision", json={"decision": "edit", "reviewer": "a", "text": "=HYPERLINK(\"x\")"})
    assert "'=HYPERLINK" in c.get("/export/events.csv").text

"""M6 quality gate: checks the committed results against the requirements and exits 1 if a guarded one falls short.

Usage: python scripts/quality_gate.py   (standard library only, so CI runs it without the models or the survey data)
Guarded checks are requirements met today, so the gate catches regressions. Targets not met yet are listed as
warnings until they are met; then they move to guarded. NFR-08 is checked by the tests in tests/.
"""

import json
import sys
from pathlib import Path

results = Path(__file__).resolve().parent.parent / "docs/results"
m1, m2, m3, m4, m6 = (json.loads((results / f).read_text()) for f in
                      ("m1_segmentation.json", "m2_sonar_detection.json", "m3_events.json", "m4_summary.json", "m6_held_out_chunks.json"))


def balanced(s):
    """Average of the share of pipe windows found and the share of no-pipe windows left empty."""
    (found, pipe), (empty, none) = s["pipe_windows_found"], s["no_pipe_windows_left_empty"]
    return (found / pipe + empty / none) / 2


assert balanced({"pipe_windows_found": [3, 4], "no_pipe_windows_left_empty": [1, 2]}) == 0.625
score, ratings = m3["score"], m4["ratings"]
checks = [  # (guarded, requirement, measured, passes)
    (True, "NFR-01 camera: mean IoU on held-out minutes >= 0.80", m1["test"]["mean_iou"], m1["test"]["mean_iou"] >= 0.80),
    *[(True, f"NFR-02 sonar {b}: mAP@0.5-0.95 above a box that never moves", f"{m2['test'][b]['detector']['ap50_95']} vs "
       f"{m2['test'][b]['fixed_box_baseline']['ap50_95']}", m2["test"][b]["detector"]["ap50_95"] > m2["test"][b]["fixed_box_baseline"]["ap50_95"])
      for b in ("HF", "LF")],
    *[(False, f"NFR-02 sonar {b}: balanced pipe or no-pipe accuracy >= 0.90", round(balanced(m2["test"][b]["detector"]), 3),
       balanced(m2["test"][b]["detector"]) >= 0.90) for b in ("HF", "LF")],
    *[(True, f"NFR-02 sonar {b}, four unseen chunks pooled: balanced accuracy above a box that never moves (0.50)",
       m6["pooled"][f"sonar_{b}"]["balanced_accuracy"], m6["pooled"][f"sonar_{b}"]["balanced_accuracy"] > 0.50) for b in ("HF", "LF")],
    *[(False, f"NFR-02 sonar {b}, four unseen chunks pooled: balanced accuracy >= 0.90",
       m6["pooled"][f"sonar_{b}"]["balanced_accuracy"], m6["pooled"][f"sonar_{b}"]["balanced_accuracy"] >= 0.90) for b in ("HF", "LF")],
    (True, "NFR-03 event recall >= 0.90 on the hand-marked minute (provisional marks)", score.get("event_recall", score["status"]),
     score.get("event_recall", 0) >= 0.90),
    (False, "NFR-04 drafts rated correct by both reviewers >= 0.80",
     "; ".join(f"{version}: " + ", ".join(f"{who} {r['share_correct']}" for who, r in v.get("by_reviewer", {}).items()) + f" ({v['status']})"
               for version, v in ratings.items() if isinstance(v, dict)) or ratings.get("status"),
     any(v.get("correct_by_both", 0) >= 0.80 for v in ratings.values() if isinstance(v, dict))),
    (False, "NFR-01 camera on unseen chunks, provisional reference outlines: mean IoU >= 0.80",
     m6["camera_reference_iou"]["pooled"]["mean_iou"], m6["camera_reference_iou"]["pooled"]["mean_iou"] >= 0.80),
    (False, "NFR-01 camera on unseen chunks, provisional frame check: false alarms <= 5%",
     "visible found {0[0]}/{0[1]}, pipe-free frames marked {1[0]}/{1[1]}".format(*(m6["camera_frame_check"]["pooled"][k] for k in ("visible_frames_found", "not_visible_frames_marked"))),
     m6["camera_frame_check"]["pooled"]["not_visible_frames_marked"][0] <= 0.05 * m6["camera_frame_check"]["pooled"]["not_visible_frames_marked"][1]),
    (True, "NFR-06 minutes of processing per hour of footage < 60", m3["processing"]["minutes_per_hour_of_footage"],
     m3["processing"]["minutes_per_hour_of_footage"] < 60),
    (True, "NFR-07 cost per survey with the local model is $0", m4["cost"], m4["cost"].startswith("$0")),
]
# Follow-up runs, checked when their results exist. They are met today, so they are guarded like the rest.
optional = {f: json.loads((results / f).read_text()) for f in
            ("m6_sonar_multichunk_test.json", "m6_sonar_multichunk_scratch_test.json", "m1_segmentation_scratch.json",
             "m2_sonar_detection_scratch.json") if (results / f).exists()}
if "m6_sonar_multichunk_test.json" in optional:  # trained on Chunks 0, 1 and 2; Chunks 3 and 4 unseen
    mc = optional["m6_sonar_multichunk_test.json"]["pooled"]
    for b in ("HF", "LF"):
        old = [m6["chunks"][c][f"sonar_{b}"]["detector"] for c in ("Chunk3", "Chunk4")]  # the Chunk-2-only model, same windows
        base = balanced({k: [sum(x[k][i] for x in old) for i in (0, 1)] for k in ("pipe_windows_found", "no_pipe_windows_left_empty")})
        checks.append((True, f"NFR-02 sonar {b}, trained on three surveys, unseen Chunks 3 and 4: balanced accuracy >= 0.90",
                       f"{mc[f'sonar_{b}']['balanced_accuracy']} (Chunk-2-only model on the same windows: {base:.3f})",
                       mc[f"sonar_{b}"]["balanced_accuracy"] >= 0.90))
if "m6_sonar_multichunk_scratch_test.json" in optional:  # the same, trained without COCO or ImageNet starting weights
    ms = optional["m6_sonar_multichunk_scratch_test.json"]["pooled"]
    for b in ("HF", "LF"):
        checks.append((True, f"Licence route: sonar {b} trained on three surveys from scratch, unseen Chunks 3 and 4: balanced accuracy >= 0.90",
                       ms[f"sonar_{b}"]["balanced_accuracy"], ms[f"sonar_{b}"]["balanced_accuracy"] >= 0.90))
if "m1_segmentation_scratch.json" in optional:
    v = optional["m1_segmentation_scratch.json"]["test"]["mean_iou"]
    checks.append((True, "Licence route: camera trained without ImageNet weights, mean IoU on held-out minutes >= 0.80",
                   f"{v} (with them: {m1['test']['mean_iou']})", v >= 0.80))
if "m2_sonar_detection_scratch.json" in optional:
    s = optional["m2_sonar_detection_scratch.json"]["test"]
    for b in ("HF", "LF"):
        checks.append((True, f"Licence route: sonar {b} trained without COCO/ImageNet weights, mAP@0.5-0.95 above the fixed box",
                       f"{s[b]['detector']['ap50_95']} vs {s[b]['fixed_box_baseline']['ap50_95']} (with them: {m2['test'][b]['detector']['ap50_95']})",
                       s[b]["detector"]["ap50_95"] > s[b]["fixed_box_baseline"]["ap50_95"]))
failed = 0
for guarded, name, measured, ok in checks:
    label = "PASS" if ok else ("FAIL" if guarded else "WARN")
    failed += label == "FAIL"
    print(f"{label}  {name}: {measured}")
print(f"\n{failed} guarded check(s) failed" if failed else "\nAll guarded checks pass")
sys.exit(1 if failed else 0)

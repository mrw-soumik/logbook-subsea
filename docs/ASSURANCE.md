# Assurance note: Logbook Subsea

For anyone deciding how far to trust this system's output: an inspection lead, a client's technical authority or a
partner. It is set out as claims backed by evidence, the approach of DNV's recommended practice for assuring
AI-enabled systems, [DNV-RP-0671](https://www.dnv.com/digital-trust/recommended-practices/assurance-of-ai-enabled-systems-dnv-rp-0671/)
(2023). It is a self-assessment of a demonstration on public data. DNV has not reviewed or endorsed it.

Status on 2026-11-23. Every figure links to the result file or test it comes from.

## 1. The system and its intended use
- **What it does:**
  1. Finds the pipe in camera frames (M1) and in side-scan sonar (M2).
  2. Turns those results into events along the route: collars or joints, and stretches where a sensor can't see the pipe (M3).
  3. Writes a draft for each event from what was measured (M5). A local vision-language model adds a note marked unverified (M4).
  4. A person accepts, edits or rejects every item. Exports give the reviewed listing and a draft report.
- **Intended use:** to help a reviewer log events from survey footage faster.
- **Not for:** integrity or fitness-for-service decisions, or any output used without review. A qualified person
  decides, and any engineering conclusion needs a licensed engineer.
- **Where it has been tested:**
  - one 9-minute survey of one outfall pipeline (SubPipe chunk 2), 16–17 m deep, about 3 m above the seabed, at 0.86 m/s;
  - one camera, in blue and green water, plus side-scan sonar at two frequencies.

  Anything outside that is untested.

## 2. Claims and evidence
| # | Claim | Evidence | Status | Limits |
|---|---|---|---|---|
| C1 | The camera model finds the pipe on footage it hasn't seen | Mean IoU 0.841 (pipe alone 0.737) on two held-out minutes ([EVALUATION.md, M1](EVALUATION.md#m1-finding-the-pipe-in-camera-frames)) | Supported for this survey; on other surveys, provisional | Buried pipe and cloudy water fail; 23 of 163 test frames score under 0.5. No camera labels exist outside chunk 2. On unseen chunks, provisionally: mean IoU 0.795 (pipe 0.672) on 19 frames against reference outlines made with Segment Anything and checked by Claude, and all 19 visible pipes found but 43 of 383 pipe-free frames marked (dark image edges, hazy patches) ([EVALUATION.md, M6](EVALUATION.md#m6-surveys-the-models-never-saw)) |
| C2 | Test scores are not inflated by leakage | Splits by time with gaps: 5 s for camera, 25 s for sonar after finding that sonar images are rolling windows sharing 96% of their pixels. The scripts assert the gaps. Test sets were scored once, and event rules were set on training stretches only | Supported | The camera is tested on held-out minutes of one survey only; the sonar also on four unseen chunks |
| C3 | The sonar detector tells you more than the survey geometry does | Better than a box that never moves on mAP@0.5–0.95 in both bands. Balanced pipe or no-pipe accuracy: 0.80 against 0.50 in high frequency ([EVALUATION.md, M2](EVALUATION.md#m2-finding-the-pipe-in-sonar)) | Supported; NFR-02 met on unseen surveys | Trained on chunk 2 only: 0.85 and 0.81 on four unseen surveys, against 0.50, while the fixed box drops to mAP 0 to 0.07. Trained on chunks 0, 1 and 2: 0.95 and 0.96 on unseen chunks 3 and 4 ([EVALUATION.md, M6](EVALUATION.md#m6-surveys-the-models-never-saw)) | Only two surveys held out, mostly chunk 4. Low-frequency boxes on chunk 3 stay loose. One run, one seed |
| C4 | The event listing finds what a person would log | Event recall 0.917 (11 of 12) on 360–420 s ([EVALUATION.md, M3](EVALUATION.md#m3-events-along-the-route)) | Provisional | Marked by the assistant that built the rules, after seeing the output. One minute, 12 events |
| C5 | Drafts say only what the system measured | Drafts are built from the listing's fields alone ([app/main.py](../app/main.py)); tests check the wording ([tests](../tests/test_review_app.py)) | Supported by construction | A provisional demonstration review of 14 events edited 3 (21%), none rejected ([EVALUATION.md, M5](EVALUATION.md#m5-a-demonstration-review-session)); real reviewers haven't used it yet (NFR-05) |
| C6 | Nothing relies on the vision-language model | It calls the pipe not (clearly) visible in 17 of 18 collar drafts ([EVALUATION.md, M4](EVALUATION.md#m4-draft-descriptions-from-a-local-model)), so it appears only as an unverified note | Supported | One provisional reviewer rates 45% to 50% of its drafts correct; two human reviewers haven't rated them yet (NFR-04) |
| C7 | Every AI suggestion and human decision can be traced | Append-only decision log with the suggestion, listing hash and model fingerprints. A changed listing can't mix with old decisions. Tested | Supported | Local SQLite with no login; the log is not tamper-proof |
| C8 | Data stays on the machine, at no cost per survey | Every model runs locally, with no API calls (NFR-07) | Supported | One-time 5.8 GB model download from Hugging Face |
| C9 | Processing keeps up with survey work | 9.6 minutes per hour of footage for the event pipeline on an Apple M3 Pro ([m3_events.json](results/m3_events.json)) | Supported | Excludes training; descriptions add about 5 s per event |
| C10 | Results can be reproduced | Each number comes from a committed script and is saved in [results/](results/); versions are pinned. M1 reruns are identical | Partly | M2 training on the Mac's graphics chip isn't deterministic (mAP@0.5 0.835 against 0.839) |
| C11 | The models needn't depend on starting weights with restrictive terms | Trained from random weights: camera mean IoU 0.818 (0.841 with ImageNet), sonar mAP@0.5–0.95 0.525 and 0.456 (0.520 and 0.514 with COCO) ([EVALUATION.md, M6](EVALUATION.md#without-imagenet-or-coco-starting-weights)) | Tested; adopted for sonar on 2026-11-24 (by Claude, on Mostafa's instruction) | Trained from scratch, the three-survey sonar model still meets NFR-02 on unseen surveys (0.903 and 0.932, against 0.950 and 0.956 with COCO), with a thin high-frequency margin. One run each |

## 3. Known failure modes
| Failure | Seen in | What catches it |
|---|---|---|
| Pipe buried, covered or in cloudy water: the camera misses it | M1 worst frames, around 390–418 s | The reviewer sees the frame; the draft says what each sonar saw |
| Vehicle over the pipe: the sonar can't see it, but the camera can | M2 and M3, 386–484 s | The draft for a sonar stretch says how long the camera also lost the pipe |
| Missing or one-sided labels | M2: one window at 95 s, and right-side detections at 464–480 s | Scores flag these as possible label problems rather than model errors; not yet re-labelled |
| Collar timing off by up to 1.6 s, or a collar skipped | M3 provisional score; gaps of about 11 s | A 2 s matching window in scoring; the reviewer sees every collar |
| The language model hedges and says it can't see the pipe | M4 | It isn't used for drafts |
| The pipe on the other side of the vehicle, or far out | Unseen chunks 0 and 3 | Sonar still tells pipe from no pipe on the other side, but boxes are loose; far out it is weak |
| Camera false alarms: hazy patches, and dark image edges | Unseen chunks 0, 3 and 4: 43 of 383 pipe-free frames | The reviewer sees the frame; a stricter rule needs camera labels from more surveys |
| A different pipeline, camera, water or altitude | Four more chunks of the same pipeline only | Not covered: re-validate before use |

## 4. Human oversight
- **Every item needs a decision:** accept, edit or reject. Nothing reaches an export without one, and the report
  counts unreviewed events.
- **Reviewers see the evidence:** the camera frame and the measured facts, next to the draft.
- **The report says what it is:** a draft, not an integrity assessment.

## 5. Keeping the assurance current
DNV-RP-0671 treats assurance as continuous, because models and data change. Here:
- **On every push,** GitHub runs the [quality gate](../scripts/quality_gate.py), the tests and a container build.
  - **Guarded:** requirements met today, which fail the build if they slip.
  - **Warnings:** unmet targets, until they're met.
- **Traceability:** the listing hash and model fingerprints go with every logged suggestion.
- **Re-run every evaluation when** a model is retrained, an event rule changes or a new survey is added.
- **Watch the edit-or-reject share (NFR-05):** a rise is an early sign the system no longer fits the footage.

## 6. Open items before any real use
Claude stood in, provisionally and labelled as such, for the human steps in items 1 to 3. On 2026-11-24 Mostafa closed
them by accepting those provisional results for now. They are still not independent checks, so before any real use
people should redo them.
1. A person re-marks 360–420 s, so NFR-03 is independent ([LABELLING.md](LABELLING.md)).
2. Two reviewers rate the M4 drafts ([M4_REVIEW.md](M4_REVIEW.md)), and real review sessions measure NFR-05.
3. People label camera frames on at least one other chunk, with pipe outlines, so NFR-01 can be scored beyond chunk 2.
   Then revisit the 2% "pipe found" rule, which raises false alarms in hazy water.
4. Settle the doubtful sonar labels at the end of chunk 2. A higher-resolution look supports the vehicle passing over
   the pipe, but no label was changed.
5. Add access control and a tamper-evident log before more than one person uses the review screen.
6. Settle the terms of the ImageNet- and COCO-trained starting weights before any paid use. The from-scratch models,
   including the three-survey sonar model, meet NFR-01 and NFR-02 at a small cost, so delivering those avoids the
   question ([LICENCES.md](LICENCES.md)).

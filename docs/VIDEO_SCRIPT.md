# Two-minute video: narration script

The video ([`video/logbook_subsea_2min.mp4`](video/logbook_subsea_2min.mp4)) is silent with captions. To record a
voiceover, read each line while its slide shows. Made by `scripts/make_video.py` from the committed results.

| Time | On screen | Say |
|---|---|---|
| 0:00 | Logbook Subsea | From subsea survey footage to a reviewed event listing, with a person approving every item. A demonstration on public data. |
| 0:08 | The problem (labelled_frames.jpg) | Inspection teams watch hours of survey video and log every finding by hand: the time, the position and what they saw. |
| 0:18 | The data (samples.png) | SubPipe, a public survey (Oceanscan-MST, CC BY 4.0): 9 minutes along 465 m of pipe, with camera, side-scan sonar and navigation. |
| 0:28 | Honest testing (track.png) | Models are scored on minutes they never saw, with gaps around them. Neighbouring frames, and sonar images that share 96% of their pixels, would otherwise leak the answers. |
| 0:38 | Finding the pipe on camera (m1_predictions.jpg) | Mean IoU 0.841 on held-out minutes (the pipe alone: 0.737). Yellow: found. Magenta: missed. Cyan: false alarm. |
| 0:48 | Where it fails (m1_worst_frames.jpg) | Buried pipe and cloudy water. The failures are published next to the average, because they are the cases an inspector cares about. |
| 0:58 | Sonar (m2_sonar_test.jpg) | The vehicle follows the pipe, so a box that never moves already scores 0.66 to 0.75 mAP. The real test is knowing when the pipe isn't there. |
| 1:08 | Events along the route (m3_route.png) | 86 collars about 5 m apart, and the stretches where each sensor loses the pipe. Near the end the sonar loses it while the camera still sees it. |
| 1:20 | Event recall 0.917 | Against one minute marked by hand. Provisional: the marks were made by the assistant that built the system, and an independent re-mark is pending. |
| 1:28 | A general AI model isn't enough (m4_frames/E085_outlined.jpg) | A local vision-language model calls the pipe 'not visible' in 17 of 18 collar drafts, even with the pipe outlined. So drafts come from measurements. |
| 1:38 | A person approves every item (m5_review_screen.jpg) | Accept, edit or reject. Every decision is logged with the model versions. Exports: the event listing and a draft report. |
| 1:50 | Runs offline on a laptop | About 10 minutes per hour of footage, $0 per survey, and the data never leaves the machine. Not for integrity decisions: a qualified person decides, and a licensed engineer signs any engineering conclusion. |

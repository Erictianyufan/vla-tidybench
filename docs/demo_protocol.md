# Final demonstration protocol

The video is an experimental artifact, not merely a promotional edit. It is
recorded only after the final code, checkpoint, action contract, prompt set and
test seeds are frozen.

## Required evidence before recording

1. Unit, protocol, simulator and final evaluation commands pass from a clean
   release candidate.
2. The final task has a machine-checkable success predicate for drawer-open,
   grasp, in-drawer placement and drawer-close stages.
3. The main policy and any RL specialist are selected on validation episodes;
   the locked final test episodes have not been used for tuning.
4. A run manifest records the Git commit, config hashes, checkpoint hash,
   dataset version, prompt, seed, simulator version and policy latency.

The frozen atomic predicate version is
`drawer_skill_v2_relative_stable`. It checks OPEN/CLOSE drawer travel, PICK
lift relative to the episode's initial bottle height plus a closed gripper,
PLACE containment plus gripper release, and bottle retention while closing.
Formal acceptance requires five consecutive successful 20-Hz states; a
single transient contact or threshold crossing is not a success.

## Storyboard (approximately 90–150 seconds)

| Time | Content | Evidence rule |
| --- | --- | --- |
| 0–8 s | Title, final language command and system versions | Text overlay only |
| 8–20 s | Two-camera observation and architecture | Use actual runtime frames |
| 20–75 s | Full OPEN → PICK → PLACE → CLOSE rollout | One uninterrupted rollout; show task clock |
| 75–100 s | Atomic-stage metrics and fixed-seed result table | Generated from saved evaluation JSON |
| 100–125 s | ID/OOD or perturbation examples | Label seed and condition |
| 125–145 s | Baseline/RL paired comparison, only if RL passes its gate | Same seed and initial condition |
| End | Repository name and exact reproduce command | No credentials or server address |

If the RL specialist does not pass its gate, the video states that result and
shows the best frozen imitation-learning system. A negative RL experiment must
not be edited into a positive claim.

## Capture requirements

- Capture the simulator viewport and save the policy camera streams used in the
  actual rollout.
- Use a stable 16:9 viewport, preferably 1920×1080 at 30 or 60 fps.
- Show the language command, active TaskGraph stage, episode seed and elapsed
  task time. Do not expose IP addresses, shell history, tokens or user names.
- No human input is allowed during an autonomous final-policy rollout. Any
  scripted-teacher or teleoperation footage must be labelled explicitly.
- Editing may trim idle time before reset and after termination, add narration,
  captions, zooms and metric overlays. It may not splice multiple attempts into
  one apparent success or omit interventions.
- Keep the raw recording and run manifest outside Git. Store a SHA-256 checksum
  with the release artifacts.

## Packaging

Given a raw recording:

```bash
cd /home/ubuntu/mycode/vla-tidybench
make package-demo INPUT=/home/ubuntu/data/vla-tidybench/videos/raw/final_run.mp4
```

The script creates:

- `artifacts/demo/vla-tidybench-demo.mp4` — H.264 1080p release artifact;
- `docs/media/demo-preview.gif` — short repository preview;
- `artifacts/demo/SHA256SUMS` — integrity record.

The full MP4 remains outside Git and is uploaded as a public GitHub Release
asset. Only the redacted preview GIF may be committed to Git, and only if it is
under the pre-publication size limit.

## Minimum final video set

- One uninterrupted successful end-to-end rollout.
- At least one representative OOD or perturbation rollout.
- One honest failure example or failure-taxonomy slide.
- A paired baseline/RL clip only when the RL release gate passes.

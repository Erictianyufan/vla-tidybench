# Portable π0.5 fine-tuning

The OpenPI training path is configured entirely through environment variables.
Large artifacts should live outside the source checkout.

## Required environment

```bash
export OPENPI_ROOT=/home/scuee_user06/tfy/openpi
export VLA_TIDYBENCH_DATA=/data/scuee_user06/vla-tidybench
export OPENPI_DATA_HOME=$VLA_TIDYBENCH_DATA/checkpoints
export HF_HOME=$VLA_TIDYBENCH_DATA/cache/huggingface
```

Optional overrides:

```bash
export PI05_CHECKPOINT_PARAMS=/absolute/path/to/pi05_droid/params
export VLA_TIDYBENCH_DRAWER_FOUR_SKILL_REPO_ID=owner/formal_dataset_name
```

The default π0.5-DROID checkpoint location is:

```text
$VLA_TIDYBENCH_DATA/checkpoints/openpi-assets/checkpoints/pi05_droid/params
```

Raw HDF5, LeRobot/Hugging Face cache, normalization assets and training
checkpoints are consequently stored on the selected data disk.

## Deterministic formal-data preparation

Copy the four nominal HDF5 files and four successful hard-recovery HDF5 files
to `$VLA_TIDYBENCH_DATA/raw`. The server-ready source manifests are:

- `configs/data/drawer_four_skill_formal.json`
- `configs/data/drawer_four_skill_hard_recovery.json`

Portable examples for changing filenames or repository ownership are also
checked in:

- `configs/data/drawer_four_skill_formal.example.json`
- `configs/data/drawer_four_skill_hard_recovery.example.json`

Failed terminal rollouts are evaluation evidence, not behavior-cloning targets.
The hard files must contain successful corrective/recovery trajectories. Create
an episode-level split and a stage-3 mixture with equal hard and nominal replay
counts per prompt:

```bash
make pi05-plan-data \
  MAIN_SOURCE_CONFIG=configs/data/drawer_four_skill_formal.json \
  HARD_SOURCE_CONFIG=configs/data/drawer_four_skill_hard_recovery.json \
  REPO_PREFIX=scuee_user06/vla_tidybench_drawer_v1

make pi05-convert-data
```

After the eight canonical files are present, the complete preparation and
three-stage training sequence can be launched with one resumable command:

```bash
make pi05-formal-pipeline \
  PI05_REPO_PREFIX=scuee_user06/vla_tidybench_drawer_v1 \
  TRAIN_STATE_FLAG=--resume
```

For a fresh experiment use `TRAIN_STATE_FLAG=--overwrite`. The pipeline invokes
the audited planner before any training, converts all four frozen manifests,
computes the nominal statistics for both the LoRA and full configuration asset
directories plus independent hard-mixture statistics for the full stage, and
then launches the LoRA, full, and hard-recovery stages.

The planner defaults to a 10% nominal validation split and a 20% hard-recovery
validation split, with at least eight nominal training, two nominal validation,
two hard-recovery training, and two hard-recovery validation episodes per
prompt. On the engineering profile this produces the prescribed 90/10 nominal
and 40/10 hard split per skill. It audits HDF5 shapes/success attributes, uses a
locked seed, rejects duplicate or cross-set episodes, and writes:

```text
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/main_train.json
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/main_validation.json
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/hard_validation.json
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/hard_mix_train.json
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/split_audit.json
```

The corresponding local LeRobot IDs are
`scuee_user06/vla_tidybench_drawer_v1_train`,
`scuee_user06/vla_tidybench_drawer_v1_validation`,
`scuee_user06/vla_tidybench_drawer_v1_hard_validation`, and
`scuee_user06/vla_tidybench_drawer_v1_hard_mix`. They do not need to be
published when training from the same mechanical disk.

## Guarded three-stage experiment

The public `vla_tidybench_drawer_four_skill_mvp` repository has only one
episode per skill. It is valid for plumbing and memory smoke tests, but the
runner refuses to use it for a formal run.

The three stages are deliberately separate:

1. `stage1-lora`: a cheap adapter baseline from π0.5-DROID (`5,000` steps,
   peak LR `2.5e-5`).
2. `stage2-full`: an independent full-parameter run from π0.5-DROID using
   factored Adafactor state and shard-all FSDP (`10,000` steps, peak LR
   `1e-5`). The vision encoder, 2B VLM, 300M action expert and action heads all
   receive gradients.
3. `stage3-hard-recovery`: continue full-parameter Adafactor training from the
   final stage-2 parameters on a replay-mixed hard-sample dataset (`3,000`
   steps, peak LR `2e-6`).

The stage-3 repository must already mix failure-recovery examples with normal
successful replay data. Do not train it on failures alone, which would cause
catastrophic forgetting of nominal behavior.

Run all three two-step systems checks on three GPUs:

```bash
make pi05-three-stage-synthetic-smoke  # no dataset needed; no task claim
make pi05-three-stage-smoke
```

The synthetic target validates model restore, LoRA/full optimization, FSDP and
checkpoint chaining only. Its losses and outputs must never be reported as task
performance.

The three-GPU synthetic systems run on 2026-08-25 produced the following
plumbing evidence (not task metrics):

| Stage | Mode | Two-step loss | Checkpoint |
| --- | --- | --- | --- |
| 1 | LoRA + AdamW | `2.1355`, `2.2328` | `pi05_tidybench_drawer_four_skill_lora/stage1-lora-smoke/1` |
| 2 | Full + factored Adafactor | `2.1372`, `2.2303` | `pi05_tidybench_drawer_four_skill_full/stage2-full-smoke/1` |
| 3 | Full + factored Adafactor, initialized from stage 2 | `2.0751`, `2.0246` | `pi05_tidybench_drawer_four_skill_full/stage3-hard-recovery-smoke/1` |

The stage-3 full checkpoint exports 31 files (`12,432,014,468` bytes). A
dataset-free identity-normalized forward check returned a finite `(16, 7)`
action chunk in `22.86 s` on cold JIT and `86.5 ms` after warm-up. Formal data
normalization and simulator success remain separate required gates.

The manifest-driven WebSocket service was also exercised end to end on the
same synthetic checkpoint. Checkpoint restore took `4.27 s`; the first request
including JIT took `14,776.8 ms` inside the policy; the second request took
`84.49 ms` inside the policy and `100.02 ms` round trip. The exact evidence is
checked in at `results/metrics/pi05_synthetic_policy_service_smoke.json`. These
are systems and latency measurements only, not drawer-task success metrics.

After freezing an episode-level train/validation split and publishing the
formal and hard-replay LeRobot repositories:

```bash
make pi05-three-stage-train \
  MAIN_DATASET_REPO=scuee_user06/vla_tidybench_drawer_v1_train \
  HARD_DATASET_REPO=scuee_user06/vla_tidybench_drawer_v1_hard_mix
```

Every newly launched stage checks the physical indices in
`CUDA_VISIBLE_DEVICES` before importing JAX. It waits until each selected GPU
has no other compute PID and at most `512 MiB` of baseline memory, reporting
the occupying PIDs every 30 seconds. The default timeout is six hours. Override
these only for a deliberate shared-GPU run with
`PI05_GPU_PREFLIGHT_MAX_USED_MIB` and `PI05_GPU_PREFLIGHT_TIMEOUT_S`;
`PI05_SKIP_GPU_PREFLIGHT=1` is an explicit emergency bypass and is not suitable
for the memory-constrained three-card full fine-tune.

Inspect the live experiment without parsing terminal output manually:

```bash
make pi05-experiment-status
```

This writes
`$VLA_TIDYBENCH_DATA/logs/pi05-three-stage-status.json` with the active stage,
latest complete checkpoint per stage, latest tqdm step/rate, detected error
signals, selected-GPU memory, training PID, and any foreign compute PIDs. Set
`PI05_STATUS_FLAG=--fail-on-conflict` when using it as a resource gate in an
external monitor.

Use `TRAIN_STATE_FLAG=--resume` only for a compatible interrupted run. The
default is `--overwrite`; experiment names are stable per stage so checkpoint
selection and comparison remain auditable. Resume is evaluated independently
for every stage: a stage with its final complete Orbax checkpoint is skipped,
an incomplete stage resumes from its latest complete numeric checkpoint, and a
stage that has not started is launched normally. A non-empty run directory
without any complete checkpoint is rejected instead of being overwritten.

Each newly launched training subprocess also writes append-only numeric metrics
to `train_metrics.jsonl` beside its checkpoints. Every record contains loss,
gradient norm, parameter norm, the dataset/config identity, and a unique process
session ID. A resumed process may replay steps after its last durable
checkpoint; the raw history is retained, while the summary uses the newest
record for each repeated step. Generate the auditable three-stage summary with:

```bash
make pi05-training-summary
```

The default report is
`$VLA_TIDYBENCH_DATA/logs/pi05-three-stage-metrics-summary.json`. Add
`TRAIN_METRICS_FLAG=--require-final-step` when the command must fail unless all
three final steps (`4999`, `9999`, and `2999`) are present. The metric file is
local and independent of WandB, so disabling external experiment tracking does
not discard the optimization trace.

Before a newly launched stage process returns success, it verifies the final
numeric checkpoint, required Orbax metadata, the unique embedded normalization
asset ID, the dataset identity, and the final JSONL metric step. A missing or
mismatched artifact makes that stage fail instead of allowing the outer runner
to advance with an unusable checkpoint.

## Validation sequence

Compute separate normalization statistics after the final episode-level splits
for the nominal and hard-replay datasets are frozen:

```bash
make pi05-prepare-norm-stats \
  MAIN_DATASET_REPO=owner/vla_tidybench_drawer_v1 \
  HARD_DATASET_REPO=owner/vla_tidybench_drawer_hard_mix_v1
```

Run a bounded three-GPU LoRA smoke test:

```bash
STEPS=2 BATCH_SIZE=3 FSDP_DEVICES=3 EXP_NAME=lora-smoke make drawer-four-skill-train-lora
```

Run a bounded full-fine-tuning memory test independently from the LoRA run:

```bash
STEPS=2 BATCH_SIZE=3 FSDP_DEVICES=3 EXP_NAME=full-smoke make drawer-four-skill-train-full
```

LoRA uses the 2B and 300M adapter variants, freezes their base parameters and
disables EMA. Full fine-tuning uses the standard π0.5 variants, trains every
parameter and starts with a lower default peak learning rate of `1e-5`. On the
three RTX 4090 host it uses factored Adafactor without momentum and
`PI05_FSDP_MIN_SIZE_MBYTES=0`; this avoids replicating full Adam moments for the
vocabulary embedding whose dimensions are not divisible by three. EMA is
disabled because its extra full-parameter copy exceeds 24-GB cards. On a
larger-memory host, opt in explicitly with `PI05_EMA_DECAY=0.99`.

## Export and real-time simulation inference

Serve the selected numeric stage-3 checkpoint first, then run the autonomous
four-skill gate on the Isaac Lab machine:

```bash
# Training/policy server
make drawer-policy-serve \
  CHECKPOINT=$VLA_TIDYBENCH_DATA/checkpoints/openpi-runs/\
pi05_tidybench_drawer_four_skill_full/stage3-hard-recovery/2999 \
  POLICY_GPU=1 POLICY_MODE=full POLICY_CONFIG_FLAG=--four-skill

# Isaac Lab machine: copy raw nominal HDF5 plus main_validation.json first.
# The suite uses 5 locked, distinct held-out contexts per skill and no
# teacher/DLS residual.
make pi05-eval-suite POLICY_HOST=<training-server-ip> POLICY_PORT=8000
```

The default engineering acceptance gate requires all four skills, five unique
seeds and five distinct held-out initial-state contexts per skill, at least
`60%` autonomous success for every skill, one exact checkpoint across all
rollouts, and overall P95 policy latency at most `250 ms`. The suite passes the
exact 20 paths from its current skill/seed matrix to the summarizer; unrelated
or stale HDF5 files below the evaluation directory are never counted toward the
gate. Every autonomous rollout also records the policy server and Isaac client
Git commits and dirty-worktree flags. Formal evaluation requires both ends to
use the same clean commit, and export requires that exact commit as well.
`main_validation.json` selects the contexts from the raw nominal HDF5 files.
Only each held-out episode's initial simulator state is restored: validation
actions are never replayed and are never supplied to the policy. The raw files
and manifest must therefore be available on the Isaac Lab machine at
`$VLA_TIDYBENCH_DATA/raw` and
`$VLA_TIDYBENCH_DATA/manifests/pi05-formal/main_validation.json`. Override
those locations explicitly when needed:

```bash
make pi05-eval-suite \
  POLICY_HOST=<training-server-ip> POLICY_PORT=8000 \
  EVAL_CONTEXT_MANIFEST=/absolute/path/main_validation.json \
  EVAL_DATA_ROOT=/absolute/path/raw
```

Formal rollouts use success predicate
`drawer_skill_v2_relative_stable`. OPEN and CLOSE must reach their drawer
thresholds; PICK must lift the bottle at least `0.08 m` relative to the frozen
context while the gripper is closed; PLACE requires the bottle inside the open
drawer and the gripper released; CLOSE additionally requires the bottle to
remain in the moving drawer frame. Every predicate must remain true for five
consecutive 20-Hz control steps. The rollout records the initial/final drawer,
bottle and gripper state, and the summarizer rejects older or missing predicate
versions, recomputes terminal success from those states, and rejects a forged or
inconsistent `success` label instead of silently mixing incomparable results.

Override `MIN_SUCCESS_RATE` or `MAX_P95_INFER_MS` only when documenting an
exploratory acceptance profile. Formal export accepts the locked profile or a
strictly stronger one; it refuses fewer than five episodes per skill, success
thresholds below `60%`, latency thresholds above `250 ms`, incomplete episode
evidence, or internally inconsistent counts. The report is written to
`$VLA_TIDYBENCH_DATA/eval/pi05-formal/evaluation.json`.

Publish the evaluated numeric checkpoint step as a stable deployment bundle:

```bash
make pi05-export-final \
  CHECKPOINT=$VLA_TIDYBENCH_DATA/checkpoints/openpi-runs/\
pi05_tidybench_drawer_four_skill_full/stage3-hard-recovery/2999 \
  DATASET_REPO=owner/vla_tidybench_drawer_hard_mix_v1 \
  EVAL_REPORT=$VLA_TIDYBENCH_DATA/eval/pi05-formal/evaluation.json
```

This creates
`$VLA_TIDYBENCH_DATA/checkpoints/deploy/pi05-tidybench-final/manifest.json`
plus a checksum-bound copy of `evaluation.json` and, by default, a complete
portable copy of the checkpoint weights. Set `EXPORT_STORAGE=symlink` only for
a same-server deployment where avoiding the additional 12-GB-class copy is
more important than portability. Both modes are built in a staging directory,
verified before publication, and preserve the prior bundle if replacement
fails. The format-2 manifest stores a deterministic SHA-256 over every relative
checkpoint path, file size and file byte. The policy service publishes that
digest into every rollout, the evaluation report requires one exact digest,
and export refuses a report whose evaluated digest differs from the checkpoint
being packaged. It also refuses failed, assisted, mixed-checkpoint, or
mismatched-checkpoint reports, and formal export requires a clean Git checkout.
Verify the bundle and start standard-model inference on a dedicated GPU using
the manifest-driven entry point:

```bash
DEPLOYMENT=$VLA_TIDYBENCH_DATA/checkpoints/deploy/pi05-tidybench-final
make pi05-verify-deployment DEPLOYMENT=$DEPLOYMENT
make pi05-deployment-serve DEPLOYMENT=$DEPLOYMENT POLICY_GPU=1
```

From a second shell, verify the real WebSocket serialization, handshake,
checkpoint identity, `(16, 7)` response and warm inference latency:

```bash
POLICY_PROBE_FLAG="--expect-deployment $DEPLOYMENT --require-evaluation --max-last-infer-ms 250" \
  make pi05-policy-probe
```

The verifier resolves the checkpoint link, checks required Orbax metadata,
file count, byte count, the complete checkpoint tree SHA-256, clean-code
provenance, evaluation SHA-256, autonomous-only status, gate result, and exact
evaluated checkpoint identity before model construction. It also requires the
manifest dataset repository to equal the checkpoint's unique embedded
normalization asset ID; the policy configuration is constructed with that ID,
so stage-3 hard-mix statistics cannot silently fall back to the MVP or nominal
dataset statistics. Direct-checkpoint serving discovers the same ID from the
checkpoint when `--dataset-repo` is omitted. Hashing the final
12-GB-class checkpoint can take roughly one to two minutes on a mechanical
disk. Policy mode and four-skill configuration are taken from the manifest, so
a full checkpoint cannot accidentally be loaded as an expert or LoRA model.

For a dataset-free restore/JIT/interface check before formal export, continue to
use the direct checkpoint smoke target. `--synthetic` supplies zero images/state
and explicit identity normalization and therefore does not validate physical
scaling:

```bash
make drawer-policy-smoke \
  CHECKPOINT=/absolute/numeric/checkpoint \
  POLICY_MODE=full POLICY_CONFIG_FLAG=--four-skill POLICY_INPUT_FLAG=--synthetic
```

An unvalidated synthetic systems bundle can also exercise the persistent
WebSocket service, but both bypasses must be explicit and are advertised in the
server handshake:

```bash
DEPLOYMENT_VALIDATION_FLAG=--allow-unvalidated \
DEPLOYMENT_SERVE_FLAG="--allow-unvalidated-deployment --synthetic-identity-norm" \
  make pi05-deployment-serve DEPLOYMENT=/absolute/synthetic-smoke-deployment
```

On the Isaac Lab machine, start the camera-enabled closed loop and point it at
the policy host:

```bash
POLICY_HOST=<training-server-ip> POLICY_PORT=8000 POLICY_SKILL=open \
  make drawer-policy-run
```

The simulation client executes four actions from each 16-action chunk, applies
the physical-action safety guard, replans from fresh table/wrist images, and
records inference latency and success in the evaluation HDF5.

## Offline-server checkpoint transfer

If the training server is behind a captive portal, bridge public GCS objects
through a networked workstation. The transfer script downloads one object at a
time, verifies the GCS MD5 locally, uploads with SCP, verifies the remote MD5,
then removes the workstation staging copy:

```powershell
python scripts/transfer_gcs_checkpoint.py `
  --bucket openpi-assets `
  --prefix checkpoints/pi05_droid/ `
  --remote scu-lab `
  --remote-root /data/$USER/vla-tidybench/checkpoints/openpi-assets/checkpoints/pi05_droid.partial `
  --identity-file $env:USERPROFILE/.ssh/id_ed25519
```

Policy creation also needs the public PaliGemma tokenizer. Cache it at the
path OpenPI derives from `OPENPI_DATA_HOME`:

```powershell
python scripts/transfer_gcs_checkpoint.py `
  --bucket big_vision `
  --object paligemma_tokenizer.model `
  --remote scu-lab `
  --remote-root /data/$USER/vla-tidybench/checkpoints/big_vision `
  --identity-file $env:USERPROFILE/.ssh/id_ed25519
```

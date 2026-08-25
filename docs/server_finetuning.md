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
to `$VLA_TIDYBENCH_DATA/raw`. Start from the checked-in source templates:

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

The planner defaults to at least eight nominal training, two validation and two
hard-recovery episodes per prompt. It audits HDF5 shapes/success attributes,
uses a locked seed, rejects duplicate or cross-set episodes, and writes:

```text
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/main_train.json
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/main_validation.json
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/hard_mix_train.json
$VLA_TIDYBENCH_DATA/manifests/pi05-formal/split_audit.json
```

The corresponding local LeRobot IDs are
`scuee_user06/vla_tidybench_drawer_v1_train`,
`scuee_user06/vla_tidybench_drawer_v1_validation`, and
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

After freezing an episode-level train/validation split and publishing the
formal and hard-replay LeRobot repositories:

```bash
make pi05-three-stage-train \
  MAIN_DATASET_REPO=scuee_user06/vla_tidybench_drawer_v1_train \
  HARD_DATASET_REPO=scuee_user06/vla_tidybench_drawer_v1_hard_mix
```

Use `TRAIN_STATE_FLAG=--resume` only for a compatible interrupted run. The
default is `--overwrite`; experiment names are stable per stage so checkpoint
selection and comparison remain auditable. Resume is evaluated independently
for every stage: a stage with its final complete Orbax checkpoint is skipped,
an incomplete stage resumes from its latest complete numeric checkpoint, and a
stage that has not started is launched normally. A non-empty run directory
without any complete checkpoint is rejected instead of being overwritten.

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

# Isaac Lab machine: 5 locked seeds per skill, no teacher/DLS residual
make pi05-eval-suite POLICY_HOST=<training-server-ip> POLICY_PORT=8000
```

The default engineering acceptance gate requires all four skills, five unique
seeds per skill, at least `60%` autonomous success for every skill, one exact
checkpoint across all rollouts, and overall P95 policy latency at most `250 ms`.
Override `MIN_SUCCESS_RATE` or `MAX_P95_INFER_MS` only when documenting a
different acceptance profile. The report is written to
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
plus a checksum-bound copy of `evaluation.json` and a same-disk `checkpoint`
link. Export refuses failed, assisted, mixed-checkpoint, or mismatched-checkpoint
reports, and formal export requires a clean Git checkout. Verify the bundle and
start standard-model inference on a dedicated GPU using the manifest-driven
entry point:

```bash
DEPLOYMENT=$VLA_TIDYBENCH_DATA/checkpoints/deploy/pi05-tidybench-final
make pi05-verify-deployment DEPLOYMENT=$DEPLOYMENT
make pi05-deployment-serve DEPLOYMENT=$DEPLOYMENT POLICY_GPU=1
```

The verifier resolves the checkpoint link, checks required Orbax metadata,
file count and byte count, clean-code provenance, evaluation SHA-256,
autonomous-only status, gate result, and exact checkpoint identity before model
construction. Policy mode and four-skill configuration are taken from the
manifest, so a full checkpoint cannot accidentally be loaded as an expert or
LoRA model.

For a dataset-free restore/JIT/interface check before formal export, continue to
use the direct checkpoint smoke target. `--synthetic` supplies zero images/state
and explicit identity normalization and therefore does not validate physical
scaling:

```bash
make drawer-policy-smoke \
  CHECKPOINT=/absolute/numeric/checkpoint \
  POLICY_MODE=full POLICY_CONFIG_FLAG=--four-skill POLICY_INPUT_FLAG=--synthetic
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

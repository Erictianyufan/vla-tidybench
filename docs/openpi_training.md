# π0.5 data and training smoke

## Locked interfaces

- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- LeRobot: `0.1.0`.
- Dataset: `erictianyufan/vla_tidybench_stack_m1_smoke`.
- Rate and horizon: 20 Hz, 16 actions (0.8 seconds).
- Input: table RGB, wrist RGB, 18D joint position/velocity state and language.
- Output: canonical 7D IK-relative physical action, padded to π0.5's 32D model width.
- Model: π0.5-DROID initialization with PaliGemma and Action Expert LoRA variants.

The converter does not expose the recorded `object`, `cube_positions`,
`cube_orientations` or serialized simulator `states` fields. It converts
Isaac actions through the same `ActionAdapter` used at deployment and fails if
the `SafetyGuard` would modify any expert label.

## Commands

```bash
cd /home/ubuntu/mycode/vla-tidybench
make convert-openpi-smoke
make openpi-norm-stats
make openpi-data-smoke
make train-pi05-smoke
```

The generated data, assets and checkpoints live under
`/home/ubuntu/data/vla-tidybench` and are not tracked by Git.

## Verified smoke result

- Conversion: 17 episodes, 5,554 frames, approximately 462 MB.
- Norm stats SHA-256:
  `0d7b005101e00a7b59886f8ae33094138428b6af503c66e78ceca16041a76567`.
- Real transformed batch: `(B,32)` state, `(B,16,32)` actions, three
  `(B,224,224,3)` model image slots; the third view is masked padding.
- Single RTX 4090 24 GB: checkpoint loaded, but the first backward step needed
  another 5.37 GiB and failed with OOM. This is an expected resource result.
- Two RTX 4090 24 GB with `FSDP_DEVICES=2`, batch 2: two optimizer steps
  completed; losses were 1.6990 and 1.9058, and the step-1 Orbax checkpoint was
  finalized. Peak observed memory was approximately 17.5/17.1 GB.

Two steps are only a systems smoke test. They are not reported as a trained or
evaluated policy. Formal fine-tuning starts after a larger immutable dataset,
episode-level train/validation split and frozen evaluation seeds are prepared.

## Checkpoint integrity note

The first cloud download left a truncated 7.8 GB `params/` directory. A retry
created the complete 12 GB checkpoint under `pi05_droid/params/`; the project
config points explicitly to the complete nested directory. The truncated copy
is retained but never loaded. No checkpoint is accepted merely because a path
exists: the smoke test must complete Orbax restore before training begins.

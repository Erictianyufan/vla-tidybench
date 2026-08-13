# Baseline data collection

## Action and observation contract

Both human and scripted demonstrations use the official Franka visuomotor
environment and the same canonical 7D action:

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

The task applies a `0.5` action scale and computes joint targets with native
damped-least-squares differential inverse kinematics. The recorder stores the
table RGB camera, wrist RGB camera, robot state, action and full simulator state
required for replay.

## Scripted teacher

`scripts/collect_scripted_stack.py` reads privileged cube and end-effector
poses, then executes this task-space state machine:

```text
ABOVE_PICK -> DESCEND_PICK -> CLOSE -> LIFT
-> ABOVE_PLACE -> DESCEND_PLACE -> OPEN -> RETREAT -> SETTLE
```

It first places red on blue and then green on red. Grasp height and stack
geometry are checked online. Failed episodes are discarded. Episode reset uses
`env.reset()` only; it does not call the full `env.sim.reset()` path that caused
the Direct GPU API error during interactive recording.

The teacher uses privileged state only to produce expert labels. Future π0.5
conversion must exclude `cube_positions`, `cube_orientations`, the compound
`object` vector and saved simulator `states` from deployable policy inputs.

## Commands

Generate and replay one smoke episode:

```bash
cd /home/ubuntu/mycode/vla-tidybench
make scripted-smoke
./scripts/replay_stack_demos.sh \
  /home/ubuntu/data/vla-tidybench/raw/stack_scripted_smoke.hdf5
```

Generate seven successful episodes:

```bash
make scripted-collect
```

Merge episodes using an explicit replay-validated allowlist:

```bash
/home/ubuntu/env_isaaclab/bin/python scripts/merge_stack_datasets.py \
  --source '/home/ubuntu/data/vla-tidybench/raw/stack_human.hdf5::0,1::human_teleop' \
  --source '/home/ubuntu/data/vla-tidybench/raw/stack_scripted.hdf5::0,1,2,3,4,5::scripted_truth_teacher' \
  --source '/home/ubuntu/data/vla-tidybench/raw/stack_scripted_replacement.hdf5::0::scripted_truth_teacher' \
  --source '/home/ubuntu/data/vla-tidybench/raw/stack_scripted_replacement_2.hdf5::0::scripted_truth_teacher' \
  --output /home/ubuntu/data/vla-tidybench/raw/stack_train_candidate_10.hdf5 \
  --overwrite

./scripts/replay_stack_demos.sh \
  /home/ubuntu/data/vla-tidybench/raw/stack_train_candidate_10.hdf5
```

## QA decision

The human recorder exported three successful episodes, although the operator
reported four attempts. The fourth attempt did not satisfy the success predicate
and was not exported. During strict physical replay, human `demo_2` was not
repeatable and was excluded from the training candidate.

The first scripted batch succeeded online 7/7. Six replayed reliably; the last
contact-rich episode failed strict replay and was excluded. Two more conservative
version-2 teacher episodes were generated and each replayed 1/1. The final
candidate therefore contains two human and eight scripted episodes and passes a
full 10/10 replay.

This filtering does not delete raw episodes. It records the accepted provenance
inside the merged HDF5 and in the checked-in dataset manifest.

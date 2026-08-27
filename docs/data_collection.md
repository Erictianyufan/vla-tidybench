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

## Mimic annotation and generation smoke

The replay-validated ten-episode candidate was automatically annotated with
Isaac Lab Mimic. All ten episodes completed during annotation and contain the
required end-effector, object, target-pose and `grasp_1`, `stack_1`, `grasp_2`
termination-signal fields.

```bash
cd /home/ubuntu/mycode/vla-tidybench
make annotate
NUM_ENVS=4 make mimic-smoke
./scripts/replay_stack_demos.sh \
  /home/ubuntu/data/vla-tidybench/raw/stack_mimic_smoke.hdf5
```

The smoke generator required 30 attempts to produce ten online-success
episodes (33.3%). A later strict physical replay reproduced 7/10; episode IDs
2, 4 and 8 did not reproduce. These are separate metrics: Mimic exports an
episode only after its online success predicate passes, while a later reset and
replay is not guaranteed to follow the exact same contact dynamics.

The original ten generated episodes remain immutable. A seven-episode subset
records the first replay allowlist without deleting or relabelling the three
non-repeatable trajectories. That accepted subset passed a second independent
strict replay at 7/7:

```bash
/home/ubuntu/env_isaaclab/bin/python scripts/merge_stack_datasets.py \
  --source '/home/ubuntu/data/vla-tidybench/raw/stack_mimic_smoke.hdf5::0,1,3,5,6,7,9::mimic_replay_validated' \
  --output /home/ubuntu/data/vla-tidybench/raw/stack_mimic_smoke_accepted_7.hdf5 \
  --overwrite
```

This is a smoke-test dataset, not the final training corpus. The checked-in
manifest is the source of truth for hashes, counts and QA decisions.

## Drawer atomic-skill prerequisites

`scripts/collect_scripted_drawer.py` records each atomic command from the
state in which that command is meaningful:

- OPEN: drawer closed, medicine bottle on the table;
- PICK: drawer open to `0.36 m`, medicine bottle on the table;
- PLACE: drawer open to `0.39 m`, medicine bottle already held by the Franka;
- CLOSE: drawer open to `0.36 m`, medicine bottle already inside it.

These states are part of the environment reset configuration, so the recorder
stores them in each HDF5 episode's `initial_state`; replay and evaluation do not
depend on an unrecorded teleport. Formal success uses the versioned predicate
in `source/vla_tidybench/task_metrics.py`. Simulator truth is used only for
teacher control and metrics, never for policy observations.

# VLA-TidyBench

Language-conditioned long-horizon manipulation with a Franka Panda, Isaac Sim,
Isaac Lab, pi0.5 and failure-driven residual reinforcement learning.

The final demonstration is:

> Put the red mug into the top drawer and close it.

The task is deliberately staged. The project first reproduces the official
Franka cube-stack task, then builds a language-conditioned bin task, and only
then adds drawers, pi0.5 adaptation and residual RL.

## Runtime layout

The deployment keeps the simulator and VLA in separate processes and Python
environments:

```text
GPU 0: Isaac Sim 6.0.1 + Isaac Lab + two RGB cameras
GPU 1: openpi pi0.5 policy server / training

Isaac worker -- WebSocket -- openpi policy server
```

The vendor Isaac Lab checkout is treated as read-only. This repository is the
only place where project code is developed.

## Current milestone: M0

- [x] Audit hardware and preinstalled software.
- [x] Start the official Franka IK-relative task on GPU 0.
- [x] Confirm the action space is 6D relative end-effector pose plus 1D gripper.
- [ ] Run the finite-step state and visuomotor smoke tests.
- [ ] Record, replay and Mimic-expand the first human demonstrations.

See `docs/deployment.md` for server paths and `docs/milestones.md` for gates.

## Quick start on the provisioned server

```bash
cd /home/ubuntu/mycode/vla-tidybench
make doctor
make test
make sim-smoke
make sim-camera-smoke
make protocol-smoke
```

The next interactive gate is `make record`, which must be started in the cloud
desktop rather than a headless SSH shell. After ten successful demonstrations:

```bash
make replay
make annotate
make mimic-smoke
```

The Isaac command must run through `scripts/run_isaac.sh`. It activates the
preinstalled Python 3.12 environment before invoking `isaaclab.sh`; the system
Python is intentionally not used.

## Safety and scope

- The policy actor receives RGB, deployable proprioception, language and prior
  actions only. Simulator truth is reserved for rewards, critics and metrics.
- The canonical action is physical 7D:
  `[dx, dy, dz, dRx, dRy, dRz, gripper]` in metres, radians and `{-1,+1}`.
- Training, conversion and deployment share one action adapter.
- RL is not started until the imitation-learning baseline and failure
  distribution are reproducible.
- This is a simulation-first project; it does not claim sim-to-real results.

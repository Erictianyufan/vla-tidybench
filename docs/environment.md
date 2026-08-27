# Provisioned environment

Audit date: 2026-08-13 (Asia/Shanghai)

> Historical host profile: this section records the previously validated cloud
> Isaac host. A 2026-08-28 probe found no NVIDIA runtime on the current Windows
> workstation and no `isaaclab`/`omni` installation on the training server
> `202.115.53.61`. Formal evaluation therefore still requires reconnecting this
> host or provisioning an equivalent Isaac Lab machine; it must not be reported
> as executed on the training server.

## Hardware

- Host: cloud instance identifier omitted from the repository
- GPU: 2 x NVIDIA GeForce RTX 4090, 24564 MiB each
- GPU topology: PHB, no NVLink; memory is not pooled
- CPU: AMD EPYC 7542, 16 physical / 32 logical cores
- RAM: 125 GiB
- Swap: none at provisioning time
- Root filesystem: 485 GiB usable, 429 GiB free at audit time

## Software

- Ubuntu 22.04.4 LTS, kernel 5.15.0-153-generic
- NVIDIA driver 570.172.08
- Isaac Sim 6.0.1.0
- Isaac Lab vendor checkout: `28a37cecdd433c22d9eabd6a5954add9f13a8951`
- Vendor describe: `perf-2026-06-24-dirty`
- Isaac Python: 3.12.13
- PyTorch: 2.10.0+cu128
- Isaac Lab Python package: 6.1.11
- Isaac Lab tasks package: 1.10.9

The cloud vendor changed `isaaclab/utils/assets.py` so the Nucleus asset root
points to `/home/ubuntu/readonly/Assets/Isaac/6.0`. This is required for the
pre-mounted local asset bundle and must not be reverted.

## Validated baseline

`Isaac-Stack-Cube-Franka-IK-Rel-v0` successfully created one environment on
GPU 0. The environment reported a 20 Hz policy step (`0.01 s * 5`), a 7D
action space and the expected Franka joints and end-effector body.

The vendor launcher must be run after activating `/home/ubuntu/env_isaaclab`.
Otherwise it falls back to Ubuntu's Python 3.10 and fails before simulation.

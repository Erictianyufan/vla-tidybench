# Server deployment

## Paths

```text
/home/ubuntu/IsaacLab                         vendor simulator checkout; do not edit
/home/ubuntu/env_isaaclab                     vendor Python 3.12 environment
/home/ubuntu/openpi                           official openpi, detached pinned commit
/home/ubuntu/openpi/.venv                     isolated Python 3.11 JAX environment
/home/ubuntu/mycode/vla-tidybench             project repository
/home/ubuntu/data/vla-tidybench               datasets, checkpoints, logs and videos
```

## GPU ownership

- GPU 0: Isaac simulation, PhysX and RTX cameras.
- GPU 1: OpenPI/JAX policy service and model training.
- The two 24 GiB devices are connected through a PCIe host bridge and their
  memory is not pooled.

Isaac is launched with renderer multi-GPU disabled so it does not consume GPU
1 for rendering. OpenPI uses `CUDA_VISIBLE_DEVICES=1` inside its own process.

## Smoke-test commands

```bash
cd /home/ubuntu/mycode/vla-tidybench
make doctor
make test
make sim-smoke
make sim-camera-smoke
make protocol-smoke
```

## Model download

The pi0.5-DROID checkpoint is a model-runtime smoke-test baseline, not the final
Franka policy. Its DROID output has eight dimensions and must never be forwarded
directly to the project's seven-dimensional Franka action adapter.

```bash
./scripts/download_pi05_droid.sh
tail -f /home/ubuntu/data/vla-tidybench/logs/pi05_droid_download.log
```

The final project policy will use a dedicated Franka data transform, norm stats
and fine-tuned checkpoint.

## Known vendor-image constraints

- `isaaclab.sh` fails if `/home/ubuntu/env_isaaclab` is not activated first.
- The vendor checkout has a required local-asset patch and must stay dirty.
- ALOHA and LIBERO OpenPI git submodules returned HTTP 403 during provisioning.
  The core OpenPI environment installed successfully and this project does not
  depend on those example source trees.
- The server initially has no swap. Monitor host RAM during large data
  conversions; do not add swap without an explicit operational decision.

## Security

SSH key authentication is configured for deployment. Rotate the initial cloud
password because it was shared in chat. Keep policy port 8000 bound to localhost
unless a firewall and API-key layer are deliberately configured.


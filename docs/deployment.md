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

`make doctor` is also the formal evaluation-host admission gate. It requires a
clean project checkout, the pinned Isaac packages/vendor commit, usable NVIDIA
devices, all 40 held-out contexts with matching content hashes, and the
evaluation runtime scripts. A failure here means the machine is not an eligible
source of the final `evaluation.json`.

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

## Formal checkpoint identity

Production exports use deployment manifest format 3. Its `sha256-tree-v1`
digest covers every checkpoint relative path, file size and file byte. The same
digest is advertised by the policy server, copied into each closed-loop rollout
and evaluation report, checked during export, and verified again before serving
the deployment. File count and total bytes remain useful diagnostics but are not
treated as cryptographic identity.

Format 3 additionally embeds a checksum-bound `training_completion.json`. It
binds the final numeric checkpoint to the terminal loss/gradient/parameter norm,
dataset and normalization asset, clean training-project commit, OpenPI source
tree fingerprint, initialization-parameter fingerprint, and checkpoint content
digest. A formal deployment is rejected if this training proof, the autonomous
evaluation report, or the copied checkpoint disagrees with either of the other
two artifacts.

Formal exports copy the checkpoint into the deployment bundle by default, so
the directory can be transferred to an Isaac host without retaining the source
training path. `EXPORT_STORAGE=symlink` is the explicit zero-copy option for a
deployment that remains on the training server. Export verifies the copied
tree before atomically publishing it; replacement preserves the previous
deployment if staging or verification fails.

The formal manifest's `dataset_repo` must match the single normalization asset
ID embedded below `checkpoint/assets`. The verifier enforces this equality and
the policy server passes the verified ID into the reconstructed training
configuration before OpenPI restores normalization statistics. This prevents a
valid weight tree from being served with statistics from a default or stale
dataset.

Formal rollouts bind both halves of the distributed system to source control:
the policy service advertises its runtime Git commit and clean-state flag, and
the Isaac client records its own. The evaluation gate requires the two commits
to be identical and clean. The exporter and deployment verifier then require
the report commit to match the clean checkout used to package and serve the
weights.

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


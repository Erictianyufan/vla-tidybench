# Public source release checklist

Target: `Erictianyufan/vla-tidybench`, visibility **Public**.

The public repository contains audited source code, reproducible command
interfaces, small metric manifests and a redacted GIF preview. It intentionally
does not publish raw datasets, LeRobot caches, model weights, RL replay buffers,
cloud connection details. The full-resolution MP4 is published only as a
GitHub Release asset, not as a Git blob.

## Reproducibility

- [x] README commands correspond to checked-in entrypoints.
- [x] Release commit and environment versions are documented.
- [x] Dataset and checkpoint results are summarized without uploading restricted artifacts.
- [x] README distinguishes verified smoke tests from uncompleted long-running experiments.

## Privacy and security

- [x] The pre-publication audit passes.
- [x] No passwords, access tokens, private keys, server IPs or shell histories are tracked.
- [x] No raw HDF5/LeRobot data, model weights, RL replay buffers or full MP4 is tracked.
- [x] Git credentials are never embedded in the remote URL.
- [x] The committed GIF contains simulator camera views and project labels only.

## GitHub upload

- [ ] Create the repository with **Public** visibility and without generated starter files.
- [ ] Push the reviewed default branch.
- [ ] Create a public release and attach the full MP4 plus SHA-256 checksum.
- [ ] Verify the repository is anonymously readable.
- [ ] Verify restricted filename patterns are absent from Git history.

## Release commands

```bash
make test
make extension-smoke
make prepublish
git status --short
```

No personal access token is written into this repository or displayed in its
documentation. The repository currently has no public license.

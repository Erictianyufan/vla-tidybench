# Private GitHub release checklist

Target: `Erictianyufan/vla-tidybench`, visibility **Private**.

Repository creation and upload happen only after the project passes its final
release gate. The GitHub action is confirmed at execution time and the
repository visibility is verified again after upload.

## Reproducibility

- [ ] All README commands exist and run from a clean checkout.
- [ ] Release commit, exact environment versions and dependency locks are saved.
- [ ] Dataset and checkpoint SHA-256 values are recorded without uploading the
      restricted artifacts themselves.
- [ ] Final metrics were generated from the locked test set after model freeze.
- [ ] README contains measured values only and clearly labels limitations.

## Privacy and security

- [ ] `make prepublish` passes from a clean worktree.
- [ ] No passwords, access tokens, private keys, server IPs or shell histories
      appear in tracked files, Git history, video frames or metadata.
- [ ] No raw HDF5/LeRobot data, model weights, RL replay buffers or full MP4 is
      tracked by Git.
- [ ] Git remote uses SSH or GitHub credential storage; credentials are never
      embedded in a remote URL.
- [ ] Video metadata and screenshots contain no cloud-provider credentials.

## GitHub upload

- [ ] Create the repository with **Private** visibility and without an
      auto-generated README, `.gitignore` or license.
- [ ] Push the reviewed default branch and tags.
- [ ] Confirm branch-protection/Actions settings are appropriate for a private
      experimental repository.
- [ ] Create a private release and attach the final MP4, `SHA256SUMS`, metrics
      summary and run manifest.
- [ ] Open the repository URL in a signed-out/incognito session and confirm it
      is not accessible.
- [ ] Keep the repository private until the owner explicitly approves a later
      visibility change.

## Suggested final commands

Run locally in the release candidate:

```bash
make test
make protocol-smoke
make eval-final
make package-demo INPUT=/absolute/path/to/final_run.mp4
git status --short
make prepublish
git tag -a v0.1.0 -m "VLA-TidyBench private demo release"
```

The GitHub repository is created through the authenticated GitHub session (or a
separately authorized CLI). No personal access token is written into this
repository or into a command shown in the README.

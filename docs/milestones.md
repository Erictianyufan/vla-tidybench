# Milestone gates

## M0: official baseline

- [x] Hardware and software audit captured.
- [x] Franka IK-relative state environment runs for finite steps.
- [x] Franka visuomotor environment returns two 200 x 200 RGB cameras.
- [x] Canonical 7D action adapter and tests exist.
- [x] Simulator-to-policy-server wire protocol passes across isolated Python environments.
- [x] Human demonstrations recorded, audited and preserved as immutable raw data.
- [x] Scripted truth/FK/DLS-IK teacher produces independent two-camera HDF5 episodes.
- [x] Ten-episode mixed training candidate passes full replay validation (10/10).
- [x] Mimic smoke generation reaches ten successful episodes; generation and
  strict replay rates are recorded separately.

No custom drawer task, model fine-tuning or RL starts before the M0 data gates
are complete.

## M1: data and first learned baseline

- [ ] Record and replay 20–30 accepted IK-relative human demonstrations.
- [ ] Generate a quality-controlled Mimic dataset.
- [ ] Convert HDF5 episodes to a versioned LeRobot/openpi dataset.
- [ ] Train and evaluate a small behavior-cloning sanity baseline.

## M2: drawer scene and atomic skills

- [ ] Validate OPEN, PICK, PLACE and CLOSE resets and success predicates.
- [ ] Validate the two RGB cameras and canonical 7D action in the custom task.
- [ ] Pass a scripted-teacher gate before collecting learned-policy data.

## M3: π0.5 adaptation

- [ ] Fine-tune π0.5 from an immutable dataset version.
- [ ] Pass atomic-skill and closed-loop validation gates.
- [ ] Freeze normalization statistics, prompt set and best checkpoint.

## M4: long-horizon integration

- [ ] Integrate the policy server with receding-horizon action execution.
- [ ] Run OPEN → PICK → PLACE → CLOSE through the deterministic TaskGraph.
- [ ] Capture a versioned failure taxonomy on validation seeds.

## M5: targeted residual RL

- [ ] Freeze the VLA and select one pre-registered failure bottleneck.
- [ ] Validate reward, replay schema, zero-residual equivalence and actor input privacy.
- [ ] Train and evaluate the bounded residual specialist.
- [ ] Enable it only if paired improvement and safety gates pass.

An informative negative RL result still completes the experiment, but it is not
released as an “RL-enhanced” policy. In that case the final TaskGraph keeps the
frozen imitation-learning route.

## M6: final evaluation and private release

- [ ] Freeze code, configs, checkpoint hashes, dataset hashes and test seeds.
- [ ] Run the locked ID/OOD evaluation exactly once.
- [ ] Record an uninterrupted successful main rollout and representative failures.
- [ ] Package the MP4, preview GIF, metrics and checksums.
- [ ] Replace pending README results with generated, auditable numbers.
- [ ] Pass `make prepublish` with a clean worktree.
- [ ] Create `Erictianyufan/vla-tidybench` as a **private** GitHub repository.
- [ ] Push the signed-off source and attach the MP4 to a private release.
- [ ] Verify anonymous access cannot view the repository or release.

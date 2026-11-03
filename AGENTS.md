# Agent guide

## Purpose and scope

This repository adapts Dream-VLA to eight Panda manipulation task families in
robosuite. The released policy is
[`RobotLearningLab/Jocasta-diff-VLA`](https://huggingface.co/RobotLearningLab/Jocasta-diff-VLA).
Read `README.md`, `docs/multitask.md`, and `THIRD_PARTY.md` before changing the
training or inference path.

The architecture, Flow Matching loss, sampler, action head, and projectors come
from the Dream-VLX authors. Our contribution here is the Panda integration and
training orchestration. Preserve upstream attribution and pinned revisions.

## Repository map

- `multitask/train.py`: current eight-task training entrypoint.
- `multitask/config.json`: recorded training recipe.
- `multitask/data.py`, `balanced_index.py`, and `prompt.py`: data, sampling, and
  language-input contracts. `docs/multitask.md` describes the required cache.
- `multitask/inference.py`: inference used by the released multitask policy.
- `bindings.py`: upstream model bindings and complete trainable components.
- `load_policy.py`: public release loader and `predict_action` interface.
- `panda_adapter.py`, `robot_state.py`: action conversion and state conventions.
- Root `train.py`, `data.py`, `inference.py`, and `panda-author.json`: historical
  single-task Lift recipe, not the released eight-task training entrypoint.
- `third_party/Dream-VLX`, `dlimp-upstream`: pinned Git submodules.
- `patches/dream-vlx-panda.patch`: explicit upstream integration changes.
- `upstream-vla` and `multitask/recipe`: required symlinks; preserve their targets.
- `evaluation-results.json`, `provenance/`: recorded results and verification.

## Working rules

- Inspect `git status` before editing. Preserve unrelated changes. The Dream-VLX
  submodule may already be dirty because the documented Panda patch is applied;
  do not reset it or apply the patch twice.
- Follow the installation and launch commands in the README and multitask guide.
  Clone with submodules. Do not silently upgrade dependencies or upstream pins.
- Keep upstream changes explicit in the patch and document their attribution.
- Use authorized GPU pods for model execution, training, simulation, and research
  validation. Do not start local computational workloads unless explicitly
  authorized. Local source/document editing and Git operations are allowed.
- Use a new durable output directory for a new run. Preserve completed data,
  checkpoints, failure diagnostics, and existing evidence.
- Do not change experiment seeds, budgets, scene cohorts, or normalization to
  improve a reported result. Treat a changed protocol as a separate experiment.
- Keep credentials, cluster-specific configuration, datasets, weights, and
  generated run artifacts out of Git. Never put tokens in commands or logs.

## Model and simulator contracts

- Inputs are two uint8 RGB images of shape `(224, 224, 3)`, an instruction, and
  eight proprioception values: XYZ, axis-angle orientation, and two gripper joints.
- The public prediction is an `(8, 7)` action block: world-frame XYZ deltas in
  metres, extrinsic XYZ Euler deltas in radians, and gripper openness.
- These are physical actions, not normalized OSC commands. Preserve the
  `PandaAdapter` conversion and recorded observation/normalization conventions.
- Load the pinned base plus LoRA, action head, noisy-action projector, and
  proprioception projector. The LoRA adapter alone is not the full policy.
- The public loader requires exactly one visible CUDA GPU. Keep model inference
  and EGL rendering in separate processes on different physical GPUs. Logical
  device indices alone do not establish physical separation.
- Preserve `hard_reset=False` for the documented evaluation. Training on cached
  observations does not require EGL.
- On GPU faults or a `GPU_STOP` signal, stop the affected workload and preserve
  diagnostics. Do not attempt stress tests or blind retries.

## Validation and reporting

- For documentation-only edits, review the diff, check referenced paths, and run
  `git diff --check`; no training or simulation is needed.
- For code changes, choose bounded checks in an authorized GPU pod that exercise
  the changed behavior. Report exactly what was and was not tested.
- Model/loss changes need upstream loss parity, optimizer coverage of all trained
  components, finite gradients, and checkpoint reload checks. Root `check.py` is
  a single-task engineering preflight, not a generic multitask test suite.
- Resume changes need model, optimizer, RNG, and data-cursor equivalence checks;
  `compare_resume.py` documents the existing engineering comparison.
- Loader/inference changes need component completeness and deterministic action
  comparisons on saved inputs from the relevant task families. A successful
  forward pass or finished training loop is not a successful task evaluation.
- Preserve historical provenance hashes. Modified code requires separately
  identified evidence; do not rewrite old receipts to make new code appear to
  have produced an existing result.
- The published result is 332/512 sustained successes on development scenes,
  from one training seed. It is not an official LIBERO or real-robot benchmark.
  The PickPlace baseline comparison changes the complete policy interface, so
  it does not isolate a backbone or diffusion-mechanism effect.
- Raw demonstrations and optimizer/RNG resume files are not published. Do not
  claim that a fresh clone alone reproduces training or that public inference
  weights are an exact training-resume checkpoint.
- Summarize the change, validation, and material limitations when handing off.

# Jocasta-diff-VLA — Diffusion-VLA tuning

**Model: [RobotLearningLab/Jocasta-diff-VLA](https://huggingface.co/RobotLearningLab/Jocasta-diff-VLA)**

Dream-VLA-7B adapted to eight simulated Panda manipulation task families in robosuite using the Dream-VLX authors' Flow Matching loss. The fixed final checkpoint (100,000 steps) achieved **332/512 sustained successes (64.8%)** on our development scenes. This is one trained policy and one training seed, not an official LIBERO benchmark or real-robot result.

| Task | Sustained successes |
|---|---:|
| Lift |64/64|
| PickPlace |40/64|
| Stack |43/64|
| Door |34/64|
| NutAssembly |32/64|
| ToolHang |9/64|
| Push |61/64|
| Drawer |49/64|

The Hub release includes the **LoRA adapter plus trained action head, noisy-action projector and proprioception projector**, normalization statistics and training configuration. The pinned original Dream-VLA base weights are downloaded separately by the loader. Loading only the LoRA adapter does not reconstruct this policy.

## Install the pinned code

```bash
git clone --recurse-submodules https://github.com/alanrbtx/Diffusion-VLA-tuning.git
cd Diffusion-VLA-tuning
git -C third_party/Dream-VLX apply --check ../../patches/dream-vlx-panda.patch
git -C third_party/Dream-VLX apply ../../patches/dream-vlx-panda.patch
python -m pip install -r requirements.lock -r requirements-multitask.txt
```

Apply the patch once. `upstream-vla` points to the pinned Dream-VLX submodule; a GitHub ZIP does not include submodules. Reference environment: Linux, Python3.10.18, PyTorch2.5.1+cu124, Transformers4.51.3, BF16/SDPA. Dependencies and the inference release are validated in an allocated CUDA pod; a fresh public-index installation is not claimed as tested.

## Load and predict

Expose exactly one CUDA GPU to the model process. For simulation, run EGL on a **different physical GPU**, using a separate process as in the evaluation.

```python
import numpy as np
from load_policy import load_policy, predict_action

policy, processor, config = load_policy('RobotLearningLab/Jocasta-diff-VLA')
# For reproducibility, also pass revision='<Hub commit>' and optionally
# base_model_path='/path/to/pinned/Dream-VLA-7B' to reuse local base weights.

# primary, wrist: uint8 RGB arrays of shape (224, 224, 3)
# proprio: float32 (8,), xyz + axis-angle orientation + two gripper joints
# Use the same observation conventions as the recorded Panda data.
actions = predict_action(policy, processor, config, primary, wrist, proprio,
                         'pick up the red cube', seed=84907)
assert actions.shape == (8, 7)
```

Each action contains physical world-frame XYZ deltas in metres, extrinsic XYZ Euler deltas in radians and gripper openness. **Do not send these directly as normalized robosuite OSC commands.** `panda_adapter.PandaAdapter` performs the recorded controller conversion; `robot_state.quat2axisangle` implements the recorded proprioception orientation convention. Replan after each8-action block.

`load_policy.py` verifies the published file hashes, loads the exact base revision and all trained components, freezes the model, and uses the same deterministic inference mode as the evaluated checkpoint. It does not train, initialize a simulator or claim a successful task from a forward pass.

The model repository's root `config.json` describes the release layout and pinned
base; its source is [`release/config.json`](release/config.json). The existing
loader uses `release.json` for integrity checks and downloads the complete
snapshot, including this metadata. Older pinned releases remain supported.
This is package metadata, not a Transformers AutoModel configuration.
Normal Hub snapshot downloads include it, enabling Hugging Face's standard
[download accounting](https://huggingface.co/docs/hub/models-download-stats).
No additional telemetry requests are sent. The Hub counter measures file
requests, not unique users or completed inference runs.

## Train

The current eight-task implementation is in [`multitask/`](multitask/). Follow [the multitask recipe and dataset contract](docs/multitask.md).

- Original base: `Dream-org/Dream-VLA-7B@0cd0e25c1771c5b468c74585b5cf35c4a4aa315c`.
- 100,000 AdamW updates at1e-4; global batch64;4 GPUs × microbatch8 × accumulation2.
- LoRA rank32/alpha16/dropout0; author Flow Matching head/projectors,4 flow steps, action chunk8.
- 8,140 successful training trajectories,30 goal variants,510 training instruction strings across eight families.
- Two224RGB cameras,8D proprioception and the task instruction.

The root `train.py`, `data.py`, `inference.py` and `panda-author.json` preserve the earlier single-task Lift implementation; its historical instructions remain in [docs/training.md](docs/training.md) and [docs/dataset.md](docs/dataset.md). Use **`multitask/train.py`** for the released eight-task model's recipe.

Raw demonstrations and optimizer/RNG training-resume files are not included in this release. Public model files are sufficient for inference with the pinned base and this code; reproducing training requires the documented dataset/cache.

## Evidence and scope

Each family used64 fixed development scenes, original physical configuration and horizon,10 settling steps, `hard_reset=False`, and success held for10 consecutive physical steps. Every recorded policy input, physical action, controller conversion and outcome trace was independently audited. See [evaluation-results.json](evaluation-results.json).

Original pretrained Dream-VLA: **0/64 PickPlace**; adapted Jocasta-diff-VLA: **40/64** on the same scenes. The original uses one image, one action per call, discrete diffusion and Bridge normalization; the adapted policy uses two images/proprio, chunk8, Flow Matching and Panda normalization. This measures the transfer of these complete policies, not a controlled diffusion-versus-autoregression or backbone-only effect.

ToolHang remains weak (9/64). Generalization to other robots, unseen task families, real hardware or the reserved test split has not been established. This release is a manipulation policy; it does not implement autonomous replication or learned mission-memory inheritance.

The core training modules are checked against their frozen training snapshot. Release loading is checked against the original checkpoint on saved inputs from every task family. See [`provenance/`](provenance/) for hashes and verification.

Architecture, loss, sampler and action modules come from Dream-VLX. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution, pinned revisions, the exact patch and the authors' citation.

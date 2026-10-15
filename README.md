# Diffusion-VLA tuning

Dream-VLA-7B fine-tuning on simulated **Panda Lift**, using the Dream-VLX authors' continuous Flow Matching objective. This repository exports the training implementation used for the 100,000-step Panda run on 14–15 October 2026.

The training loop, model bindings, dataset adapter, inference function, checks, and Panda configuration are copied from the working experiment. Upstream dependencies are pinned Git submodules; the three modified upstream files are reproduced by a versioned patch.

## Training recipe

| Setting | Value |
| --- | --- |
| Base checkpoint | `Dream-org/Dream-VLA-7B` |
| Model revision | `0cd0e25c1771c5b468c74585b5cf35c4a4aa315c` |
| Objective | Author's Flow Matching velocity MSE |
| Optimizer | AdamW, learning rate `1e-4` |
| Training | 100,000 optimizer updates, global batch 64 |
| LoRA | Rank 32, alpha 16, dropout 0, all linear layers |
| Inputs | Current external RGB, wrist RGB, proprioception (8D), instruction |
| Actions | 8-action chunks, 7 values per action |
| Inference | 4 Flow Matching steps, execute 8 actions between observations |
| Data | 1,024 training and 64 development trajectories |
| Seed | `84907` |
| Reference training hardware | 8 H100 GPUs, one node, microbatch 8 per GPU |

The current adapter uses one instruction, `pick up the red cube`. It is a single-task Panda transfer. The number of steps follows the paper's 100,000-step setting; the upstream release shell script uses a different step count. This repository retains the Panda run's explicit configuration in [`panda-author.json`](panda-author.json).

## Get the exact source

```bash
git clone --recurse-submodules https://github.com/alanrbtx/Diffusion-VLA-tuning.git
cd Diffusion-VLA-tuning
git -C third_party/Dream-VLX apply --check ../../patches/dream-vlx-panda.patch
git -C third_party/Dream-VLX apply ../../patches/dream-vlx-panda.patch
```

Apply the patch once after initializing the submodules. `upstream-vla` is a relative symlink to `third_party/Dream-VLX/vla`, preserving the imports used during training. Downloading GitHub's ZIP alone does not populate submodules.

The patch preserves the author loss and sampler while adding:

- Explicit Panda action/proprioception dimensions.
- Lazy allocation for the unused discrete CART branch.
- Flow Matching head/projector inclusion in the upstream optimizer setup.
- Hugging Face loading diagnostics through the model wrapper.

Our `train.py` wraps all trainable components in one DDP module, checks optimizer coverage, and saves model, optimizer, per-rank RNG, and sampler state atomically.

## Environment and launch

Use an allocated **Linux CUDA GPU pod**. The executable research entrypoints require Kubernetes and CUDA. The reference model environment is Python **3.10.18**, PyTorch **2.5.1**, Transformers **4.51.3**, TensorFlow **2.15.0**, and NumPy **1.24.0**.

From the repository root inside that pod:

```bash
python3.10 -m venv /tmp/dream-vla-training
source /tmp/dream-vla-training/bin/activate
python -m pip install pip==25.0.1
python -m pip install -r requirements.lock
python -m pip check
```

[`requirements.lock`](requirements.lock) retains the recorded package versions. Its original machine-specific `dlimp` file URL is replaced with `./dlimp-upstream`, pinned to the same commit. Installation through public package indexes has not been re-tested for this publication.

Then follow:

1. [Dataset format and cache preparation](docs/dataset.md).
2. [Training, engineering checks, and checkpoint resume](docs/training.md).

Training consumes saved observations and does not initialize EGL. For later simulation or evaluation, **CUDA and EGL must use different physical GPUs**. The recorded simulator environment was separate: Python 3.12, NumPy 1.26.4, SciPy 1.17.1, MuJoCo 3.3.5, robosuite 1.5.1, `hard_reset=False`.

## Contents

| File | Purpose |
| --- | --- |
| `train.py` | DDP training, optimizer coverage, atomic checkpoints, resume |
| `bindings.py` | Direct bindings to author model, loss, heads, and projectors |
| `data.py` | Panda arrays, author transforms, normalization, stateful sampler |
| `inference.py` | Two-image prompt and 4-step Flow Matching inference |
| `cache_data.py` | Verify recorded episode archives and decode a training cache |
| `check.py` | Author-loss parity, parameter updates, reload, inference preflight |
| `compare_resume.py` | Compare uninterrupted and resumed engineering checkpoints |
| `robot_state.py` | Quaternion-to-axis-angle convention used for Panda state |
| `patches/dream-vlx-panda.patch` | Exact modifications to the pinned author source |

Datasets, base-model weights, trained checkpoints, private cluster orchestration, and experiment logs are not distributed here. This release trains on an existing Panda dataset; it does not include a new demonstration collector or a closed-loop evaluation launcher. `inference.py` exposes the policy function for integration with a simulator.

## Validation and provenance

The preceding GPU run verified author-forward loss parity, updates to all trainable components, checkpoint reload, and a deterministic DDP resume comparison. This publication does not repeat training or evaluation. A completed training loop writes `trained_pending_evaluation`; task success requires a separate closed-loop evaluation.

See [upstream provenance and attribution](THIRD_PARTY.md). Core source hashes from the recorded training snapshot are preserved in [`provenance/training-core.sha256`](provenance/training-core.sha256).

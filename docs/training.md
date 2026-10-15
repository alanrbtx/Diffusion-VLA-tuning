# Run the preserved training recipe

All commands below are intended for an allocated Linux CUDA GPU pod, after source preparation and dependency installation in the root README. No cluster resources are created by these commands.

## Inputs

Set paths for the pinned base model, an existing decoded dataset, and a new durable run directory:

```bash
export MODEL_DIR=/data/models/Dream-VLA-7B
export DATASET_DIR=/tmp/panda-training-cache
export RUN_DIR=/data/runs/panda-author-reproduction
```

If the base model is not already available, download the exact revision inside the GPU pod before enabling offline mode:

```bash
huggingface-cli download Dream-org/Dream-VLA-7B \
  --revision 0cd0e25c1771c5b468c74585b5cf35c4a4aa315c \
  --local-dir "$MODEL_DIR"
```

Keep `panda-author.json` unchanged to reproduce the recorded recipe. Its `original_run` field records the demonstration source; the training entrypoint receives its actual dataset location through `--dataset`.

## Record a bounded execution contract

Set `TRAINING_TIMEOUT_SECONDS` to your allocated training time budget. Run this source/contract preparation inside the GPU pod from the repository root:

```bash
: "${TRAINING_TIMEOUT_SECONDS:?Set the allocated training time budget in seconds}"
export TRAINING_TIMEOUT_SECONDS
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch

assert os.environ.get("KUBERNETES_SERVICE_HOST") and torch.cuda.is_available()
budget = int(os.environ["TRAINING_TIMEOUT_SECONDS"])
assert budget > 0
root = Path.cwd()
run = Path(os.environ["RUN_DIR"])
run.mkdir(parents=True, exist_ok=False)
paths = [root / name for name in (
    "train.py", "bindings.py", "data.py", "inference.py", "panda-author.json",
    "requirements.lock", "patches/dream-vlx-panda.patch",
)]
for directory in ("third_party/Dream-VLX/vla", "dlimp-upstream/dlimp"):
    paths.extend(sorted((root / directory).rglob("*.py")))
pins = {
    "third_party/Dream-VLX": "a72fa4106eae97a923b531342fb692742809782f",
    "dlimp-upstream": "040105d256bd28866cc6620621a3d5f7b6b91b46",
}
for directory, commit in pins.items():
    actual = subprocess.check_output(["git", "-C", directory, "rev-parse", "HEAD"], text=True).strip()
    assert actual == commit
subprocess.run([
    "git", "-C", "third_party/Dream-VLX", "apply", "--reverse", "--check",
    "../../patches/dream-vlx-panda.patch",
], check=True)
contract = {
    "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "upstream_commits": pins,
    "deadline_unix": time.time() + budget,
}
(run / "execution.json").write_text(json.dumps(contract, indent=2))
PY
```

The original driver also monitored GPU temperature, Xid, and ECC. Integrate those checks with the allocated pod's monitor. The training loop honors `$RUN_DIR/GPU_STOP` between optimizer updates; the contract enforces a wall-time deadline and retains a resumable checkpoint when that deadline is reached.

## Train

The reference run used eight H100 GPUs. Make exactly those allocated training devices visible, then run:

```bash
export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export PYTHONHASHSEED=84907

python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train.py \
  --model "$MODEL_DIR" \
  --dataset "$DATASET_DIR" \
  --config panda-author.json \
  --contract "$RUN_DIR/execution.json" \
  --output "$RUN_DIR/training" \
  --microbatch 8
```

For another GPU count, `world_size * microbatch` must divide 64; gradient accumulation preserves global batch 64. Memory requirements and runtime on another allocation need separate verification. Do not change GPU count or microbatch when resuming an existing run, because they are part of its saved identity.

The final checkpoint is `training/checkpoint-100000`. `progress.json` reports the latest optimizer update. `result.json` records `trained_pending_evaluation` on completion. Intermediate checkpoints serve recovery; final 100000 is the fixed quality endpoint.

## Resume

Use the same command, output directory, config, dataset, source, world size, and microbatch, adding:

```text
--resume /data/runs/panda-author-reproduction/training/checkpoint-NNNNNN
```

Choose a complete checkpoint with `READY.json` and `optimizer_available: true`. The last two complete checkpoints retain optimizer state; older 10,000-step milestones retain inference weights and provenance only. A stale deadline needs an explicitly updated time budget; resume does not extend it automatically. Source/config/dataset identity checks remain enforced.

## Optional engineering checks

These are separate diagnostic updates, not part of the 100,000-step run. Their weights must not initialize the main training run.

For `check.py`, save the unmodified upstream training file with:

```bash
git -C third_party/Dream-VLX show \
  a72fa4106eae97a923b531342fb692742809782f:vla/vla-scripts/finetune_dream.py \
  > "$RUN_DIR/original_finetune_dream.py"
```

With exactly one allocated CUDA GPU visible:

```bash
python check.py \
  --model "$MODEL_DIR" --dataset "$DATASET_DIR" --config panda-author.json \
  --output "$RUN_DIR/engineering-parity" \
  --original "$RUN_DIR/original_finetune_dream.py"
```

This checks the preserved author forward function, all optimizer components, 12 engineering updates, fresh reload, and 4-step inference.

For a deterministic resume comparison, invoke `train.py` through `torch.distributed.run` for four uninterrupted updates, two updates, then resume the two-update checkpoint to four. Use `--engineering-updates 4` / `2` / `4`, output directory names containing `engineering`, and `NCCL_ALGO=Ring` for every distributed engineering invocation. Compare the final checkpoints with:

```bash
python compare_resume.py \
  --continuous "$RUN_DIR/engineering-continuous/checkpoint-000004" \
  --resumed "$RUN_DIR/engineering-resumed/checkpoint-000004" \
  --output "$RUN_DIR/resume-verification.json"
```

The full recipe uses its normal fast kernels and default NCCL settings. It restores saved states on resume but does not promise a bitwise trajectory across nondeterministic kernels.

# Eight-task Jocasta-diff-VLA recipe

The `multitask/` directory contains the training/data/prompt/inference modules from the final eight-task run. `multitask/recipe -> ..` resolves the existing pinned model bindings. The root single-task implementation remains available for historical Panda Lift reproduction.

## Exact recipe

- Dream-VLA-7B revision `0cd0e25c1771c5b468c74585b5cf35c4a4aa315c`; same patched Dream-VLX commit as the root repository.
- Fresh LoRA rank32/alpha16/dropout0 and author Flow Matching head/projectors; not initialized from the earlier Lift policy.
- 100,000 AdamW updates, learning rate1e-4, global batch64, training seed84907, sampling seed190019.
- Recorded training:4 GPUs, microbatch8, accumulation2. The fixed final checkpoint is100000.
- 8,140 successful training trajectories,4,841,083 recorded timesteps,30 goal variants across eight families; 510 train instruction strings. The120 held-out language-development strings are excluded by the sampler. The reported evaluations use canonical instructions, not a held-out-language test.
- Balanced family/goal sampling, causal eight-step chunks, original image augmentations, two224RGB views and8D proprioception.

## Dataset contract

Raw demonstration data are not part of this checkpoint/code release. Training requires the existing decoded `paired-rgb-zstd-v1` cache. It contains:

```
manifest.json                 # format, zstandard_version, episodes (8140 rows)
dataset_statistics.json       # panda_multitask_v1 action/proprio quantiles
instruction-bank.json          # balanced sampler's task-specific strings
<family>/<seed>/frames.zst     # concatenated independently compressed frame pairs
<family>/<seed>/offsets.npy    # uint64: T+1 offsets into frames.zst
<family>/<seed>/action.npy     # T x 7 physical actions
<family>/<seed>/proprio.npy    # T x 8 proprioception
```

Each decoded frame pair is224x224x3 uint8 primary followed by wrist. Manifest rows include family, goal_index, goal_id, seed, steps and cache_dir. The lossless cache/normalization and instruction bank must match the saved training identity; this release does not provide a fabricated replacement dataset. `balanced_index.py` and `data.py` define the exact sampling/schema contracts.

## Launch in a CUDA pod

Prepare submodules and the patch as described in the root README. Install `requirements.lock` and `requirements-multitask.txt`. Set MODEL_DIR to the pinned base snapshot, DATASET_DIR to the prepared cache, RUN_DIR to a new durable directory, and TRAINING_TIMEOUT_SECONDS to the allocated budget. Generate the execution contract with:

```bash
python multitask/make_contract.py --output "$RUN_DIR/execution.json" \
  --seconds "$TRAINING_TIMEOUT_SECONDS"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  multitask/train.py --model "$MODEL_DIR" --dataset "$DATASET_DIR" \
  --config multitask/config.json --contract "$RUN_DIR/execution.json" \
  --output "$RUN_DIR/training" --microbatch 8
```

The existing training entrypoint requires Kubernetes and CUDA. Use the same source, data, world size and microbatch for resume and add `--resume <complete-checkpoint-directory>`. Public inference weights omit optimizer/RNG state, so they are an inference release, not an exact training-resume package. Training uses saved observations without EGL; closed-loop simulation requires separate physical CUDA/EGL GPUs.

## Development results

64 fixed scenes per task; sustained success for10 physical steps; original task horizons; hard_reset=False; final checkpoint only; one training seed; no reserved test scenes. Inputs and every physical action were independently audited. See [evaluation-results.json](../evaluation-results.json) and the Hugging Face model card.

| Task | Successes | Rate |
|---|---:|---:|
| Lift |64/64|100.0%|
| PickPlace |40/64|62.5%|
| Stack |43/64|67.2%|
| Door |34/64|53.1%|
| NutAssembly |32/64|50.0%|
| ToolHang |9/64|14.1%|
| Push |61/64|95.3%|
| Drawer |49/64|76.6%|
| Total |332/512|64.8%|

The original pretrained Dream-VLA obtained0/64 on the same PickPlace scenes with its single-image, single-action discrete-diffusion interface and Bridge normalization. The trained policy uses two images/proprio, chunk8, Flow Matching and Panda normalization. This compares ready policies before/after adaptation; it does not isolate a backbone or diffusion-mechanism effect. These are development results on our Panda tasks, not official LIBERO scores or real-robot validation.

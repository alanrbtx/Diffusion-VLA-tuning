# Panda dataset contract

The current release consumes the saved Panda Lift recordings. It does not download a public dataset or collect new trajectories automatically.

## Training cache

`--dataset` points to a directory with this structure:

```text
dataset/
  manifest.json
  dataset_statistics.json
  episodes/
    train-20000/
      image_primary.npy
      image_wrist.npy
      action.npy
      proprio.npy
    ...
```

Each episode uses:

| Array | Shape | Content |
| --- | --- | --- |
| `image_primary` | `(T, 224, 224, 3)` | External-camera RGB, uint8 |
| `image_wrist` | `(T, 224, 224, 3)` | Wrist-camera RGB, uint8 |
| `action` | `(T, 7)` | Physical delta XYZ in metres, XYZ Euler rotation deltas in radians, gripper openness |
| `proprio` | `(T, 8)` | EEF XYZ, axis-angle orientation, two gripper joint positions |

Gripper openness is `1 = open`, `0 = closed`. It is not robosuite's raw gripper command. Each observation precedes the corresponding action. The robot controller used world-frame delta OSC at 20 Hz.

`manifest.json` contains an `episodes` list. Each row required by `Dataset` has `split`, `success`, `seed`, `steps`, and `cache_dir`. For example, the structure of one row is:

```json
{
  "split": "train",
  "success": true,
  "seed": 20000,
  "steps": 150,
  "cache_dir": "episodes/train-20000"
}
```

The step count above is illustrative; use the actual array length. The archived manifest additionally contains archive filenames, SHA-256 digests, and scene provenance. `data.py` selects successful `train` episodes only. Development rows use `split: "validation"`.

The original split is 1,024 train scenes with seeds 20000–21023 and 64 development scenes with seeds 30000–30063. Ten initial settling steps were outside the recordings; no-op removal was performed during physical replay before recording observations.

## Normalization and language

`dataset_statistics.json` is keyed by the configuration's `normalization_key`, currently `panda_author_e7b941`. Under that key, `action` and `proprio` each contain arrays named `q01`, `q99`, `min`, `max`, `mean`, `std`, and `mask`, with one entry per dimension. The action mask is `[true, true, true, true, true, true, false]`; all eight proprioception dimensions are normalized.

Statistics must come from training trajectories only. The adapter clips normalized values to `[-1, 1]` and maps constant dimensions to zero. It samples 8-action windows with replacement and omits the last seven possible start positions, following the author chunking convention.

The current instruction is read from `panda-author.json` for every sample. Per-episode multitask language is not implemented in this snapshot.

## Decode the existing compressed recordings

If the dataset is stored as the recorded per-episode NPZ archives, run the existing decoder inside a GPU pod:

```bash
python cache_data.py \
  --source /data/panda-demonstrations \
  --output /tmp/panda-training-cache \
  --receipt /data/panda-cache-receipt.json
```

The source must include `result.json` with `status: "completed"`, `manifest.json`, `dataset_statistics.json`, and all 1,088 episode archives. Every manifest row must include `file`, `sha256`, and `cache_dir`. The decoder verifies archive hashes and shapes before writing the memory-mapped arrays.

This preserved decoder requires a new output directory, more than 96 GiB of free cache space, Linux `dd` with direct-read support, and storage that supports those reads. Keep recorded archives on durable storage; the decoded cache can be temporary.

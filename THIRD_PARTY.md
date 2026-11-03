# Upstream provenance

The Dream-VLA architecture, loss, sampler, action heads, and projectors originate from the Dream-VLX authors. This repository supplies the Panda adapter and training orchestration around that implementation.

| Component | Source | Pinned revision |
| --- | --- | --- |
| Dream-VLX author code | https://github.com/DreamLM/Dream-VLX | `a72fa4106eae97a923b531342fb692742809782f` |
| dlimp augmentation/data dependency | https://github.com/moojink/dlimp_openvla | `040105d256bd28866cc6620621a3d5f7b6b91b46` |
| Dream-VLA-7B model | https://huggingface.co/Dream-org/Dream-VLA-7B | `0cd0e25c1771c5b468c74585b5cf35c4a4aa315c` |

Both source repositories are Git submodules, preserving their original history and copyright notices. The Panda changes are distributed as `patches/dream-vlx-panda.patch`. Upstream code also credits OpenVLA and OpenVLA-OFT. Dependency and model terms remain those of their respective sources; this repository does not relicense them.

The patch affects only:

- `vla/dreamvla/modeling_dreamvl.py`: retain the optional loading-info tuple.
- `vla/prismatic/vla/constants.py`: recognize the explicit Panda platform.
- `vla/vla-scripts/finetune_dream.py`: avoid eager CART allocation and include Flow Matching components in the upstream optimizer branches.

The author Flow Matching `run_forward_pass` and prediction sampler are retained. In our entrypoint, `bindings.py` directly calls the upstream forward function; `train.py` constructs and verifies its own complete optimizer groups and synchronizes all components through one DDP wrapper.

The local package path in the recorded dependency lock was changed to `./dlimp-upstream` for portability. The training configuration and core Python files retain the recorded experiment's contents. Cluster-specific orchestration and generated artifacts are outside this source release.

## Cite the model authors

```bibtex
@article{ye2025dreamvla,
  title={Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone},
  author={Ye, Jiacheng and Gong, Shansan and Gao, Jiahui and Fan, Junming and Wu, Shuang and Bi, Wei and Bai, Haoli and Shang, Lifeng and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2512.22615},
  year={2025}
}
```

## Eight-task release

`multitask/{train,data,single_task_data,inference,prompt,balanced_index}.py` and its config are byte-checked against the frozen final eight-task training source. `provenance/multitask-core.sha256` records that mapping; the earlier `training-core.sha256` still identifies the preserved single-task root files. `load_policy.py` is the public packaging layer; it performs no training and loads the same LoRA and component tensors. The release validation compares its actions to the original loader on saved observations from all eight families.

Model: https://huggingface.co/RobotLearningLab/Jocasta-diff-VLA . The checkpoint is derived from the pinned Dream-VLA base and requires it for inference; the adapter alone is not a standalone model. Dream-VLA upstream attribution and model terms remain applicable.

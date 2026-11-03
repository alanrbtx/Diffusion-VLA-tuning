"""Load the published Jocasta-diff-VLA inference checkpoint on one CUDA GPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

BASE_MODEL = 'Dream-org/Dream-VLA-7B'
BASE_REVISION = '0cd0e25c1771c5b468c74585b5cf35c4a4aa315c'
MODEL_ID = 'RobotLearningLab/Jocasta-diff-VLA'


def load_policy(checkpoint=MODEL_ID, *, revision=None, base_model_path=None, local_files_only=False):
    """Return (policy, processor, config); no simulator or EGL context is created.

    `checkpoint` can be a downloaded release directory or the Hugging Face ID.
    Supply `revision` to pin a published model commit and `base_model_path` to
    reuse a local copy of the exact upstream base revision.
    """
    import torch
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one CUDA GPU before loading the policy')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    directory=Path(checkpoint)
    if not directory.is_dir():
        directory=Path(snapshot_download(str(checkpoint),revision=revision,local_files_only=local_files_only))
    manifest=json.loads((directory/'release.json').read_text())
    if manifest['base_model']!=BASE_MODEL or manifest['base_revision']!=BASE_REVISION:
        raise ValueError('Unexpected base checkpoint identity')
    for name,entry in manifest['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe release member')
        digest=hashlib.sha256()
        with (directory/path).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
        if digest.hexdigest()!=entry['sha256']:raise ValueError(f'Checksum mismatch: {name}')
    base=Path(base_model_path) if base_model_path else Path(snapshot_download(BASE_MODEL,revision=BASE_REVISION,local_files_only=local_files_only))
    # The public multitask modules retain the exact model-input wrapper used in evaluation.
    sys.path.insert(0,str(Path(__file__).resolve().parent/'multitask'))
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from bindings import Policy, processor_for
    cfg=json.loads((directory/'training_config.json').read_text())
    stats=json.loads((directory/'dataset_statistics.json').read_text())
    policy=Policy(str(base),stats,cfg,directory).eval().requires_grad_(False)
    processor=processor_for(str(base))
    return policy,processor,cfg


def predict_action(policy, processor, config, primary, wrist, proprio, instruction, *, seed=None):
    """Return an (8, 7) physical action block from two uint8 RGB224 images.

    proprio: xyz + quaternion-derived axis-angle + two gripper joint positions.
    Actions: world-frame XYZ metres, extrinsic XYZ Euler radians, gripper openness.
    Convert these physical deltas to the target controller's normalized command
    space; they are not direct robosuite OSC inputs.
    """
    sys.path.insert(0,str(Path(__file__).resolve().parent/'multitask'))
    from inference import predict
    import numpy as np
    return np.asarray(predict(policy,processor,config,primary,wrist,proprio,instruction,seed)['actions'])

"""Freeze the public source identity for a new explicitly budgeted training run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=int,required=True)
a=p.parse_args()
assert os.environ.get('KUBERNETES_SERVICE_HOST') and torch.cuda.is_available()
assert a.seconds>0
root=Path(__file__).resolve().parent.parent
paths=[root/n for n in ('bindings.py','requirements.lock','requirements-multitask.txt','patches/dream-vlx-panda.patch')]
paths.extend(sorted((root/'multitask').glob('*.py')))
paths.append(root/'multitask/config.json')
paths.extend(sorted((root/'third_party/Dream-VLX/vla').rglob('*.py')))
paths.extend(sorted((root/'dlimp-upstream/dlimp').rglob('*.py')))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:
    json.dump({'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'deadline_unix':time.time()+a.seconds},f,indent=2)

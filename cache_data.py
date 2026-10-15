"""Verify persistent replay episodes and decode to pod-local mmap files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch

p=argparse.ArgumentParser()
for key in ('source','output','receipt'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args()
assert os.environ.get('KUBERNETES_SERVICE_HOST') and torch.cuda.is_available()
assert json.loads((a.source/'result.json').read_text())['status']=='completed'
manifest=json.loads((a.source/'manifest.json').read_text())
assert len(manifest['episodes'])==1088
assert not a.output.exists();a.output.mkdir()
assert shutil.disk_usage(a.output).free>96*1024**3

def process(row):
    source=a.source/row['file'];archive=a.output/row['file']
    subprocess.run(['dd','if='+str(source),'of='+str(archive),'bs=1M','iflag=direct','status=none'],check=True,timeout=500)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();assert digest==row['sha256']
    directory=a.output/row['cache_dir'];directory.mkdir(parents=True,exist_ok=False)
    with np.load(archive,allow_pickle=False) as data:
        assert data['action'].shape==(row['steps'],7) and data['proprio'].shape==(row['steps'],8)
        for key in ('image_primary','image_wrist','action','proprio'):np.save(directory/(key+'.npy'),data[key])
    archive.unlink()
    return {'file':row['file'],'sha256':digest,'steps':row['steps']}

with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(process,manifest['episodes']))
for name in ('manifest.json','dataset_statistics.json'):shutil.copyfile(a.source/name,a.output/name)
a.receipt.write_text(json.dumps({'status':'passed','episodes':records,'source':str(a.source),'cache':str(a.output)},indent=2))
print(json.dumps({'status':'passed','episodes':len(records)}),flush=True)

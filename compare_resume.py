"""Exact GPU comparison of four uninterrupted versus two-plus-two DDP updates."""
import argparse
import json
import os
from pathlib import Path
import torch
from safetensors.torch import load_file

p=argparse.ArgumentParser()
for key in ('continuous','resumed','output'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args()
assert os.environ.get('KUBERNETES_SERVICE_HOST') and torch.cuda.is_available()
for path in (a.continuous,a.resumed):
 receipt=json.loads((path/'READY.json').read_text());assert receipt['step']==4 and receipt['optimizer_available']

def compare(x,y):
 if isinstance(x,torch.Tensor):
  assert isinstance(y,torch.Tensor) and x.shape==y.shape and x.dtype==y.dtype
  assert torch.equal(x.cuda(),y.cuda()),'Checkpoint tensor differs after resume'
 elif isinstance(x,dict):
  assert x.keys()==y.keys()
  for key in x:compare(x[key],y[key])
 elif isinstance(x,(list,tuple)):
  assert len(x)==len(y)
  for first,second in zip(x,y):compare(first,second)
 else:assert x==y

compare(load_file(a.continuous/'lora_adapter/adapter_model.safetensors',device='cuda'),
        load_file(a.resumed/'lora_adapter/adapter_model.safetensors',device='cuda'))
for name in ('components.pt','optimizer.pt'):
 compare(torch.load(a.continuous/name,map_location='cuda',weights_only=False),
         torch.load(a.resumed/name,map_location='cuda',weights_only=False))
for path in a.continuous.glob('rank-*.pt'):
 x=torch.load(path,map_location='cpu',weights_only=False)
 y=torch.load(a.resumed/path.name,map_location='cpu',weights_only=False)
 compare(x['dataset'],y['dataset']);compare(x['cuda'],y['cuda']);compare(x['torch'],y['torch'])
for trace in a.continuous.parent.glob('engineering-batches-rank-*.jsonl'):
 assert trace.read_bytes()==(a.resumed.parent/trace.name).read_bytes(),'Engineering inputs or noise RNG differ after resume'
a.output.write_text(json.dumps({'status':'passed','exact_model_optimizer_rng_and_cursor':True,
    'updates':4,'continuous':str(a.continuous),'resumed':str(a.resumed),'engineering_weights_discarded':True},indent=2))
print(a.output.read_text(),flush=True)

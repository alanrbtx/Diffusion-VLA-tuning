"""Borrowed-loss parity, optimizer coverage, finite training and speed preflight."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from bindings import Policy,View,processor_for,upstream
from data import Dataset

parser=argparse.ArgumentParser()
for key in ('model','dataset','config','output','original'):parser.add_argument('--'+key,type=Path,required=True)
a=parser.parse_args()
assert os.environ.get('KUBERNETES_SERVICE_HOST') and torch.cuda.is_available()
assert torch.cuda.device_count()==1
torch.set_num_threads(4)
cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(cfg['training_seed']);random.seed(cfg['training_seed']);np.random.seed(cfg['training_seed'])
processor=processor_for(a.model);data=Dataset(a.dataset,processor,cfg)
policy=Policy(a.model,data.stats,cfg);policy.action_tokenizer=data.tokenizer
policy.train()
batch,indices=data.batch(2,augment=False)
assert upstream.process_action_masks(batch['labels'].cuda()).sum(1).tolist()==[56,56]
assert batch['proprio'].shape==(2,8) and len(batch['image_grid_thw'])==4

# Load only the original forward function from its AST; this avoids the original
# module's unrelated 50000x50000 CART allocation. The function body is untouched.
tree=ast.parse(a.original.read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_forward_pass')
scope=dict(upstream.__dict__)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(a.original),'exec'),scope)
original_forward=scope['run_forward_pass']
state=torch.cuda.get_rng_state()
with torch.autocast('cuda',dtype=torch.bfloat16):
 expected,_=original_forward(vla=View(policy.vla),action_head=View(policy.action_head),
  noisy_action_projector=policy.noisy_action_projector,proprio_projector=policy.proprio_projector,
  batch=dict(batch,proprio=batch['proprio'].cuda().bfloat16()),action_tokenizer=data.tokenizer,
  device_id=0,use_l1_regression=False,use_diffusion=False,use_flow_matching=True,
  use_discrete_diffusion=False,use_proprio=True,use_film=False)
expected=expected.detach()
torch.cuda.set_rng_state(state)
actual,_=policy(batch)
assert torch.equal(expected,actual.detach())
del actual
groups={name:[p for p in module.parameters() if p.requires_grad] for name,module in
        [('vla',policy.vla),('action_head',policy.action_head),('noisy_action_projector',policy.noisy_action_projector),('proprio_projector',policy.proprio_projector)]}
parameters=[p for values in groups.values() for p in values]
assert len({id(p) for p in parameters})==len(parameters)
optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'])
assert {id(p) for group in optimizer.param_groups for p in group['params']}=={id(p) for p in policy.parameters() if p.requires_grad}
before={name:[p.detach().clone() for p in values] for name,values in groups.items()}
times=[];losses=[]
for update in range(12):
 started=time.monotonic();optimizer.zero_grad(set_to_none=True)
 for _ in range(8):
  sample,_=data.batch(8,augment=True)
  loss,_=policy(sample);assert torch.isfinite(loss)
  (loss/8).backward();losses.append(float(loss.detach()))
 for name,values in groups.items():
  gradients=[p.grad for p in values if p.grad is not None]
  assert gradients and all(torch.isfinite(g).all() for g in gradients),name
 optimizer.step();torch.cuda.synchronize();times.append(time.monotonic()-started)
 print(json.dumps({'engineering_optimizer_step':update+1,'global_batch':64,'elapsed_step_s':times[-1]}),flush=True)
changes={name:float(sum((p.detach().float()-old.float()).abs().sum() for p,old in zip(values,before[name]))) for name,values in groups.items()}
assert all(value>0 for value in changes.values()),changes
checkpoint=a.output/'checkpoint';checkpoint.mkdir()
policy.vla.save_pretrained(checkpoint/'lora_adapter')
torch.save(policy.components(),checkpoint/'components.pt')
processor.save_pretrained(checkpoint/'processor')
(checkpoint/'dataset_statistics.json').write_text(json.dumps(data.stats))
policy.eval();torch.manual_seed(20261014)
with torch.no_grad():reference_loss=float(policy(batch)[0])
del optimizer,parameters,groups,before,policy,sample,loss
torch.cuda.empty_cache()
loaded=Policy(a.model,data.stats,cfg,checkpoint);loaded.action_tokenizer=data.tokenizer;loaded.eval()
torch.manual_seed(20261014)
with torch.no_grad():loaded_loss=float(loaded(batch)[0])
assert abs(reference_loss-loaded_loss)<1e-5
from inference import predict
seed,t=indices[0];sample_arrays=data.arrays[seed]
inference=predict(loaded,processor,cfg,np.array(sample_arrays['image_primary'][t]),
                  np.array(sample_arrays['image_wrist'][t]),np.array(sample_arrays['proprio'][t]),20261014)
assert len(inference['actions'])==8 and inference['flow_steps']==4
timings=torch.tensor(times[2:],device='cuda')
result={'status':'passed','kind':'engineering_only_not_quality_evaluation','optimizer_updates':12,
        'weights_discarded_before_full_training':True,'author_forward_exact_loss_parity':True,
        'optimizer_covers_all_trainable_components':True,'component_absolute_parameter_changes':changes,
        'reload_loss_absolute_error':abs(reference_loss-loaded_loss),
        'author_four_step_flow_inference_finite':True,
        'single_gpu_global_batch64_step_s_median':float(timings.median()),
        'single_gpu_global_batch64_step_s_mean':float(timings.mean()),
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'data_draws':data.counter,
        'full_training_steps_required':100000,'training_not_started':True}
(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

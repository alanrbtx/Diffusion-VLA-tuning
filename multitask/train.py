"""100000-step eight-family transfer of the borrowed Dream-VLX Flow Matching loss.

Engineering orchestration only: one DDP wrapper, atomic resumable checkpoints,
fixed global batch, and bounded shutdown. No metric-driven checkpoint selection.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE/'recipe'))
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from bindings import Policy,processor_for
from data import Dataset


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check_rank_agreement(policy):
    """Compare every trainable tensor to rank zero, on CUDA, without hashes."""
    mismatches=torch.zeros((),device='cuda',dtype=torch.int64)
    for parameter in policy.parameters():
        if not parameter.requires_grad:continue
        reference=parameter.detach().clone()
        dist.broadcast(reference,src=0)
        mismatches.add_((reference!=parameter.detach()).any())
    dist.all_reduce(mismatches,op=dist.ReduceOp.SUM)
    assert mismatches.item()==0,'Trainable weights differ across DDP ranks'


def main():
    parser=argparse.ArgumentParser()
    for key in ('model','dataset','config','output','contract'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--engineering-updates',type=int,default=0)
    parser.add_argument('--microbatch',type=int,default=8)
    a=parser.parse_args()
    # Strict numerical determinism is a regression-test instrument. The full
    # recipe keeps the author's normal fast kernels; resume restores state,
    # without claiming a bitwise trajectory across nondeterministic kernels.
    deterministic=bool(a.engineering_updates)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic=deterministic
    assert os.environ.get('KUBERNETES_SERVICE_HOST') and torch.cuda.is_available()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local);torch.set_num_threads(4)
    # Upstream prismatic Overwatch/Accelerate may initialize the group at import.
    if not dist.is_initialized():dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    assert dist.get_backend()=='nccl' and dist.get_rank()==rank and dist.get_world_size()==world
    if deterministic and world>1:
        assert os.environ.get('NCCL_ALGO')=='Ring','Strict DDP resume test requires fixed NCCL reduction order'
    cfg=json.loads(a.config.read_text());contract=json.loads(a.contract.read_text())
    assert cfg['max_steps']==100000 and cfg['global_batch_size']==64
    assert 64%(world*a.microbatch)==0
    accumulation=64//(world*a.microbatch)
    total=a.engineering_updates or cfg['max_steps']
    assert not a.engineering_updates or (a.engineering_updates<=4 and 'engineering' in a.output.name)
    identity={'config_sha256':sha(a.config),'manifest_sha256':sha(a.dataset/'manifest.json'),
              'statistics_sha256':sha(a.dataset/'dataset_statistics.json'),
              'instruction_bank_sha256':sha(a.dataset/'instruction-bank.json'),
              'source_hashes':contract['source_hashes'],'world_size':world,
              'microbatch':a.microbatch,'accumulation':accumulation,'global_batch':64,
              'deterministic_algorithms':deterministic,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
              'nccl_algorithm':os.environ.get('NCCL_ALGO','default')}
    if rank==0:
        a.output.mkdir(parents=True,exist_ok=bool(a.resume))
        if not a.resume:atomic_json(a.output/'identity.json',identity)
    dist.barrier()
    assert json.loads((a.output/'identity.json').read_text())==identity
    if a.resume:
        receipt=json.loads((a.resume/'READY.json').read_text())
        assert receipt['identity']==identity and receipt['optimizer_available']
        assert all(sha(a.resume/name)==digest for name,digest in receipt['hashes'].items())
    random.seed(cfg['training_seed']);np.random.seed(cfg['training_seed']);torch.manual_seed(cfg['training_seed'])
    processor=processor_for(a.model);dataset=Dataset(a.dataset,processor,cfg,rank)
    policy=Policy(a.model,dataset.stats,cfg,a.resume);policy.action_tokenizer=dataset.tokenizer;policy.train()
    groups={name:[p for p in module.parameters() if p.requires_grad] for name,module in
            [('vla',policy.vla),('action_head',policy.action_head),
             ('noisy_action_projector',policy.noisy_action_projector),('proprio_projector',policy.proprio_projector)]}
    parameters=[p for values in groups.values() for p in values]
    assert len({id(p) for p in parameters})==len(parameters)
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'])
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in policy.parameters() if p.requires_grad}
    ddp=DDP(policy,device_ids=[local],find_unused_parameters=True,gradient_as_bucket_view=True)
    start_step=0
    if a.resume:
        # AdamW's non-capturable step counters must stay on CPU. Loading every
        # tensor directly onto CUDA changes its bias-correction arithmetic.
        # load_state_dict moves parameter moments to the correct CUDA device.
        state=torch.load(a.resume/'optimizer.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);start_step=state['step'];del state
        assert all(value['step'].device.type=='cpu' for value in optimizer.state.values())
        local_state=torch.load(a.resume/f'rank-{rank}.pt',map_location='cpu',weights_only=False)
        random.setstate(local_state['python']);np.random.set_state(local_state['numpy'])
        torch.set_rng_state(local_state['torch']);torch.cuda.set_rng_state(local_state['cuda'])
        dataset.load_state_dict(local_state['dataset'])
    else:
        torch.manual_seed(cfg['training_seed']+rank)
    assert start_step<total
    progress=a.output/'progress.jsonl'
    started=time.monotonic();window=time.monotonic();window_losses=[]

    def checkpoint(step):
        target=a.output/f'checkpoint-{step:06d}';temporary=target.with_name(target.name+'.tmp')
        if rank==0:
            assert not target.exists() and not temporary.exists()
            temporary.mkdir()
        dist.barrier()
        torch.save({'python':random.getstate(),'numpy':np.random.get_state(),
                    'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),
                    'dataset':dataset.state_dict()},temporary/f'rank-{rank}.pt')
        if rank==0:
            policy.vla.save_pretrained(temporary/'lora_adapter')
            torch.save(policy.components(),temporary/'components.pt')
            torch.save({'optimizer':optimizer.state_dict(),'step':step},temporary/'optimizer.pt')
            processor.save_pretrained(temporary/'processor')
            (temporary/'dataset_statistics.json').write_text(json.dumps(dataset.stats))
        dist.barrier()
        if rank==0:
            hashes={str(p.relative_to(temporary)):sha(p) for p in temporary.rglob('*') if p.is_file()}
            atomic_json(temporary/'READY.json',{'step':step,'identity':identity,'hashes':hashes,'optimizer_available':True})
            temporary.rename(target)
            atomic_json(a.output/'latest-checkpoint.json',{'step':step,'path':str(target),'ready_sha256':sha(target/'READY.json')})
            # Last two complete recovery points; earlier 10k milestones retain
            # inference weights and provenance, without large optimizer states.
            checkpoints=sorted(a.output.glob('checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9]'))
            for previous in checkpoints[:-2]:
                old_step=int(previous.name.split('-')[-1])
                if old_step%cfg['milestone_every']==0:
                    record=json.loads((previous/'READY.json').read_text())
                    if record['optimizer_available']:
                        (previous/'optimizer.pt').unlink()
                        record['hashes'].pop('optimizer.pt');record['optimizer_available']=False
                        atomic_json(previous/'READY.json',record)
                else:shutil.rmtree(previous)
        dist.barrier()

    for step in range(start_step+1,total+1):
        assert not (a.output.parent/'GPU_STOP').exists(),'GPU_STOP'
        if time.time()>contract['deadline_unix']:
            if step>start_step+1 and not (a.output/f'checkpoint-{step-1:06d}').exists():checkpoint(step-1)
            raise TimeoutError('Frozen wall-time limit reached; resumable checkpoint preserved')
        optimizer.zero_grad(set_to_none=True)
        value=torch.zeros((),device='cuda')
        for micro in range(accumulation):
            batch,indices=dataset.batch(a.microbatch,augment=True)
            if a.engineering_updates:
                fingerprints={key:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                              for key,value in batch.items() if isinstance(value,torch.Tensor)}
                fingerprint={'step':step,'micro':micro,'indices':indices,'tensors':fingerprints,
                             'cuda_rng':hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()}
                with (a.output/f'engineering-batches-rank-{rank}.jsonl').open('a') as stream:
                    stream.write(json.dumps(fingerprint)+'\n')
            with ddp.no_sync() if micro+1<accumulation else nullcontext():
                loss,_=ddp(batch)
                assert torch.isfinite(loss),'Nonfinite Flow Matching loss'
                (loss/accumulation).backward();value.add_(loss.detach()/accumulation)
        for name,values in groups.items():
            gradients=[p.grad for p in values if p.grad is not None]
            assert gradients,name+' has no gradients'
            norms=torch.stack(torch._foreach_norm(gradients))
            assert torch.isfinite(norms).all(),name+' nonfinite gradients'
        optimizer.step()
        dist.all_reduce(value);value.div_(world);window_losses.append(value)
        if step==start_step+1 or step==total:check_rank_agreement(policy)
        if step%10==0 or step==start_step+1 or step==total:
            torch.cuda.synchronize();now=time.monotonic()
            row={'step':step,'max_steps':total,'global_batch':64,'world_size':world,
                 'engineering_only':bool(a.engineering_updates),'loss':float(torch.stack(window_losses).mean()),
                 'mean_step_seconds':(now-window)/len(window_losses),'elapsed_s':now-started,
                 'rank_draws':dataset.counter,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                 'learning_rate':optimizer.param_groups[0]['lr'],'timestamp':time.time()}
            if rank==0:
                with progress.open('a') as stream:stream.write(json.dumps(row)+'\n')
                atomic_json(a.output/'progress.json',row);print(json.dumps(row),flush=True)
            window=now;window_losses=[]
        if step%cfg['checkpoint_every']==0 or step==total:checkpoint(step)
    if rank==0:
        atomic_json(a.output/'result.json',{'status':'engineering_passed' if a.engineering_updates else 'trained_pending_evaluation',
                    'optimizer_steps':total,'global_batch':64,'rank_weight_agreement':True,
                    'all_trainable_components_in_optimizer':True,'identity':identity,
                    'weights_discarded':bool(a.engineering_updates),'elapsed_s':time.monotonic()-started})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

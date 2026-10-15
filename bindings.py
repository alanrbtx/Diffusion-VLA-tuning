"""Thin bindings to the vendored Dream-VLX model, loss and action modules."""
import importlib.util
import os
from pathlib import Path
import sys

os.environ['DIFF_ARA_PLATFORM']='PANDA'
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('WANDB_MODE','disabled')
VENDOR=Path(__file__).resolve().parent/'upstream-vla'
sys.path.insert(0,str(VENDOR))

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor
from dreamvla.configuration_dreamvl import DreamVLAConfig
from dreamvla.modeling_dreamvl import DreamVLAForActionPrediction
from prismatic.models.action_heads import FlowMatchingActionHead
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import DreamRLDSBatchTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import NUM_ACTIONS_CHUNK,ACTION_DIM,PROPRIO_DIM

spec=importlib.util.spec_from_file_location('dream_author_finetune',VENDOR/'vla-scripts/finetune_dream.py')
upstream=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=upstream
spec.loader.exec_module(upstream)
assert (NUM_ACTIONS_CHUNK,ACTION_DIM,PROPRIO_DIM)==(8,7,8)


class View:
    """Satisfy author's .module interface inside one encompassing DDP module."""
    def __init__(self,module):self.module=module
    def __call__(self,*args,**kwargs):return self.module(*args,**kwargs)


class Policy(torch.nn.Module):
    def __init__(self,model_path,stats,cfg,checkpoint=None):
        super().__init__()
        config=DreamVLAConfig.from_pretrained(model_path,n_action_bins=256,norm_stats=stats)
        config._attn_implementation='sdpa'
        config.vision_config._attn_implementation='sdpa'
        config.use_cache=False
        base,loading=DreamVLAForActionPrediction.from_pretrained(model_path,config=config,
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True,output_loading_info=True)
        assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'],loading
        base=base.cuda()
        base.norm_stats=stats
        if checkpoint is None:
            self.vla=get_peft_model(base,LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],
                lora_dropout=cfg['lora_dropout'],target_modules='all-linear',init_lora_weights='gaussian'))
        else:
            self.vla=PeftModel.from_pretrained(base,str(checkpoint/'lora_adapter'),is_trainable=True)
        dim=base.config.hidden_size
        self.action_head=FlowMatchingActionHead(input_dim=dim,hidden_dim=dim,action_dim=7).cuda().bfloat16()
        self.noisy_action_projector=NoisyActionProjector(llm_dim=dim).cuda().bfloat16()
        # Match upstream init_module default: proprio master weights FP32,
        # forward computation BF16 under autocast.
        self.proprio_projector=ProprioProjector(llm_dim=dim,proprio_dim=8).cuda()
        self.action_tokenizer=None
        if checkpoint is not None:
            parts=torch.load(checkpoint/'components.pt',map_location='cuda',weights_only=True)
            for name,state in parts.items():getattr(self,name).load_state_dict(state)

    def forward(self,batch):
        batch=dict(batch)
        batch['proprio']=batch['proprio'].reshape(-1,8).cuda().to(torch.bfloat16)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return upstream.run_forward_pass(vla=View(self.vla),action_head=View(self.action_head),
                noisy_action_projector=self.noisy_action_projector,proprio_projector=self.proprio_projector,
                batch=batch,action_tokenizer=self.action_tokenizer,device_id=torch.cuda.current_device(),
                use_l1_regression=False,use_diffusion=False,use_flow_matching=True,
                use_discrete_diffusion=False,use_proprio=True,use_film=False)

    def components(self):
        return {name:getattr(self,name).state_dict() for name in
                ('action_head','noisy_action_projector','proprio_projector')}


def processor_for(model_path):
    processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
    processor.image_processor.size['min_pixels']=64*28*28
    processor.image_processor.min_pixels=64*28*28
    processor.tokenizer.add_tokens([f'<|action_{n}|>' for n in range(256)])
    assert len(processor.tokenizer)-257==151666
    return processor

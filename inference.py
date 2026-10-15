"""Author's two-image prompt and Flow Matching sampler, with Panda state stats."""
import hashlib
import numpy as np
from PIL import Image
import torch
from bindings import upstream
from data import normalize
from experiments.robot.openvla_utils import center_crop_image


def predict(policy,processor,cfg,primary,wrist,proprio,seed=None):
    assert primary.shape==wrist.shape==(224,224,3)
    assert primary.dtype==wrist.dtype==np.uint8
    images=[center_crop_image(Image.fromarray(frame).convert('RGB')) for frame in (primary,wrist)]
    conversation=[{'role':'user','content':[{'type':'image'},{'type':'image'},
                  {'type':'text','text':f"What action should the robot take to {cfg['instruction']}?"}]}]
    text=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=images,padding=True,return_tensors='pt').to('cuda')
    inputs['pixel_values']=inputs['pixel_values'].bfloat16()
    model=policy.vla.get_base_model()
    state=torch.as_tensor(proprio,device='cuda',dtype=torch.float32).reshape(1,8)
    state=normalize(state,model.norm_stats[cfg['normalization_key']]['proprio']).bfloat16()
    if seed is not None:torch.manual_seed(seed)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        action=model.predict_action(**inputs,unnorm_key=cfg['normalization_key'],
            action_head=policy.action_head,noisy_action_projector=policy.noisy_action_projector,
            proprio=state,proprio_projector=policy.proprio_projector,
            action_chunk=cfg['action_chunk'],steps=cfg['eval_flow_steps'],use_flow_matching=True,
            use_discrete_diffusion=False,use_cache=False)
    assert action.shape==(8,7) and torch.isfinite(torch.as_tensor(action,device='cuda')).all()
    digest=hashlib.sha256(primary.tobytes()+wrist.tobytes()+np.asarray(proprio,dtype='<f4').tobytes()).hexdigest()
    return {'actions':action.tolist(),'input_sha256':digest,'input_tokens':int(inputs['input_ids'].shape[1]),
            'image_grid_thw':inputs['image_grid_thw'].tolist(),'flow_steps':cfg['eval_flow_steps']}

"""Panda dataset adapter using the author's transform, collator and augmentation.

Frame decode/augmentation follows the upstream CPU input pipeline inside the GPU
pod. Action/proprio normalization and all scientific checks execute on CUDA.
"""
import json
from pathlib import Path
import random
import numpy as np
import torch
import tensorflow as tf
import dlimp as dl
from bindings import ActionTokenizer,DreamRLDSBatchTransform,PaddedCollatorForActionPrediction

AUG=dict(random_resized_crop=dict(scale=[.9,.9],ratio=[1.,1.]),random_brightness=[.2],
         random_contrast=[.8,1.2],random_saturation=[.8,1.2],random_hue=[.05],
         augment_order=['random_resized_crop','random_brightness','random_contrast','random_saturation','random_hue'])


def normalize(x,stats):
    assert x.is_cuda
    low=torch.tensor(stats['q01'],device=x.device,dtype=x.dtype)
    high=torch.tensor(stats['q99'],device=x.device,dtype=x.dtype)
    mask=torch.tensor(stats.get('mask',[True]*x.shape[-1]),device=x.device)
    constant=torch.tensor(stats['min'],device=x.device)==torch.tensor(stats['max'],device=x.device)
    value=torch.where(mask,(2*(x-low)/(high-low+1e-8)-1).clamp(-1,1),x)
    return torch.where(constant,torch.zeros_like(value),value)


class Dataset:
    def __init__(self,root,processor,cfg,rank=0):
        self.root=Path(root);self.cfg=cfg;self.rank=rank;self.counter=0
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.stats=json.loads((self.root/'dataset_statistics.json').read_text())
        self.rows=[r for r in self.manifest['episodes'] if r['split']=='train' and r['success']]
        self.indices=[];self.arrays={}
        for row in self.rows:
            directory=self.root/row['cache_dir']
            self.arrays[row['seed']]={key:np.load(directory/(key+'.npy'),mmap_mode='r')
                for key in ('image_primary','image_wrist','action','proprio')}
            # Exact author's chunk_act_obs: omit last 7 starts, no synthetic
            # repetition/padding of final commands.
            self.indices.extend((row['seed'],t) for t in range(row['steps']-7))
        assert self.indices
        self.rng=random.Random(cfg['training_seed']+rank*1000003)
        self.tokenizer=ActionTokenizer(processor.tokenizer)
        self.transform=DreamRLDSBatchTransform(self.tokenizer,processor,use_wrist_image=True,use_proprio=True)
        self.collator=PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                                       processor.tokenizer.pad_token_id,padding_side='right')

    def state_dict(self):return {'rng':self.rng.getstate(),'counter':self.counter,'rank':self.rank}
    def load_state_dict(self,state):
        assert state['rank']==self.rank
        self.rng.setstate(state['rng']);self.counter=state['counter']

    def batch(self,size,augment=True,indices=None):
        selected=indices if indices is not None else self.rng.choices(self.indices,k=size)
        samples=[]
        stat=self.stats[self.cfg['normalization_key']]
        for seed,t in selected:
            arrays=self.arrays[seed]
            images=[]
            for camera,key in enumerate(('image_primary','image_wrist')):
                frame=np.array(arrays[key][t],copy=True)
                if augment:
                    frame=dl.transforms.augment_image(tf.convert_to_tensor(frame),
                        seed=tf.constant([self.cfg['training_seed']+self.rank,self.counter*2+camera],tf.int32),**AUG).numpy()
                images.append(frame)
            action=torch.tensor(np.array(arrays['action'][t:t+8],copy=True),device='cuda')
            proprio=torch.tensor(np.array(arrays['proprio'][t:t+1],copy=True),device='cuda')
            action=normalize(action,stat['action']).cpu().numpy()
            proprio=normalize(proprio,stat['proprio']).cpu().numpy()
            assert action.shape==(8,7) and proprio.shape==(1,8)
            sample={'dataset_name':self.cfg['normalization_key'],'action':action,
                    'observation':{'image_primary':images[0][None],'image_wrist':images[1][None],
                                   'proprio':proprio},
                    'task':{'language_instruction':self.cfg['instruction'].encode()}}
            samples.append(self.transform(sample));self.counter+=1
        batch=self.collator(samples)
        batch['proprio']=batch['proprio'].reshape(len(selected),8)
        assert batch['image_grid_thw'].shape==(len(selected)*2,3)
        return batch,selected

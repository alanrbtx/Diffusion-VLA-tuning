"""Balanced multitask adapter; retain upstream image augmentation and action labels."""
from collections import OrderedDict
import json
import os
from pathlib import Path
import zstandard as zstd
import numpy as np
from PIL import Image
import torch
import tensorflow as tf
import dlimp as dl
from bindings import ActionTokenizer,PaddedCollatorForActionPrediction
from single_task_data import AUG,normalize
from balanced_index import BalancedIndex
from prompt import instruction_prompt

class Transform:
    """DreamRLDSBatchTransform with only the language wrapper generalized."""
    def __init__(self,tokenizer,processor):
        self.action_tokenizer=tokenizer;self.processor=processor

    def __call__(self,sample):
        action=sample['action'].reshape(-1)
        imgs=[Image.fromarray(sample['observation'][k][0]) for k in ('image_primary','image_wrist')]
        conversation=[
            {'role':'user','content':[{'type':'image'}]*2+[{'type':'text','text':instruction_prompt(sample['task']['language_instruction'].decode())}]},
            {'role':'assistant','content':[{'type':'text','text':self.action_tokenizer(action)}]},
        ]
        text=self.processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=False)
        inputs=self.processor(text=[text],images=imgs,padding=True,return_tensors='pt')
        ids=inputs['input_ids'][0];labels=ids.clone();labels[:-(len(action)+2)]=-100
        # Check the exact author convention: 56 action tokens followed by two
        # chat terminators. Longer language must never shift the action labels.
        action_mask=labels>self.action_tokenizer.action_token_begin_idx
        assert int(action_mask.sum())==56 and int((labels!=-100).sum())==58
        assert ids.numel()<=self.processor.tokenizer.model_max_length
        expected=self.processor.tokenizer(self.action_tokenizer(action),add_special_tokens=False)['input_ids']
        assert ids[action_mask].tolist()==expected
        return {'pixel_values':inputs['pixel_values'],'image_grid_thw':inputs['image_grid_thw'],
                'input_ids':ids,'labels':labels,'dataset_name':sample['dataset_name'],
                'actions':sample['action'],'proprio':sample['observation']['proprio']}

class Dataset:
    def __init__(self,root,processor,cfg,rank=0):
        self.root=Path(root);self.cfg=cfg;self.rank=rank;self.world=int(os.environ.get('WORLD_SIZE','1'));self.counter=0
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.stats=json.loads((self.root/'dataset_statistics.json').read_text())
        self.bank=json.loads((self.root/'instruction-bank.json').read_text())
        self.rows=self.manifest['episodes']
        if self.manifest.get('engineering_probe'):
            assert cfg.get('_engineering_probe') and len(self.rows)==30
        else:assert len(self.rows)==8140
        assert self.manifest['format']=='paired-rgb-zstd-v1' and zstd.__version__==self.manifest['zstandard_version']
        self.index=BalancedIndex(self.rows,self.bank,seed=cfg['sampling_seed'])
        self.tokenizer=ActionTokenizer(processor.tokenizer);self.transform=Transform(self.tokenizer,processor)
        self.collator=PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,processor.tokenizer.pad_token_id,padding_side='right')
        self.opened=OrderedDict()
        self.decompressor=zstd.ZstdDecompressor()

    def state_dict(self):
        return {'counter':self.counter,'rank':self.rank,'world':self.world,'sampling_seed':self.cfg['sampling_seed']}

    def load_state_dict(self,state):
        assert state['rank']==self.rank and state['world']==self.world and state['sampling_seed']==self.cfg['sampling_seed']
        self.counter=state['counter']

    def episode(self,row):
        key=(row['family'],row['seed'])
        if key in self.opened:
            entry=self.opened.pop(key);self.opened[key]=entry;return entry
        path=self.root/row['cache_dir']
        entry={'fd':os.open(path/'frames.zst',os.O_RDONLY),
               'offsets':np.load(path/'offsets.npy',allow_pickle=False),
               'action':np.load(path/'action.npy',allow_pickle=False),
               'proprio':np.load(path/'proprio.npy',allow_pickle=False)}
        assert entry['action'].shape==(row['steps'],7) and entry['proprio'].shape==(row['steps'],8)
        self.opened[key]=entry
        if len(self.opened)>128:
            _,old=self.opened.popitem(last=False);os.close(old['fd'])
        return entry

    def batch(self,size,augment=True,indices=None):
        assert indices is None,'Use the frozen stateless global sample ids'
        samples=[];selected=[];stat=self.stats[self.cfg['normalization_key']]
        for _ in range(size):
            sample_id=self.counter*self.world+self.rank
            selected_row=self.index.sample(sample_id);row=selected_row['episode'];t=selected_row['start']
            arrays=self.episode(row);start,end=map(int,arrays['offsets'][t:t+2])
            raw=self.decompressor.decompress(os.pread(arrays['fd'],end-start,start))
            # This cluster charges file pages to the pod without dependable
            # reclaim. Keep the decoded bytes, release the read cache range.
            page=os.sysconf('SC_PAGE_SIZE')
            lower=start//page*page;upper=(end+page-1)//page*page
            os.posix_fadvise(arrays['fd'],lower,upper-lower,os.POSIX_FADV_DONTNEED)
            assert len(raw)==2*224*224*3
            pair=np.frombuffer(raw,dtype=np.uint8).reshape(2,224,224,3)
            images=[]
            for camera in range(2):
                frame=np.array(pair[camera],copy=True)
                if augment:
                    frame=dl.transforms.augment_image(tf.convert_to_tensor(frame),
                        seed=tf.constant([self.cfg['training_seed'],sample_id*2+camera],tf.int32),**AUG).numpy()
                images.append(frame)
            action=normalize(torch.tensor(arrays['action'][t:t+8],device='cuda'),stat['action']).cpu().numpy()
            proprio=normalize(torch.tensor(arrays['proprio'][t:t+1],device='cuda'),stat['proprio']).cpu().numpy()
            assert action.shape==(8,7) and proprio.shape==(1,8)
            item={'dataset_name':self.cfg['normalization_key'],'action':action,
                  'observation':{'image_primary':images[0][None],'image_wrist':images[1][None],'proprio':proprio},
                  'task':{'language_instruction':selected_row['instruction'].encode()}}
            samples.append(self.transform(item));selected.append((sample_id,row['family'],row['goal_index'],row['seed'],t))
            self.counter+=1
        batch=self.collator(samples);batch['proprio']=batch['proprio'].reshape(size,8)
        assert batch['image_grid_thw'].shape==(size*2,3)
        assert ((batch['labels']>self.tokenizer.action_token_begin_idx).sum(dim=1)==56).all()
        return batch,selected

    def close(self):
        for entry in self.opened.values():os.close(entry['fd'])
        self.opened.clear()

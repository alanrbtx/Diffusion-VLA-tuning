"""Stateless balanced chunk selection; sample ids are checkpointable across ranks."""
from collections import defaultdict
import hashlib
import random

class BalancedIndex:
    def __init__(self,rows,instructions,seed=190019):
        self.seed=seed;self.buckets=defaultdict(lambda:defaultdict(list))
        for row in rows:
            assert row['split']=='train' and row['success'] and row['steps']>=8
            self.buckets[row['family']][row['goal_index']].append(row)
        self.families=sorted(self.buckets)
        assert len(self.families)==8
        self.instructions={g['goal_id']:g for g in instructions['goals']}
        for family in self.families:
            for goal,items in self.buckets[family].items():
                items.sort(key=lambda r:r['seed'])
                assert f'{family}:{goal}' in self.instructions

    def rng(self,*parts):
        digest=hashlib.sha256((':'.join(map(str,(self.seed,)+parts))).encode()).digest()
        return random.Random(int.from_bytes(digest[:16],'big'))

    def sample(self,sample_id):
        assert isinstance(sample_id,int) and sample_id>=0
        cycle,slot=divmod(sample_id,len(self.families))
        families=self.families.copy();self.rng('family',cycle).shuffle(families)
        family=families[slot];goals=sorted(self.buckets[family])
        goal_cycle,goal_slot=divmod(cycle,len(goals))
        self.rng('goal',family,goal_cycle).shuffle(goals)
        goal=goals[goal_slot];rng=self.rng('sample',sample_id)
        row=rng.choice(self.buckets[family][goal]);start=rng.randrange(row['steps']-7)
        bank=self.instructions[row['goal_id']]
        instruction=rng.choice(bank['train'])
        assert instruction not in bank['heldout_language_development']
        return {'sample_id':sample_id,'episode':row,'start':start,'end':start+8,'instruction':instruction}

import torch
from torch.utils.data import Sampler

class CyclingBudgetSampler(Sampler):
    """Fixed global draws per virtual epoch, without replacement until the full pool is exhausted."""
    def __init__(self, size, draws, rank, world_size, seed):
        assert size>0 and draws>0 and draws%world_size==0 and 0<=rank<world_size
        self.size,self.draws,self.rank,self.world_size,self.seed=size,draws,rank,world_size,seed
        self.cursor=0
        self.cycle=-1
        self.order=None

    def __len__(self):
        return self.draws//self.world_size

    def __iter__(self):
        chunks=[]
        remaining=self.draws
        while remaining:
            cycle,offset=divmod(self.cursor,self.size)
            if cycle!=self.cycle:
                self.order=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+cycle))
                self.cycle=cycle
            take=min(remaining,self.size-offset)
            chunks.append(self.order[offset:offset+take])
            self.cursor+=take
            remaining-=take
        return iter(torch.cat(chunks)[self.rank::self.world_size].tolist())

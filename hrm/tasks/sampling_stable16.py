import torch
from torch.utils.data import Sampler


class InfiniteSampler(Sampler[int]):
    """Shard shuffled passes evenly, dropping the tail before each new pass."""

    def __init__(self, size: int, rank: int = 0, world_size: int = 1, seed: int = 42):
        if size < 1:
            raise ValueError("Training dataset must contain at least one row")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be between 0 and world_size - 1")
        if size < world_size:
            raise ValueError("Training dataset must contain at least world_size rows")
        self.size = size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        usable_size = self.size - self.size % self.world_size
        while True:
            indices = torch.randperm(self.size, generator=generator)[:usable_size]
            yield from indices[self.rank::self.world_size].tolist()

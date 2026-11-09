"""Save exact optimizer/RNG/data-loader position without changing sample order."""
import copy
import numpy as np
import torch


class ReplayableLoader:
    def __init__(self, loader):
        self.loader = loader
        self._new_epoch()

    def _new_epoch(self):
        self.epoch_start_rng = self.loader.generator.get_state().clone()
        self.iterator = iter(self.loader)
        self.batches = 0

    def next(self):
        try:
            result = next(self.iterator)
        except StopIteration:
            self._new_epoch()
            result = next(self.iterator)
        self.batches += 1
        return result

    def state_dict(self):
        return dict(epoch_start_rng=self.epoch_start_rng.clone(), batches=self.batches,
                    current_loader_rng=self.loader.generator.get_state().clone())

    def load_state_dict(self, state):
        self.loader.generator.set_state(state['epoch_start_rng'].cpu())
        self._new_epoch()
        # Replay only dataset reads on CPU. All augmentation/global RNG states
        # are restored afterwards; no model/optimizer work is repeated.
        for _ in range(state['batches']):
            next(self.iterator)
            self.batches += 1
        assert torch.equal(self.loader.generator.get_state(), state['current_loader_rng'].cpu())


def capture(optimizer, corruption_rng, stream, dataset):
    return dict(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy_rng=np.random.get_state(), corruption_rng=corruption_rng.get_state(),
                stream=stream.state_dict(), augmentation=dataset.augmentation_state_dict()
                if hasattr(dataset, 'augmentation_state_dict') else None)


def restore(state, optimizer, corruption_rng, stream, dataset):
    optimizer.load_state_dict(state['optimizer'])
    stream.load_state_dict(state['stream'])
    if state['augmentation'] is not None:
        dataset.load_augmentation_state_dict(state['augmentation'])
    corruption_rng.set_state(state['corruption_rng'].cpu())
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])

"""Epoch-boundary state for the single-worker, single-GPU Sudoku trainer."""
from pathlib import Path
import random
import numpy as np
import torch


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


class StatefulCollate:
    def __init__(self, base):
        self.base = base

    def __call__(self, batch):
        x, y = self.base(batch)
        return x, y, rng_state()


class RestoreWorker:
    def __init__(self, base, state):
        self.base, self.state = base, state

    def __call__(self, worker_id):
        self.base(worker_id)
        if self.state is not None:
            restore_rng(self.state)


def prepare_loader(loader, seed, saved=None):
    assert loader.num_workers == 1 and loader.persistent_workers
    loader.collate_fn = StatefulCollate(loader.collate_fn)
    loader.worker_init_fn = RestoreWorker(loader.worker_init_fn, None if saved is None else saved['worker_rng'])
    loader.generator = torch.Generator().manual_seed(seed)
    if saved is not None:
        loader.generator.set_state(saved['generator'])
        loader.sampler.set_epoch(saved['sampler_epoch'])


def save_checkpoint(path, model, optimizer, epoch, step, budget, history,
                    config, loader, worker_rng, test_loader, stopped):
    state = {'format_version': 1, 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'epoch': epoch, 'step': step,
             'budget': budget, 'eval_history': history, 'config': config,
             'rng': rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
             'loader': {'worker_rng': worker_rng, 'generator': loader.generator.get_state(),
                        'sampler_epoch': loader.sampler.epoch, 'position': 'epoch_boundary'},
             'test_generator': test_loader.generator.get_state(), 'stopped': stopped}
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training(state, model, optimizer, config):
    assert state['format_version'] == 1
    # The user-authorized trend threshold may change; training hyperparameters must match.
    ignored = {'resume_training_state', 'log_interval', 'extension_min_gain'}
    assert {k:v for k,v in state['config'].items() if k not in ignored} == {k:v for k,v in config.items() if k not in ignored}, 'Training configuration changed on resume'
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    # This optimizer uses CPU scalar steps and learning rates.
    for item in optimizer.state.values():
        item['step'] = item['step'].cpu()
    for group in optimizer.param_groups:
        group['lr'] = group['lr'].cpu()
    restore_rng(state['rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])

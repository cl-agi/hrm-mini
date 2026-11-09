from hrm.training.trainer import _exclude_from_code_artifact
from hrm.training.trainer import ArchConfig, DataConfig, load_module, update_lr
from hrm.training.steps import train_step, run_inference
from typing import Any, Optional
import os
import importlib
import math
import json
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import pydantic
import hydra
import tqdm
import wandb
import coolname
from hydra.core.hydra_config import HydraConfig
from hrm.training.optimizer import AdamATan2
from hrm.models.layers import Carry
from hrm.training.outer_schedule import outer_for_epoch

class TrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    compile_model: bool = True
    output_dir: str
    run_name: str = "stochastic"
    arch: ArchConfig
    data: DataConfig
    seeds: list[int] = [42]
    eval_dataset_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    act_patience: int = 0
    pilot_checkpoint: Optional[str] = None
    cycles_per_data: int
    epochs: int
    eval_cycles: int = 128
    outer_symmetry: bool = False
    outer_schedule: list[int] = []
    local_batch_size: int
    lr: float
    lr_warmup_steps: int
    lr_min_ratio: float
    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float] = None
    log_interval: int = 5

def train_single_seed(config: TrainConfig, seed: int, group_name: str, WORLD_SIZE: int, RANK: int):
    """Run a full training run for a single seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    create_dataloader = load_module(f'dataset.{config.data.name}@create_dataloader')
    data_kwargs = dict(config.data.__pydantic_extra__ or {})
    train_loader, train_metadata = create_dataloader('train', config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **data_kwargs)
    eval_data_kwargs = dict(data_kwargs)
    if config.eval_dataset_name is not None:
        eval_data_kwargs['dataset_name'] = config.eval_dataset_name
    eval_loaders = {split_name: create_dataloader(split_name, config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **eval_data_kwargs)[0] for split_name in ['test_hard']}
    total_steps = int(sum((outer_for_epoch(config, e) for e in range(config.epochs))) * len(train_loader))
    from hrm.registry import model_class
    model_cls = model_class(f'arch.{config.arch.name}', config.arch.__pydantic_extra__ or {})
    with torch.device('cuda'):
        model: nn.Module = model_cls(config.arch.__pydantic_extra__ | train_metadata)
        if config.compile_model:
            model = torch.compile(model, dynamic=False, fullgraph=True)
        model = DDP(model, static_graph=True)
    optim = AdamATan2(model.parameters(), lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device='cpu'), betas=(config.beta1, config.beta2), weight_decay=config.weight_decay, ema=config.ema)
    step = 0
    start_epoch = 0
    if config.resume_from_checkpoint is not None:
        checkpoint = torch.load(config.resume_from_checkpoint, map_location=torch.device('cuda', torch.cuda.current_device()), weights_only=True)
        saved_total_steps = int(checkpoint['total_steps'])
        if saved_total_steps != total_steps:
            raise ValueError(f'Resume total_steps mismatch: checkpoint={saved_total_steps}, current={total_steps}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optim.state.values():
            state['step'] = state['step'].cpu()
        step = int(checkpoint['step'])
        start_epoch = int(checkpoint['epoch']) + 1
        if RANK == 0:
            print(f'RESUME_RESULT checkpoint={config.resume_from_checkpoint} start_epoch={start_epoch} step={step}', flush=True)
    if config.resume_from_checkpoint is None and config.pilot_checkpoint is None:
        assert step == 0 and all((int(state['step']) == 0 for state in optim.state.values()))
        if RANK == 0:
            print(json.dumps(dict(event='fresh_start', seed=seed, step=0, world_size=WORLD_SIZE, optimizer_steps_zero=True, resume_checkpoint=None)), flush=True)
    pilot_admitted = 0
    if config.pilot_checkpoint is not None:
        if config.resume_from_checkpoint is not None:
            raise ValueError('Choose native resume or pilot migration, not both')
        pilot = torch.load(config.pilot_checkpoint, map_location='cpu', weights_only=False)
        model.module._orig_mod.load_state_dict(pilot['model'], strict=True)
        optim.load_state_dict(pilot['optimizer'])
        for state in optim.state.values():
            state['step'] = state['step'].cpu()
        for group in optim.param_groups:
            group['lr'] = group['lr'].cpu()
        step = int(pilot['step'])
        assert all((int(state['step']) == step for state in optim.state.values()))
        pilot_admitted = int(pilot['runtime']['source']['admitted'])
        native_epoch_size = len(train_loader) * config.local_batch_size * WORLD_SIZE
        start_epoch = pilot_admitted // native_epoch_size
        if RANK == 0:
            for actual, expected in zip(optim.state.values(), pilot['optimizer']['state'].values()):
                for key in ('exp_avg', 'exp_avg_sq', 'param_ema'):
                    torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=0, atol=0)
            print(json.dumps(dict(event='pilot_migration', step=step, admitted=pilot_admitted, optimizer_exact=True, world_size=WORLD_SIZE, reset_inflight=int(pilot['runtime']['pool']['active'].sum()))), flush=True)
        del pilot
    if config.act_patience:
        torch.manual_seed(seed + RANK)
    run_name = f'{group_name}/seed_{seed}'
    checkpoint_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'model_config.json'), 'w') as f:
        json.dump(config.model_dump(mode="json"), f, indent=2)
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=total_steps, initial=step, desc=f'seed={seed}')
        wandb.init(project=config.data.name, name=run_name, group=group_name, config=config.model_dump() | {'seed': seed}, settings=wandb.Settings(x_disable_stats=True))
        if wandb.run is not None and os.environ.get('WANDB_DISABLE_CODE', '').lower() not in {'true', '1'}:
            wandb.run.log_code(exclude_fn=_exclude_from_code_artifact)
    act_pool = None
    if config.act_patience:
        from hrm.training.act_training import ACTPool
        act_pool = ACTPool(model, config.local_batch_size, train_metadata['seq_len'])
        if config.resume_from_checkpoint is not None and 'act_state_files' in checkpoint:
            runtime = torch.load(checkpoint['act_state_files'][RANK], map_location='cpu', weights_only=True)
            act_pool.load_state_dict(runtime['pool'])
            torch.set_rng_state(runtime['rng_cpu'])
            torch.cuda.set_rng_state(runtime['rng_cuda'])
            if RANK == 0:
                print('ACT_RESUME restored live pool, latent states, ages and streaks', flush=True)
    symmetry_rng = None
    if config.outer_symmetry:
        assert config.resume_from_checkpoint is None, 'Outer symmetry RNG resume is not supported'
        from hrm.training.outer_symmetry import augment_outer
        symmetry_rng = torch.Generator(device=torch.device('cuda', torch.cuda.current_device()))
        symmetry_rng.manual_seed(20261017 + seed * 1009 + RANK)
    for epoch in range(start_epoch, config.epochs):
        train_outer = outer_for_epoch(config, epoch)
        if RANK == 0:
            print(json.dumps(dict(event='outer_schedule', epoch=epoch, train_outer=train_outer)), flush=True)
        model.train()
        if config.act_patience:
            from hrm.training.act_training import train_act_epoch
            epoch_size = len(train_loader) * config.local_batch_size * WORLD_SIZE
            skipped = max(0, pilot_admitted - epoch * epoch_size)
            local_skip = skipped // WORLD_SIZE + int(RANK < skipped % WORLD_SIZE)
            step = train_act_epoch(config, model, optim, train_loader, epoch, step, total_steps, RANK, WORLD_SIZE, train_step, update_lr, checkpoint_dir, pool=act_pool, local_skip=local_skip, symmetry_rng=symmetry_rng)
        else:
            for x, y in train_loader:
                x = x.cuda()
                y = y.cuda()
                metrics = {}
                lr = None
                carry: Carry = model.module.initial_carry
                for outer_index in range(train_outer):
                    step += 1
                    lr = update_lr(config, optim, step, total_steps)
                    carry, metrics = train_step(model, carry, optim, x, y)
                    metrics.pop('_correct', None)
                    if config.outer_symmetry and outer_index + 1 < train_outer:
                        x, y, carry = augment_outer(x, y, carry, symmetry_rng)
                if RANK == 0 and progress_bar is not None and (step - progress_bar.n >= config.log_interval):
                    progress_bar.update(step - progress_bar.n)
                    wandb.log({f'train/{k}': v.item() for k, v in metrics.items()} | {'train/lr': lr}, step=step)
                del x, y, carry, metrics
        act_state_files = None
        if config.act_patience:
            training_rng_cpu = torch.get_rng_state()
            training_rng_cuda = torch.cuda.get_rng_state()
            act_state_files = [os.path.abspath(os.path.join(checkpoint_dir, f'act_{step}_rank{r}.pt')) for r in range(WORLD_SIZE)]
            state_path = act_state_files[RANK]
            torch.save(dict(pool=act_pool.state_dict(), rng_cpu=training_rng_cpu, rng_cuda=training_rng_cuda), state_path + '.tmp')
            os.replace(state_path + '.tmp', state_path)
            dist.barrier()
        if RANK == 0:
            training_checkpoint_path = os.path.join(checkpoint_dir, 'latest_train.pt')
            temporary_checkpoint_path = training_checkpoint_path + '.tmp'
            torch.save({'format_version': 2, **({'act_state_files': act_state_files} if act_state_files is not None else {}), 'epoch': epoch, 'step': step, 'total_steps': total_steps, 'seed': seed, 'world_size': WORLD_SIZE, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optim.state_dict()}, temporary_checkpoint_path)
            os.replace(temporary_checkpoint_path, training_checkpoint_path)
        if WORLD_SIZE > 1:
            dist.barrier()
        model.eval()
        optim.swap_ema()
        if RANK == 0:
            torch.save({k.replace('_orig_mod.', ''): v for k, v in model.module.state_dict().items()}, os.path.join(checkpoint_dir, f'epoch_{epoch}.pt'))
        for eval_name, eval_loader in eval_loaders.items():
            num_total_correct = torch.zeros(2, dtype=torch.long, device='cuda')
            for x, y in eval_loader:
                carry: Carry = model.module.initial_carry
                y_hat = None
                for _ in range(config.eval_cycles):
                    carry, y_hat = run_inference(model, carry, x.cuda())
                num_total_correct[0] += torch.all(y_hat == y.cuda(), dim=-1).sum()
                num_total_correct[1] += y.shape[0]
                del carry, y_hat
            dist.reduce(num_total_correct, dst=0)
            num_total_correct = num_total_correct.cpu().tolist()
            if RANK == 0:
                wandb.log({f'eval/{eval_name}_exact_match': num_total_correct[0] / num_total_correct[1]}, step=step)
                print(f'EVAL_RESULT run={run_name} epoch={epoch} split={eval_name} correct={num_total_correct[0]} total={num_total_correct[1]} exact_match={num_total_correct[0] / num_total_correct[1]:.10f}', flush=True)
                from hrm.training.records import record_evaluation
                record_evaluation(os.path.join(checkpoint_dir,'evaluation.jsonl'),seed=seed,step=step,split=eval_name,correct=num_total_correct[0],total=num_total_correct[1],cycles=config.eval_cycles,weights='EMA' if config.ema is not None else 'raw')
                record = dict(epoch=epoch, step=step, correct=num_total_correct[0], total=num_total_correct[1], exact_match=num_total_correct[0] / num_total_correct[1], eval_cycles=config.eval_cycles, checkpoint=f'epoch_{epoch}.pt')
                with open(os.path.join(checkpoint_dir, 'eval_metrics.jsonl'), 'a') as handle:
                    handle.write(json.dumps(record) + '\n')
        optim.swap_ema()
        if config.act_patience:
            torch.set_rng_state(training_rng_cpu)
            torch.cuda.set_rng_state(training_rng_cuda)
    if RANK == 0:
        from pathlib import Path
        records = [json.loads(line) for line in (Path(checkpoint_dir) / 'eval_metrics.jsonl').read_text().splitlines()]
        best = max(records, key=lambda r: (r['exact_match'], r['step']))
        (Path(checkpoint_dir) / 'training_summary.json').write_text(json.dumps(dict(complete=True, epochs=config.epochs, optimizer_steps=step, best=best, final=records[-1]), indent=2))
    if progress_bar is not None:
        progress_bar.close()
    if RANK == 0:
        wandb.finish()

def train(config_dict):
    from hrm.training.runtime import run_training
    run_training(config_dict, TrainConfig, train_single_seed)

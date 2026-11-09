from hrm.training.trainer import ArchConfig, DataConfig, load_module
from hrm.training.steps import train_step, run_inference
"""Fresh fixed-outer runs; step-triggered saves/evaluation, small-data sampling."""
import os
import json
from pathlib import Path
import yaml
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
import wandb
import coolname
from hydra.core.hydra_config import HydraConfig
from hrm.training.stochastic import TrainConfig as BaseConfig
from hrm.training.trainer import update_lr
from hrm.training.optimizer import AdamATan2
from hrm.training.step_control import optimizer_batches, evaluation_due, validate_budget


class TrainConfig(BaseConfig):
    max_steps: int
    eval_interval_steps: int


def train_single_seed(config, seed, group_name, world_size, rank):
    validate_budget(config.max_steps, config.eval_interval_steps,
                    config.cycles_per_data, config.epochs)
    if config.act_patience or config.outer_schedule or config.outer_symmetry:
        raise ValueError('Step-aligned rerun supports the fixed-outer recipe only')
    if config.resume_from_checkpoint or config.pilot_checkpoint:
        raise ValueError('This rerun starts fresh; partial-run recovery is not implemented')
    data = dict(config.data.__pydantic_extra__ or {})
    if 'epoch_draws' in data:
        raise ValueError('Use the small-data loader, not epoch_draws')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    create = load_module(f'dataset.{config.data.name}@create_dataloader')
    loader, metadata = create('train', config.local_batch_size, rank=rank,
                              world_size=world_size, seed=seed, **data)
    eval_data = dict(data)
    if config.eval_dataset_name is not None:
        eval_data['dataset_name'] = config.eval_dataset_name
    eval_loader, _ = create('test_hard', config.local_batch_size, rank=rank,
                            world_size=world_size, seed=seed, **eval_data)
    with torch.device('cuda'):
        model = load_module(f'arch.{config.arch.name}')(config.arch.__pydantic_extra__ | metadata)
        model = DDP(torch.compile(model, dynamic=False, fullgraph=True) if config.compile_model else model, static_graph=True)
    optim = AdamATan2(model.parameters(), lr=torch.tensor(0., device='cpu'),
                     betas=(config.beta1, config.beta2), weight_decay=config.weight_decay,
                     ema=config.ema)
    run_name = f'{group_name}/seed_{seed}'
    ck = Path(config.output_dir) / run_name
    ck.mkdir(parents=True, exist_ok=True)
    if (ck/'eval_metrics.jsonl').exists():
        raise ValueError('Existing metrics: refusing to append a second training run')
    progress = None
    if rank == 0:
        (ck/'model_config.json').write_text(json.dumps(config.model_dump(mode="json"), indent=2))
        print(json.dumps(dict(event='fresh_start', seed=seed, step=0, world_size=world_size,
                              optimizer_steps_zero=True, resume_checkpoint=None)), flush=True)
        progress = tqdm.tqdm(total=config.max_steps, desc=f'seed={seed}')
        wandb.init(project=config.data.name, name=run_name, group=group_name,
                   config=config.model_dump() | {'seed': seed},
                   settings=wandb.Settings(x_disable_stats=True))
    records = []

    def save_and_evaluate(step, data_pass):
        # 'epoch' remains the zero-based evaluation index for existing consumers.
        epoch = step // config.eval_interval_steps - 1
        if rank == 0:
            raw = dict(format_version=3, epoch=epoch, step=step, total_steps=config.max_steps,
                       seed=seed, world_size=world_size, data_pass=data_pass,
                       fresh_run_only=True, model_state_dict=model.module.state_dict(),
                       optimizer_state_dict=optim.state_dict())
            torch.save(raw, ck/'latest_train.pt.tmp')
            os.replace(ck/'latest_train.pt.tmp', ck/'latest_train.pt')
        dist.barrier()
        model.eval()
        optim.swap_ema()
        try:
            if rank == 0:
                torch.save({k.replace('_orig_mod.', ''): v for k, v in model.module.state_dict().items()},
                           ck/f'epoch_{epoch}.pt')
            counts = torch.zeros(2, device='cuda')
            for x_eval, y_eval in eval_loader:
                carry_eval = model.module.initial_carry
                x_eval = x_eval.cuda()
                y_eval = y_eval.cuda()
                for _ in range(config.eval_cycles):
                    carry_eval, pred = run_inference(model, carry_eval, x_eval)
                counts[0] += torch.all(pred == y_eval, dim=-1).sum()
                counts[1] += y_eval.shape[0]
            dist.reduce(counts, dst=0)
            if rank == 0:
                correct, total = counts.cpu().tolist()
                from hrm.training.records import record_evaluation
                record_evaluation(ck/'evaluation.jsonl',seed=seed,step=step,split='test_hard',correct=int(correct),total=int(total),cycles=config.eval_cycles,weights='EMA' if config.ema is not None else 'raw')
                rec = dict(epoch=epoch, step=step, correct=int(correct), total=int(total),
                           exact_match=correct/total, eval_cycles=config.eval_cycles,
                           checkpoint=f'epoch_{epoch}.pt')
                records.append(rec)
                with (ck/'eval_metrics.jsonl').open('a') as f:
                    f.write(json.dumps(rec)+'\n')
                print('EVAL_RESULT '+json.dumps(rec), flush=True)
                wandb.log({'eval/test_hard_exact_match': rec['exact_match']}, step=step)
        finally:
            optim.swap_ema()
            model.train()

    model.train()
    previous_pass = None
    for step, outer_index, data_pass, batch in optimizer_batches(
            loader, config.max_steps, config.cycles_per_data):
        if outer_index == 0:
            x, y = (tensor.cuda() for tensor in batch)
            carry = model.module.initial_carry
        if data_pass != previous_pass and rank == 0:
            print(json.dumps(dict(event='data_pass', data_pass=data_pass, step=step)), flush=True)
        previous_pass = data_pass
        lr = update_lr(config, optim, step, config.max_steps)
        carry, metrics = train_step(model, carry, optim, x, y)
        metrics.pop('_correct', None)
        if outer_index + 1 == config.cycles_per_data:
            if rank == 0 and step-progress.n >= config.log_interval:
                progress.update(step-progress.n)
                wandb.log({f'train/{k}': v.item() for k, v in metrics.items()} | {'train/lr': lr}, step=step)
                print(json.dumps(dict(event='train_progress', step=step, lr=lr,
                                      loss=float(metrics['loss']))), flush=True)
            del x, y, carry, metrics
        if evaluation_due(step, config.eval_interval_steps):
            save_and_evaluate(step, data_pass)
    if rank == 0:
        assert [r['step'] for r in records] == list(range(config.eval_interval_steps,
                                                        config.max_steps+1, config.eval_interval_steps))
        summary = dict(complete=True, epochs=len(records), optimizer_steps=step,
                       best=max(records, key=lambda r:(r['exact_match'],r['step'])), final=records[-1])
        (ck/'training_summary.json').write_text(json.dumps(summary, indent=2))
        progress.update(step-progress.n)
        progress.close()
        wandb.finish()





def train(config_dict):
    from hrm.training.runtime import run_training
    run_training(config_dict, TrainConfig, train_single_seed)

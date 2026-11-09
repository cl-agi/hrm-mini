"""Measure native hrm-mini inference; without --execute, print the plan only."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def duration_stats(values):
    return {'n': len(values), 'mean_seconds': statistics.mean(values) if values else None, 'sample_sd_seconds': statistics.stdev(values) if len(values) > 1 else None, 'total_seconds': sum(values)}

def timed_call(function, synchronize, clock=time.perf_counter):
    synchronize()
    started = clock()
    value = function()
    synchronize()
    return (value, clock() - started)

def run_cycles(model, native_inference, x, cycles, attn_mask=None):
    carry = model.initial_carry
    prediction = None
    for _ in range(cycles):
        if attn_mask is None:
            carry, prediction = native_inference(model, carry, x)
        else:
            carry, prediction = native_inference(model, carry, x, attn_mask)
    return prediction

def planned_counts(dataset_size, batch_size, max_examples):
    available = dataset_size // batch_size * batch_size
    measured = min(available, max_examples) if max_examples else available
    measured = measured // batch_size * batch_size
    return {'dataset_examples': dataset_size, 'loader_examples': available, 'measured_examples': measured, 'native_drop_last_examples': dataset_size - available, 'limit_excluded_examples': available - measured}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--task', required=True, choices=['sudoku', 'maze', 'sat'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup-batches', type=int, default=3, help='Repeat the first batch with a fresh carry; includes compilation.')
    parser.add_argument('--max-examples', type=int, default=0, help='0 uses the full split.')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.warmup_batches < 1 or args.max_examples < 0:
        parser.error('batch-size and warmup-batches must be positive; max-examples must be nonnegative')
    for name in ('runtime', 'checkpoint', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    return args

def execute(args, plan):
    pass
    os.chdir(args.runtime)
    import torch
    import yaml
    from hrm.training.trainer import TrainConfig, load_module, run_inference
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA device is required for the prepared inference timing experiment')
    config_path = args.checkpoint.parent / 'model_config.json'
    config_dict = yaml.safe_load(config_path.read_text())
    config = TrainConfig(**config_dict)
    if config.data.name != args.task:
        raise ValueError('Checkpoint task does not match --task')
    if config.cycles_per_data != 16 or args.seed not in config.seeds:
        raise ValueError('Expected the native 16-cycle checkpoint and its recorded training seed')
    data_extra = dict(config.data.__pydantic_extra__ or {})
    data_extra.pop('eval_splits', None)
    create_loader = load_module(f'dataset.{config.data.name}@create_dataloader')
    loader, metadata = create_loader(plan['split'], args.batch_size, rank=0, world_size=1, seed=args.seed, **data_extra)
    if loader.sampler.shuffle:
        raise ValueError('Evaluation requires the native sequential evaluation sampler')
    counts = planned_counts(len(loader.dataset), args.batch_size, args.max_examples)
    if counts['measured_examples'] == 0:
        raise ValueError('The selected split/limit contains no full evaluation batch')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    init_started = time.perf_counter()
    with torch.device('cuda'):
        model_cls = load_module(f'arch.{config.arch.name}')
        model = model_cls(dict(config.arch.__pydantic_extra__ or {}) | metadata)
        state = torch.load(args.checkpoint, map_location='cuda', weights_only=True)
        model.load_state_dict(state, assign=True)
        del state
        model = torch.compile(model, dynamic=False, fullgraph=True)
        model.eval()
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - init_started
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    iterator = iter(loader)
    first = next(iterator)

    def unpack(batch):
        if isinstance(batch, dict):
            return (batch['x'], batch['y'], batch.get('y_mask'), batch.get('attn_mask'))
        x, y = batch
        return (x, y, None, None)

    def to_device(batch):
        x, _, _, attn_mask = unpack(batch)
        return (x.cuda(), attn_mask.cuda() if attn_mask is not None else None)
    first_device = to_device(first)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    warmups = []
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            prediction, elapsed = timed_call(lambda: run_cycles(model, run_inference, first_device[0], config.cycles_per_data, first_device[1]), torch.cuda.synchronize)
            warmups.append(elapsed)
            del prediction
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    del first_device
    write_json(args.output / 'warmup.json', {'same_first_batch_repeated': True, 'carry_reset_each_repetition': True, 'includes_compile': True, 'seconds': warmups, 'initialization_seconds': initialization_seconds})

    def tensor_hash(tensor):
        if tensor is None:
            return None
        return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
    durations, transfers, scoring = ([], [], [])
    total_correct = 0
    sample_offset = 0
    batches = itertools.islice(itertools.chain([first], iterator), counts['measured_examples'] // args.batch_size)
    with (args.output / 'per_call.jsonl').open('w') as records, torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            x, y, y_mask, attn_mask = unpack(batch)
            (device_x, device_mask), transfer_seconds = timed_call(lambda: to_device(batch), torch.cuda.synchronize)
            prediction, elapsed = timed_call(lambda: run_cycles(model, run_inference, device_x, config.cycles_per_data, device_mask), torch.cuda.synchronize)
            score_start = time.perf_counter()
            prediction = prediction.cpu()
            scored_prediction = prediction if y_mask is None else prediction * y_mask
            scored_target = y if y_mask is None else y * y_mask
            correct = torch.all(scored_prediction == scored_target, dim=-1).tolist()
            score_seconds = time.perf_counter() - score_start
            durations.append(elapsed)
            transfers.append(transfer_seconds)
            scoring.append(score_seconds)
            total_correct += sum(correct)
            for index in range(len(y)):
                row = {'sample_index': sample_offset + index, 'source_row_index': sample_offset + index, 'split': plan['split'], 'seed': args.seed, 'batch_index': batch_index, 'batch_size': len(y), 'batch_duration_seconds': elapsed, 'latency_seconds': elapsed if len(y) == 1 else None, 'amortized_seconds_per_sample': elapsed / len(y), 'batch_input_transfer_seconds': transfer_seconds, 'batch_scoring_seconds': score_seconds, 'input_sha256': tensor_hash(x[index]), 'attention_mask_sha256': tensor_hash(attn_mask[index]) if attn_mask is not None else None, 'x': x[index].tolist(), 'target': y[index].tolist(), 'y_mask': y_mask[index].tolist() if y_mask is not None else None, 'prediction': prediction[index].tolist(), 'correct': bool(correct[index])}
                records.write(json.dumps(row, allow_nan=False) + '\n')
            records.flush()
            sample_offset += len(y)
            del device_x, device_mask, prediction, scored_prediction, scored_target
    if sample_offset != counts['measured_examples']:
        raise RuntimeError(f"Expected {counts['measured_examples']} samples, measured {sample_offset}")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    summary = {'status': 'completed', **plan, **counts, 'correct': total_correct, 'accuracy': total_correct / sample_offset, 'timing_scope': 'synchronized device-resident native inference including carry initialization and argmax', 'cycles_per_data': config.cycles_per_data, 'batch_latency': duration_stats(durations), 'amortized_seconds_per_sample': sum(durations) / sample_offset, 'samples_per_second': sample_offset / sum(durations), 'input_transfer': duration_stats(transfers), 'scoring': duration_stats(scoring), 'initialization_seconds': initialization_seconds, 'warmup_seconds': warmups, 'statistics_ddof': 1, 'checkpoint_sha256': sha256_file(args.checkpoint), 'per_call_sha256': sha256_file(args.output / 'per_call.jsonl'), 'runtime_config': config_dict, 'runtime_config_sha256': sha256_file(config_path), 'runtime_python_sha256': {str(p.relative_to(args.runtime)): sha256_file(p) for p in sorted(args.runtime.rglob('*.py'))}, 'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'device': device.name, 'device_total_memory_bytes': device.total_memory, 'device_capability': [device.major, device.minor], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch_compile': {'dynamic': False, 'fullgraph': True}, 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}}
    write_json(args.output / 'inference_summary.json', summary)
    print(json.dumps({'status': 'completed', 'output': str(args.output), 'measured_examples': sample_offset, 'accuracy': summary['accuracy']}))

def main(argv=None):
    args = parse_args(argv)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != 'execute'}
    plan['split'] = 'test_hard' if args.task == 'sudoku' else 'test'
    if not args.execute:
        print(json.dumps({'status': 'planned', **plan}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'plan.json', plan)
    try:
        execute(args, plan)
    except BaseException as error:
        write_json(args.output / 'failed.json', {'status': 'failed', 'exception_type': type(error).__name__, 'message': str(error)})
        raise
if __name__ == '__main__':
    main()

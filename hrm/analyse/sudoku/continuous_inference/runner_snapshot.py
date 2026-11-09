"""Independent fixed-input, continuous-carry Sudoku test-time scaling audit.

Each outer cycle is exactly one native model.forward(carry, x), with no answer
argument, no carry reset, no restart and no early stopping. CUDA graph replay
only removes Python launch overhead. Every prediction is checked against both
the target (metrics only) and a target-free complete Sudoku/clue validator.
The saved per-example trajectories have shape [example, outer_cycle].
"""
from __future__ import annotations
from hrm.paths import analysis_directory
import argparse
import csv
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
DEFAULT_POINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_metadata(args):
    checkpoint = Path(args.checkpoint).resolve()
    from hrm.paths import checkpoint_config
    if args.config:
        path=Path(args.config).resolve()
        if not path.is_file():raise FileNotFoundError(path)
    else:
        try:path=checkpoint_config(checkpoint)
        except FileNotFoundError:path=checkpoint.parent/'model_config.json'
    if getattr(args,"public_arch",None) is not None:
        arch=dict(args.public_arch);training={};source="published_HF_config";fallback=False
    elif path.exists():
        training = yaml.safe_load(path.read_text())
        arch = dict(training.get('transformer', {})) | dict(training.get('arch', training))
        source = str(path)
        fallback = False
    else:
        if not args.architecture:
            raise ValueError('Missing model_config.json: specify --architecture for the documented fallback')
        arch = dict(seq_len=82, vocab_size=10, hidden_size=512, intermediate_size=2048, head_dim=64, is_causal=False, norm_eps=1e-06, rope_theta=10000.0, bptt=True, forward_dtype='bfloat16')
        names = {'hrm': 'hrm@HRM', 'trm': 'hrm_ablations@HRM', 'rt': 'rt@RecurrentTransformer', 'hrm_flow_bias': 'hrm_flow_bias@HRMFlowBias', 'trm_flow_bias': 'trm_flow_bias@TRMFlowBias', 'rt_flow_bias': 'rt_flow_bias@RecurrentTransformer'}
        arch['name'] = names[args.architecture]
        if args.architecture.startswith('rt'):
            arch.update(num_layers=4, cycles=7)
        else:
            arch.update(num_layers=2, H_cycles=2, L_cycles=6)
        if args.architecture == 'trm':
            arch.update(dual_module=False, hh_link=True, sym_io=False)
        if 'flow_bias' in args.architecture:
            arch.update(flow_lambda=0.025, mixture_logit_std=1.0, preserve_first_zero=True, normalize_mixture_variance=True)
        training, source, fallback = ({}, 'explicit_standard_fallback', True)
    arch = {'seq_len': 82, 'vocab_size': 10, 'is_causal': False, **arch}
    if arch['seq_len'] != 82 or arch['vocab_size'] != 10:
        raise ValueError('This evaluator is specifically for 82-token Sudoku models')
    module, cls = arch['name'].split('@')
    from hrm.registry import resolve_module
    module = resolve_module(module if module.startswith(('arch.','hrm.')) else 'arch.'+module)
    if module not in {'hrm.models.hrm','hrm.models.ablations','hrm.models.rt','hrm.models.hrm_flow_bias','hrm.models.rt_flow_bias'}:
        raise ValueError(f"Architecture {module} is outside this independent audit's supported set")
    with torch.device(args.device):
        from hrm.registry import model_class
        model = model_class(f'{module}@{cls}', arch)(arch).eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.removeprefix('module.').removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return (model, {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'config_source': source, 'config_fallback': fallback, 'arch': arch, 'training_cycles_per_data': training.get('cycles_per_data'), 'training_config': training})

def load_inputs(args):
    path = Path(args.dataset)
    if path.is_dir():
        path = path / 'test_hard.csv'
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.max_examples > len(rows):
        raise ValueError(f'Requested {args.max_examples}, but CSV has {len(rows)} rows')
    selected = np.random.default_rng(args.seed).permutation(len(rows))[:args.max_examples]
    if args.indices:
        selected = np.load(args.indices).astype(np.int64)
        if selected.ndim != 1 or len(selected) != args.max_examples:
            raise ValueError('Explicit indices must be a vector matching --max-examples')
        if len(set(selected.tolist())) != len(selected) or np.any((selected < 0) | (selected >= len(rows))):
            raise ValueError('Explicit indices must be unique valid CSV row indices')
    indices = selected[args.shard_index::args.shard_count]
    xs, ys = ([], [])
    for idx in indices:
        question, answer = (rows[int(idx)]['question'], rows[int(idx)]['answer'])
        if len(question) != 81 or len(answer) != 81:
            raise ValueError(f'Unexpected Sudoku length at CSV row {idx}')
        xs.append([0] + [int(c) for c in question.replace('.', '0')])
        ys.append([0] + [int(c) for c in answer])
    if not xs:
        raise ValueError('Empty shard')
    return (np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64), indices, {'csv': str(path.resolve()), 'csv_sha256': sha256(path), 'csv_examples': len(rows), 'selection': str(Path(args.indices).resolve()) if args.indices else 'numpy.default_rng(seed).permutation(csv_examples)[:max_examples]', 'seed': args.seed, 'max_examples': args.max_examples, 'all_indices_sha256': hashlib.sha256(selected.tobytes()).hexdigest(), 'shard_index': args.shard_index, 'shard_count': args.shard_count})

def sudoku_valid(prediction, input_ids):
    """No targets: each row, column, box is 1..9 and every clue is retained."""
    board = prediction[:, 1:].reshape(-1, 9, 9)
    required = torch.arange(1, 10, device=board.device).reshape(1, 1, 9)
    rows = board.sort(dim=-1).values.eq(required).all(dim=(1, 2))
    cols = board.transpose(1, 2).sort(dim=-1).values.eq(required).all(dim=(1, 2))
    boxes = board.reshape(-1, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes = boxes.sort(dim=-1).values.eq(required).all(dim=(1, 2))
    clues = (input_ids[:, 1:].eq(0) | prediction[:, 1:].eq(input_ids[:, 1:])).all(dim=1)
    return rows & cols & boxes & clues

def fixed_mixture(indices, seed, model, device):
    weights = []
    for index in indices:
        generator = torch.Generator(device='cpu').manual_seed(seed * 1000003 + int(index))
        logits = torch.randn((82, 9), generator=generator, dtype=torch.float32)
        weights.append(logits)
    return torch.softmax(torch.stack(weights).to(device) * model.mixture_logit_std, dim=-1)

class GraphRunner:

    def __init__(self, model, batch, cycles, device):
        self.model, self.batch, self.cycles, self.device = (model, batch, cycles, device)
        self.x = torch.zeros(batch, 82, dtype=torch.long, device=device)
        self.y = torch.zeros_like(self.x)
        self.carry = {key: value.expand(batch, 82, -1).clone() for key, value in model.initial_carry.items()}
        self.flow = hasattr(model, 'sample_mixture_weights')
        if self.flow:
            self.mixture = torch.zeros(batch, 82, 9, dtype=torch.float32, device=device)
            self.carry['mixture_weights'] = self.mixture
        self.trace = torch.zeros(cycles, batch, 3, dtype=torch.bool, device=device)
        self.step = torch.zeros(1, dtype=torch.long, device=device)
        self.first_valid = torch.zeros(batch, dtype=torch.long, device=device)
        self.first_valid_prediction = torch.zeros_like(self.x)
        self.graph = None

    def reset(self, x, y, mixture=None):
        self.x.copy_(x)
        self.y.copy_(y)
        for key, initial in self.model.initial_carry.items():
            self.carry[key].copy_(initial)
        if self.flow:
            self.mixture.copy_(mixture)
        self.trace.zero_()
        self.step.zero_()
        self.first_valid.zero_()
        self.first_valid_prediction.zero_()

    def transition(self):
        new_carry, logits = self.model(self.carry, self.x)
        for key in self.model.initial_carry:
            self.carry[key].copy_(new_carry[key])
        self.logits = logits
        self.prediction = logits.argmax(dim=-1)
        correct = self.prediction.eq(self.y)
        valid = sudoku_valid(self.prediction, self.x)
        metrics = torch.stack((correct.all(dim=-1), correct[:, 1:].all(dim=-1), valid), dim=-1)
        self.trace.index_copy_(0, self.step, metrics.unsqueeze(0))
        first = valid & self.first_valid.eq(0)
        self.first_valid.copy_(torch.where(first, self.step[0] + 1, self.first_valid))
        self.first_valid_prediction.copy_(torch.where(first[:, None], self.prediction, self.first_valid_prediction))
        self.step.add_(1)
        return logits

    def capture(self, x, y, mixture):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.reset(x, y, mixture)
                self.transition()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.reset(x, y, mixture)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.transition()
        self.reset(x, y, mixture)

    def replay(self):
        if self.graph is None: self.transition()
        else: self.graph.replay()

@torch.inference_mode()
def parity_check(runner, x, y, mixture, steps):
    model = runner.model
    carry = model.initial_carry
    if runner.flow:
        carry = carry | {'mixture_weights': mixture}
    eager_logits = []
    for _ in range(steps):
        carry, logits = model(carry, x)
        eager_logits.append(logits.clone())
    eager_carry = {key: value.clone() for key, value in carry.items() if key != 'mixture_weights'}
    runner.reset(x, y, mixture)
    graph_logits = []
    for _ in range(steps):
        runner.replay()
        graph_logits.append(runner.logits.clone())
    if runner.device.type == "cuda": torch.cuda.synchronize(runner.device)
    eager_logits, graph_logits = (torch.stack(eager_logits), torch.stack(graph_logits))
    bitwise = torch.equal(eager_logits, graph_logits)
    argmax_equal = torch.equal(eager_logits.argmax(-1), graph_logits.argmax(-1))
    maximum = float((eager_logits.float() - graph_logits.float()).abs().max())
    carry_equal = all((torch.equal(value, runner.carry[key]) for key, value in eager_carry.items()))
    eager_pred = eager_logits.argmax(-1)
    flat = eager_pred.reshape(-1, 82)
    x_repeated = x.unsqueeze(0).expand(steps, -1, -1).reshape(-1, 82)
    reference_valid = sudoku_valid(flat, x_repeated).reshape(steps, -1)
    metrics_equal = torch.equal(reference_valid, runner.trace[:steps, :, 2])
    record = {'steps': steps, 'logits_bitwise_equal': bitwise, 'argmax_equal': argmax_equal, 'max_abs_logit_error': maximum, 'final_carry_bitwise_equal': carry_equal, 'valid_trace_equal': metrics_equal, 'native_initial_carry': 'unexpanded vector', 'graph_initial_carry': 'expanded contiguous broadcast', 'native_api': 'model(carry, x); no target argument'}
    if not (bitwise and argmax_equal and carry_equal and metrics_equal):
        raise RuntimeError(f'Buffered/native eager parity failed: {record}')
    runner.reset(x, y, mixture)
    return record

def summarize(trace, first_valid, points):
    records = []
    for step in points:
        current = trace[:, step - 1]
        records.append({'outer_cycles': step, 'examples':len(trace),
                        'exact_reversal_examples':int((np.maximum.accumulate(trace[:, :step, 0],axis=1) & ~trace[:, :step, 0]).any(axis=1).sum()),
                        'valid_reversal_examples':int((np.maximum.accumulate(trace[:, :step, 2],axis=1) & ~trace[:, :step, 2]).any(axis=1).sum()), 'exact_all82': float(current[:, 0].mean()), 'exact_board81': float(current[:, 1].mean()), 'valid_final': float(current[:, 2].mean()), 'ever_exact_all82': float(trace[:, :step, 0].any(axis=1).mean()), 'ever_exact_board81': float(trace[:, :step, 1].any(axis=1).mean()), 'ever_valid': float(((first_valid > 0) & (first_valid <= step)).mean())})
    return records

@torch.inference_mode()
def main(args):
    torch.set_num_threads(args.cpu_threads)
    if args.device.type == "cuda": torch.cuda.set_device(args.device.index if args.device.index is not None else 0)
    from hrm import hub
    from hrm.paths import repository_root
    if args.seed is None:
        if args.shard_count>1:raise ValueError('Sharded evaluation requires one explicitly shared --seed')
        args.seed=int(time.time())
    if args.checkpoint is None:
        args.checkpoint=hub.checkpoint('sudoku')
        if args.config is None:
            args.public_arch=hub.model_config('sudoku')
    else:args.checkpoint=hub.local_file(args.checkpoint)
    if args.dataset is None:args.dataset=hub.puzzle_data('sudoku',split='test')
    else:hub.local_file(Path(args.dataset)/'test_hard.csv' if Path(args.dataset).is_dir() else args.dataset)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Require 0 <= shard-index < shard-count')
    output = Path(args.output)
    plan_path=output.with_suffix('.plan.json')
    if output.exists() or output.with_suffix('.npz').exists() or plan_path.exists():
        raise FileExistsError('Existing frozen evaluation; choose a new --output')
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model, metadata = load_metadata(args)
    xs, ys, indices, data = load_inputs(args)
    plan_path.write_text(json.dumps(dict(model=metadata,data=data,seed=args.seed,
        max_cycles=args.max_cycles,batch_size=args.batch_size,device=str(args.device)),indent=2)+'\n')
    steps = args.max_cycles
    points = sorted(set([p for p in args.points if p <= steps] + [steps]))
    batch_size = min(args.batch_size, len(xs))
    runner = GraphRunner(model, batch_size, steps, args.device)
    traces, predictions, first_cycles, first_predictions = ([], [], [], [])
    terminal_carries = []
    prefix_audits = []
    reference = None
    if args.prefix_reference:
        with np.load(args.prefix_reference) as raw:
            reference = {k: raw[k] for k in raw.files}
        if not np.array_equal(reference['indices'], indices):
            raise ValueError('Prefix reference does not match shard IDs/order')
        if not np.array_equal(reference['inputs'], xs) or not np.array_equal(reference['targets'], ys):
            raise ValueError('Prefix reference inputs/targets mismatch')
        if reference['exact_all82'].shape[1] > steps:
            raise ValueError('Requested horizon is shorter than prefix reference')
    parity = None
    measured_seconds = 0.0
    for offset in range(0, len(xs), batch_size):
        n = min(batch_size, len(xs) - offset)
        selection = np.minimum(np.arange(offset, offset + batch_size), len(xs) - 1)
        x = torch.from_numpy(xs[selection]).to(args.device)
        y = torch.from_numpy(ys[selection]).to(args.device)
        mix = fixed_mixture(indices[selection], args.seed, model, args.device) if runner.flow else None
        if parity is None:
            if args.device.type == "cuda":runner.capture(x, y, mix)
            parity = parity_check(runner, x, y, mix, min(16, steps))
            print(json.dumps({'event': 'parity', **parity}), flush=True)
        runner.reset(x, y, mix)
        if args.device.type == "cuda": torch.cuda.synchronize(args.device)
        batch_start = time.perf_counter()
        checkpoints = set(range(args.progress_cycles, steps + 1, args.progress_cycles)) | {steps}
        if reference is not None:
            checkpoints.add(reference['exact_all82'].shape[1])
        previous = 0
        for until in sorted(checkpoints):
            for _ in range(until - previous):
                runner.replay()
            if args.device.type == "cuda": torch.cuda.synchronize(args.device)
            current = runner.trace[until - 1, :n].sum(dim=0).cpu().tolist()
            progress = {'event': 'cycle_progress', 'batch_offset': offset, 'batch_examples': n, 'outer_cycles': until, 'correct_all82': current[0], 'correct_board81': current[1], 'valid': current[2], 'batch_elapsed_seconds': time.perf_counter() - batch_start}
            if reference is not None and until == reference['exact_all82'].shape[1]:
                actual = runner.trace[:until, :n].permute(1, 0, 2).cpu().numpy()
                mismatch = {}
                for metric, key in enumerate(('exact_all82', 'exact_board81', 'valid')):
                    mismatch[key] = int(np.count_nonzero(actual[:, :, metric] != reference[key][offset:offset + n]))
                mismatch['final_prediction'] = int(np.count_nonzero(runner.prediction[:n].cpu().numpy() != reference['final_prediction'][offset:offset + n]))
                if any(mismatch.values()):
                    raise RuntimeError(f'Prefix repeatability failed: {mismatch}')
                prefix_audits.append({'batch_offset': offset, 'examples': n, 'cycles': until, 'mismatches': mismatch})
                progress['prefix_audit'] = prefix_audits[-1]
            print(json.dumps(progress), flush=True)
            progress_path = output.with_suffix('.progress.json')
            temporary = progress_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(progress) + '\n')
            temporary.replace(progress_path)
            previous = until
        if args.device.type == "cuda": torch.cuda.synchronize(args.device)
        batch_seconds = time.perf_counter() - batch_start
        measured_seconds += batch_seconds
        traces.append(runner.trace[:, :n].permute(1, 0, 2).cpu().numpy().copy())
        predictions.append(runner.prediction[:n].cpu().numpy().copy())
        first_cycles.append(runner.first_valid[:n].cpu().numpy().copy())
        first_predictions.append(runner.first_valid_prediction[:n].cpu().numpy().copy())
        if args.save_final_carry:
            terminal_carries.append({key: runner.carry[key][:n].cpu().clone() for key in model.initial_carry})
        partial = output.with_name(output.stem + f'.batch_{offset // batch_size}.npz')
        np.savez_compressed(partial, indices=indices[offset:offset + n], exact_all82=traces[-1][:, :, 0], exact_board81=traces[-1][:, :, 1], valid=traces[-1][:, :, 2], final_prediction=predictions[-1], first_valid_cycle=first_cycles[-1], first_valid_prediction=first_predictions[-1])
        compact = summarize(traces[-1], first_cycles[-1], [p for p in (16, 256, 2048, steps) if p <= steps])
        print(json.dumps({'event': 'batch', 'completed': offset + n, 'total': len(xs), 'seconds': batch_seconds, 'example_outer_cycles_per_second': n * steps / batch_seconds, 'summary': compact}), flush=True)
    trace = np.concatenate(traces)
    prediction = np.concatenate(predictions)
    first = np.concatenate(first_cycles)
    first_prediction = np.concatenate(first_predictions)
    artifact = output.with_suffix('.npz')
    np.savez_compressed(artifact, indices=indices, inputs=xs, targets=ys, exact_all82=trace[:, :, 0], exact_board81=trace[:, :, 1], valid=trace[:, :, 2], first_valid_cycle=first, first_valid_prediction=first_prediction, final_prediction=prediction)
    carry_artifact = None
    if args.save_final_carry:
        carry_artifact = output.with_suffix('.carry.pt')
        torch.save({'indices': torch.from_numpy(indices), 'input_ids': torch.from_numpy(xs), 'completed_outer_cycles': steps, 'checkpoint_sha256': metadata['checkpoint_sha256'], 'carry': {key: torch.cat([v[key] for v in terminal_carries], dim=0) for key in terminal_carries[0]}}, carry_artifact)
    arch = metadata['arch']
    blocks = arch['num_layers'] * (arch['cycles'] if 'cycles' in arch else arch['H_cycles'] * (arch['L_cycles'] + 1))
    result = {**metadata, 'data': data, 'examples': len(xs), 'max_outer_cycles': steps, 'points': points, 'rows': summarize(trace, first, points), 'parity': parity, 'protocol': {'one_outer_cycle': 'one unchanged native model.forward(carry, input_ids)', 'transformer_blocks_per_outer': blocks, 'H_updates_per_outer': arch.get('H_cycles'), 'input_fixed': True, 'carry_continuous': True, 'early_stopping': False, 'restarts': 0, 'target_passed_to_model': False, 'validator': 'every cycle, row/column/box permutations of 1..9 and all clues respected', 'flow_mixture': 'one fixed per-example seeded random mixture, independent of sharding' if runner.flow else None, 'trajectory_dimensions': ['example', 'outer_cycle_1_based'], 'first_valid_sentinel': 0}, 'runtime': {'device': str(args.device), 'gpu': torch.cuda.get_device_name(args.device) if args.device.type=='cuda' else None, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__, 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'float32_matmul_precision': torch.get_float32_matmul_precision(), 'sdpa_backends_enabled': {'flash': torch.backends.cuda.flash_sdp_enabled(), 'memory_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(), 'math': torch.backends.cuda.math_sdp_enabled()}, 'initial_carry_dtypes': {key: str(value.dtype) for key, value in model.initial_carry.items()}, 'batch_size': batch_size, 'model_parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}), 'cuda_graph': args.device.type=='cuda', 'torch_compile': False, 'timed_replay_seconds': measured_seconds, 'example_outer_cycles_per_second': len(xs) * steps / measured_seconds, 'wall_seconds': time.perf_counter() - start}, 'artifact': str(artifact.resolve()), 'final_carry_artifact': str(carry_artifact.resolve()) if carry_artifact else None, 'prefix_reference': str(Path(args.prefix_reference).resolve()) if args.prefix_reference else None, 'prefix_audits': prefix_audits}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'event': 'complete', 'output': str(output), 'rows': result['rows'], 'runtime': result['runtime']}), flush=True)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',nargs='?')
    parser.add_argument('--config')
    parser.add_argument('--architecture', choices=('hrm', 'trm', 'rt', 'hrm_flow_bias', 'trm_flow_bias', 'rt_flow_bias'))
    parser.add_argument('--dataset','--dataset-path')
    parser.add_argument('--max-examples', type=int, default=4096)
    parser.add_argument('--max-cycles', type=int, default=16384)
    parser.add_argument('--points', type=int, nargs='+', default=DEFAULT_POINTS)
    parser.add_argument('--seed', type=int, help='Frozen cohort/evaluation seed; Unix seconds when omitted for one shard')
    parser.add_argument('--indices', help='Optional .npy file of global selected CSV row indices')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int, default=1)
    from hrm.paths import repository_root
    import os
    parser.add_argument('--output', default=str(Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/sudoku/records/continuous'))/'continuous.json'))
    parser.add_argument('--prefix-reference')
    parser.add_argument('--progress-cycles', type=int, default=2048)
    parser.add_argument('--save-final-carry', action='store_true')
    args = parser.parse_args(argv)
    if min(args.max_examples, args.max_cycles, args.batch_size, args.shard_count, args.progress_cycles) <= 0:
        parser.error('Counts must be positive')
    if any((point <= 0 for point in args.points)):
        parser.error('All sampling points must be positive outer-cycle counts')
    args.device = torch.device(args.device)
    return args
if __name__ == '__main__':
    main(parse_args())

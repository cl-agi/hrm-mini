"""Uninterrupted Maze inference; report current and ever-valid endpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import HERE, batch, digest, load_arrays, make_model
from .metrics import cpu_valid, metrics
from hrm.training.steps import forward


class Runner:
    """Keep native carry buffers across all inference cycles, without early stop."""
    def __init__(self, model, inputs, cycles):
        self.model, self.inputs = model, inputs
        n, s = inputs['x'].shape
        self.carry = {k: v.expand(n, s, -1).clone() for k, v in model.initial_carry.items()}
        device = inputs['x'].device
        self.trace = torch.zeros(cycles, n, 2, dtype=torch.bool, device=device)
        self.step = torch.zeros(1, dtype=torch.long, device=device)
        self.graph = None
        self.reset()

    def reset(self):
        for k, v in self.model.initial_carry.items():
            self.carry[k].copy_(v)
        self.trace.zero_()
        self.step.zero_()

    def transition(self):
        carry, self.logits = forward(self.model, self.carry, self.inputs['x'])
        for k in self.carry:
            self.carry[k].copy_(carry[k])
        self.pred = self.logits.argmax(-1)
        values = torch.stack(metrics(self.pred, self.inputs), -1)
        self.trace.index_copy_(0, self.step, values[None])
        self.step.add_(1)

    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.reset()
                self.transition()
        torch.cuda.current_stream().wait_stream(stream)
        self.reset()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.transition()
        self.reset()

    def advance(self):
        if self.graph is None:
            self.transition()
        else:
            self.graph.replay()


@torch.inference_mode()
def parity(runner, steps):
    carry = runner.model.initial_carry
    expected = []
    for _ in range(steps):
        carry, logits = forward(runner.model, carry, runner.inputs['x'])
        expected.append(logits.clone())
    runner.reset()
    for logits in expected:
        runner.advance()
        if not torch.equal(logits, runner.logits):
            raise RuntimeError('Buffered/native inference logits differ')
    if not all(torch.equal(carry[k], runner.carry[k]) for k in carry):
        raise RuntimeError('Buffered/native final carry differs')
    expected_metrics = torch.stack([
        torch.stack(metrics(z.argmax(-1), runner.inputs), -1) for z in expected
    ])
    if not torch.equal(expected_metrics, runner.trace[:steps]):
        raise RuntimeError('Buffered/native metrics differ')
    runner.reset()
    return {'steps': steps, 'logits_bitwise_equal': True, 'carry_bitwise_equal': True,
            'metrics_equal': True}


def summarize(trace):
    """trace: [examples, inference cycles, semantic-valid/reference-exact]."""
    if trace.ndim != 3 or trace.shape[2] != 2 or not all(trace.shape):
        raise ValueError('Expected nonempty [examples, cycles, 2] trace')
    valid = trace[:, :, 0].astype(bool)
    ever = np.maximum.accumulate(valid, axis=1)
    cycles = trace.shape[1]
    points = sorted({2**i for i in range(cycles.bit_length()) if 2**i <= cycles} | {cycles})
    return {'count': len(trace), 'cycles': cycles,
            'correct_to_wrong_transitions': int((valid[:, :-1] & ~valid[:, 1:]).sum()),
            'examples_with_correct_to_wrong': int((valid[:, :-1] & ~valid[:, 1:]).any(1).sum()),
            'curve': [{'inference_cycles': c, 'current_valid': int(valid[:, c-1].sum()),
                       'ever_valid': int(ever[:, c-1].sum()),
                       'reference_exact': int(trace[:, c-1, 1].sum())} for c in points]}


@torch.inference_mode()
def evaluate(kind, checkpoint=None, data_dir=None, output=None, cycles=16384,
             batch_size=128, device='cuda', shard=0, shards=1):
    if cycles < 1 or batch_size < 1 or shards < 1 or not 0 <= shard < shards:
        raise ValueError('Invalid cycle, batch or shard specification')
    checkpoint = Path(checkpoint) if checkpoint else HERE/'models'/f'maze_{kind}'/'best.pt'
    data_dir = Path(data_dir) if data_dir else HERE/'data'
    source = data_dir/'maze_test.npz'
    # Analysis consumes prepared records; absent inputs are never replaced by a download.
    for path in (checkpoint, source):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(output) if output else HERE/'results'/f'maze_{kind}_shard{shard}'
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    arrays = load_arrays('maze', 'test', data_dir)
    if len(set(arrays['ids'].tolist())) != len(arrays['ids']):
        raise ValueError('Duplicate evaluation IDs')
    chosen = np.arange(len(arrays['ids']))[shard::shards]
    if not len(chosen):
        raise ValueError('Empty evaluation shard')
    plan = dict(kind=kind, checkpoint_sha256=digest(checkpoint), input_sha256=digest(source),
                cycles=cycles, batch_size=batch_size, device=device, shard=shard, shards=shards,
                total_count=len(arrays['ids']), indices=chosen.tolist(), ids=arrays['ids'][chosen].tolist())
    model = make_model('maze', kind, device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan, indent=2))
    traces, checks = [], []
    for start in range(0, len(chosen), batch_size):
        rows = chosen[start:start+batch_size]
        inputs = batch('maze', arrays, rows, 0, augment=False, device=device)
        runner = Runner(model, inputs, cycles)
        if torch.device(device).type == 'cuda':
            runner.capture()
        checks.append(parity(runner, min(16, cycles)))
        for _ in range(cycles):
            runner.advance()
        trace = runner.trace.cpu().numpy().transpose(1, 0, 2)
        pred = runner.pred.cpu().numpy()
        independent = np.array([cpu_valid(p, x) for p, x in zip(pred, arrays['x'][rows])])
        if not np.array_equal(independent, trace[:, -1, 0]):
            raise RuntimeError('Independent final shortest-path check failed')
        traces.append(trace)
        np.savez_compressed(output/f'batch_{start//batch_size}.npz', indices=rows,
                            ids=arrays['ids'][rows], metrics=trace)
    result = dict(plan, **summarize(np.concatenate(traces)), parity=checks,
                  final_cpu_audit=True, status='complete')
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['hrm', 'trm', 'rt'], required=True)
    p.add_argument('--checkpoint'); p.add_argument('--data-dir'); p.add_argument('--output')
    p.add_argument('--cycles', type=int, default=16384)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(evaluate(**vars(a)), indent=2))


if __name__ == '__main__':
    main()

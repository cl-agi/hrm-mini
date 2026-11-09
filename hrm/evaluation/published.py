"""Evaluation entry points for the four published task checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from hrm import hub
from hrm.config import read_config


def add_arguments(p):
    p.add_argument('--task', choices=['sudoku','maze','sat','cremp'], default='sudoku')
    p.add_argument('--config', type=Path, help='Override the published model configuration')
    p.add_argument('--checkpoint','--ckpt','--checkpoint-dir','--ckpt-dir', type=Path)
    p.add_argument('--data','--dataset-path','--data-root', type=Path)
    p.add_argument('--split', help='HF split; defaults to test (puzzles), test_public (SAT), validation (CREMP)')
    p.add_argument('--cycles', type=int, default=16)
    p.add_argument('--stable-cycles', type=int)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--maze-size', type=int, default=30)
    p.add_argument('--n-vars', type=int, choices=[512,1024], default=1024)
    p.add_argument('--starts', type=int, default=64)
    p.add_argument('--weight-state',choices=['raw','ema'],default='raw',help='SAT local checkpoints only; published weights use raw')
    p.add_argument('--k', type=int, default=64, help='CREMP candidates per molecule')
    p.add_argument('--batch-molecules', type=int, default=16)


def run(args):
    if args.cycles < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('Budgets and batch size must be positive')
    if args.task!='sat' and args.weight_state!='raw':raise ValueError('--weight-state is supported only for SAT here')
    if args.task == 'sat':
        if args.split not in (None, 'test_public'): raise ValueError('This entry uses the public SAT test split')
        if not 0 < args.starts <= 64: raise ValueError('SAT starts must be in 1..64')
        from hrm.analyse.sat.evaluate import evaluate
        cfg = read_config(args.config) if args.config else None
        protocol = hub.sat_eval_config(args.n_vars)
        if args.seed is not None: protocol['seed'] = args.seed
        result = evaluate(protocol, cfg, args.data, cycles=args.cycles, starts=args.starts,
                          limit=args.limit, checkpoint=args.checkpoint, device=args.device,weight_state=args.weight_state)
    elif args.task == 'cremp':
        if args.config is not None: raise ValueError('CREMP model config is embedded in its checkpoint')
        import runpy, sys
        command = ['hrm_cremp.generate_angles', '--checkpoint', str(hub.checkpoint('cremp', args.checkpoint)),
                   '--data', str(hub.cremp_data(args.data, split=args.split or 'validation')),
                   '--output', str(args.output), '--device', args.device, '--k', str(args.k),
                   '--batch-molecules', str(args.batch_molecules), '--seed', str(args.seed if args.seed is not None else 327681)]
        if args.limit is not None: command += ['--limit', str(args.limit)]
        previous = sys.argv
        try:
            sys.argv = command
            runpy.run_module('hrm_cremp.generate_angles', run_name='__main__')
        finally:
            sys.argv = previous
        return
    else:
        result = evaluate_puzzles(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


def evaluate_puzzles(args):
    import numpy as np
    import torch
    from hrm.registry import model_class
    from hrm.evaluation.inference import evaluate_batch
    from hrm.tasks import sudoku, maze
    seed = args.seed if args.seed is not None else (1 if args.task == 'sudoku' else 2)
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    checkpoint = hub.checkpoint(args.task, args.checkpoint)
    data = hub.puzzle_data(args.task, args.data, split=args.split or 'test', size=args.maze_size)
    config_path=args.config
    if config_path is None and args.checkpoint is not None:
        for name in ('model_config.json','resolved_config.json'):
            candidate=checkpoint.parent/name
            if candidate.is_file():
                config_path=candidate;break
    saved = read_config(config_path) if config_path else hub.model_config(args.task)
    cfg = dict(saved.get('transformer', {})) | saved.get('arch', saved.get('architecture', saved)).copy()
    identifier = cfg.pop('name', 'hrm@HRM')
    if not identifier.startswith(('hrm.','arch.')): identifier = 'arch.'+identifier
    with data.open() as f: rows = list(csv.DictReader(f))
    if args.limit: rows = rows[:args.limit]
    if not rows: raise ValueError('Evaluation dataset is empty')
    sizes = {len(r['question']) for r in rows}
    if len(sizes) != 1: raise ValueError('Use one puzzle size per evaluation')
    cfg.update(seq_len=len(rows[0]['question'])+1, vocab_size=10 if args.task=='sudoku' else len(maze.VOCAB), is_causal=False)
    with torch.device(args.device): model = model_class(identifier, cfg)(cfg)
    weights = torch.load(checkpoint, map_location=args.device, weights_only=True)
    weights = weights.get('model', weights)
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in weights.items()}, strict=True); model.eval()
    collate = (lambda batch: sudoku.collate_fn(batch, augment=False)) if args.task=='sudoku' else maze.collate_fn
    correct = ever = total = shortest = 0; digest = hashlib.sha256()
    for offset in range(0, len(rows), args.batch_size):
        x,y = collate(rows[offset:offset+args.batch_size]); digest.update(x.numpy().tobytes()); digest.update(y.numpy().tobytes())
        result = evaluate_batch(model, x.to(args.device), y.to(args.device), args.cycles, args.stable_cycles)
        correct += result['current_correct']; ever += result['ever_correct']; total += result['total']
        if args.task == 'maze':
            from hrm.analyse.maze.rt_extrapolation.maze_metrics import score_predictions
            shortest += int(score_predictions(x.numpy(), result['predictions'].cpu().numpy(), y.numpy())['shortest_path_correct'].sum())
    result = dict(checkpoint=str(checkpoint), data=str(data), seed=seed, inference_cycles=args.cycles,
                  stable_cycles=args.stable_cycles, total=total, current_correct=correct, ever_correct=ever,
                  current_accuracy=correct/total, ever_accuracy=ever/total, ordered_input_target_sha256=digest.hexdigest())
    if args.task == 'maze': result.update(shortest_path_correct=shortest, shortest_path_accuracy=shortest/total)
    return result

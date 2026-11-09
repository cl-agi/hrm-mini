"""Numerical single-clue evaluation using the original native final readout."""
from typing import Any
from pathlib import Path
import argparse,json
import numpy as np
import torch
from torch import nn
import tqdm
from hrm.paths import repo_path,work_path
from hrm.evaluation.config import load_config,EvalConfig as TrainConfig
from hrm.models.layers import Carry
from hrm.training.steps import generate,run_inference
from hrm.registry import model_class

SHIFT_BUCKETS = [("1_10", 1, 10), ("11_20", 11, 20), ("21_30", 21, 30), ("31plus", 31, 81)]
@torch.inference_mode()
def predict(model: nn.Module, config: TrainConfig, is_autoregressive: bool,
            questions: np.ndarray, batch_size: int) -> np.ndarray:
    """[N, 81] puzzles -> [N, 81] predicted solutions, with the same decoding `eval.py` uses."""
    out = []
    for start in tqdm.trange(0, len(questions), batch_size, desc="  inference", leave=False):
        batch = np.pad(questions[start:start + batch_size], ((0, 0), (1, 0))).astype(np.int32)  # BOS
        x = torch.from_numpy(batch).to(next(model.parameters()).device)
        if is_autoregressive:
            y_hat = generate(model, x)
        else:
            carry: Carry = model.initial_carry  # pyright: ignore[reportAssignmentType]
            y_hat = None
            for _ in range(config.cycles_per_data):
                carry, y_hat = run_inference(model, carry, x)
        out.append(y_hat[:, 1:].cpu().numpy())  # drop the BOS slot
    return np.concatenate(out, axis=0).astype(np.int64)

def summarize(pred_old: np.ndarray, pred_new: np.ndarray,
              sol_old: np.ndarray, sol_new: np.ndarray, cells: np.ndarray) -> dict[str, Any]:
    """All statistics for one model. `cells` holds the perturbed cell of each pair."""
    solved_old = np.all(pred_old == sol_old, axis=-1)

    # Diagnostic cells: where the two solutions disagree, minus the perturbed clue (it is given).
    diagnostic = sol_old != sol_new
    diagnostic[np.arange(len(cells)), cells] = False

    num_diagnostic = diagnostic.sum(-1)
    board = np.ones_like(diagnostic)

    def rates(pairs: np.ndarray, scope: np.ndarray, exclusive: bool) -> dict[str, Any]:
        """copy / solve rate over the cells `scope` selects, on the pairs `pairs` selects."""
        sel = scope[pairs]
        if not sel.any():
            return {"copy_rate": float("nan"), "solve_rate": float("nan")} | ({"other_rate": float("nan")} if exclusive else {})
        copy = float((pred_new[pairs] == sol_old[pairs])[sel].mean())
        solve = float((pred_new[pairs] == sol_new[pairs])[sel].mean())
        # Only on the diagnostic cells are the two exclusive; over the whole board they overlap on
        # every cell the edit left alone, so they there sum to well over 1 and have no remainder.
        return {"copy_rate": copy, "solve_rate": solve} | ({"other_rate": 1.0 - copy - solve} if exclusive else {})

    everything = np.ones(len(cells), dtype=bool)
    return {
        "num_pairs": int(len(cells)),
        "num_solved_original": int(solved_old.sum()),
        "exact_match_original": float(solved_old.mean()),
        "exact_match_perturbed": float(np.all(pred_new == sol_new, axis=-1).mean()),
        "cell_acc_original": float((pred_old == sol_old).mean()),
        "cell_acc_perturbed": float((pred_new == sol_new).mean()),
        # How much the prediction moved, against how much the true solution moved
        "pred_agreement": float((pred_new == pred_old).mean()),
        "solution_agreement": float((sol_new == sol_old).mean()),
        "num_cells_solution_changed": float((sol_new != sol_old).sum(-1).mean()),
        "num_cells_pred_changed": float((pred_new != pred_old).sum(-1).mean()),
        "num_diagnostic_cells": float(diagnostic.sum(-1).mean()),
        "clue_echoed": float((pred_new[np.arange(len(cells)), cells] == sol_new[np.arange(len(cells)), cells]).mean()),
        # copy / solve at both scopes: the whole 81-cell board, and the diagnostic cells only
        "board": {"all": rates(everything, board, False),
                  "solved_original": rates(solved_old, board, False)},
        "diagnostic": {"all": rates(everything, diagnostic, True),
                       "solved_original": rates(solved_old, diagnostic, True)},
        # The same diagnostic-cell read, split by how far the edit moved the solution
        "by_shift": {name: {"num_pairs": int(bucket.sum()),
                            "em_edited": float(np.all(pred_new[bucket] == sol_new[bucket], axis=-1).mean())
                                         if bucket.any() else float("nan"),
                            **rates(bucket, diagnostic, True)}
                     for name, lo, hi, bucket in
                     [(n, lo, hi, (num_diagnostic >= lo) & (num_diagnostic <= hi)) for n, lo, hi in SHIFT_BUCKETS]},
    }

def load_model(ckpt, config_path=None, device="cuda"):
    from hrm import hub
    from hrm.config import read_config
    saved = read_config(config_path) if config_path else hub.model_config('sudoku')
    if config_path is None:
        for name in ('model_config.json', 'resolved_config.json'):
            candidate = Path(ckpt).parent / name
            if candidate.is_file():
                saved = read_config(candidate); break
    arch = dict(saved.get('transformer', {})) | saved.get('arch', saved).copy()
    name = arch.pop('name', 'hrm@HRM')
    arch.update(vocab_size=10, seq_len=82, is_causal=False)
    cls = model_class(name if name.startswith(('arch.', 'hrm.')) else 'arch.'+name, arch)
    model = cls(arch).to(device)
    weights = torch.load(ckpt, map_location=device, weights_only=True)
    weights = weights.get('model', weights)
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in weights.items()}, strict=True)
    model.eval()
    return model, getattr(cls, 'is_autoregressive', False)


def load_pairs(dataset_path=None):
    from hrm import hub
    if dataset_path:
        with np.load(dataset_path) as z: data = dict(z)
    else:
        rows = hub.dataset('sudoku', 'test_single_clue')
        parents = {r['id']:r for split in ('train','test') for r in hub.dataset('sudoku',split)}
        original = [parents[r['parent_id']] for r in rows]
        digits = lambda rs, field: np.array([[int(c) for c in r[field]] for r in rs],dtype=np.int64)
        data = dict(question_old=digits(original,'puzzle'), question_new=digits(rows,'puzzle'),
                    solution_old=digits(original,'solution'), solution_new=digits(rows,'solution'),
                    ids=np.array([r['id'] for r in rows]), parent_ids=np.array([r['parent_id'] for r in rows]))
    changed = data['question_old'] != data['question_new']
    if changed.ndim!=2 or changed.shape[1]!=81 or not np.all(changed.sum(1)==1):
        raise ValueError('Each pair must change exactly one clue in an 81-cell puzzle')
    data['cell']=changed.argmax(1)
    return data


def main(argv=None):
    from types import SimpleNamespace
    from hrm import hub
    from hrm.paths import repository_root
    from hrm.analyse.dynamics.constraint_readout.run import finite
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--dataset-path',type=Path,help='Paired NPZ with question_old/new and solution_old/new')
    parser.add_argument('--output',type=Path,default=repository_root()/'exp/generalization/records/single_clue')
    parser.add_argument('--batch-size',type=int,default=32);parser.add_argument('--cycles',type=int,default=16)
    parser.add_argument('--device',default='cuda');parser.add_argument('--limit',type=int)
    parser.add_argument('--arm',default='hrm',help='Label for the evaluated checkpoint')
    args=parser.parse_args(argv)
    if min(args.batch_size,args.cycles)<1 or (args.limit is not None and args.limit<1):parser.error('Positive budgets required')
    if args.arm!='hrm' and args.checkpoint is None:parser.error('Comparison arms require --checkpoint')
    data=load_pairs(args.dataset_path)
    if args.limit:data={k:v[:args.limit] for k,v in data.items()}
    ckpt=hub.checkpoint('sudoku',args.checkpoint)
    model,autoregressive=load_model(ckpt,args.config,args.device)
    config=SimpleNamespace(cycles_per_data=args.cycles)
    old=predict(model,config,autoregressive,data['question_old'],args.batch_size)
    new=predict(model,config,autoregressive,data['question_new'],args.batch_size)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'pairs.npz',**data,pred_old=old,pred_new=new)
    result=dict(checkpoint=str(ckpt),arm=args.arm,inference_cycles=args.cycles,subset_limit=args.limit,
                results=summarize(old,new,data['solution_old'],data['solution_new'],data['cell']))
    if 'parent_ids' in data:
        result['by_parent_split']={}
        for split in ('train','test'):
            mask=np.array([str(i).startswith(split+':') for i in data['parent_ids']])
            if mask.any():result['by_parent_split'][split]=summarize(old[mask],new[mask],data['solution_old'][mask],data['solution_new'][mask],data['cell'][mask])
    (args.output/'summary.json').write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(finite(result)))

if __name__=='__main__':main()

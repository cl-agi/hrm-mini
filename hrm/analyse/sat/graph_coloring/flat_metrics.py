"""CNF checking and strict raw-record resume audit for frozen SAT checkpoints."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_predictions(predictions, clauses):
    """Return per-cycle CNF success and violated-clause counts.

    Original vocabulary IDs 1/2 represent False/True. Other output IDs are
    retained in the raw archive and make an assignment invalid.
    """
    p, c = np.asarray(predictions), np.asarray(clauses)
    require(p.ndim == 3 and c.ndim == 3, 'Expected [time,batch,variable] and [batch,clause,width]')
    require(p.shape[1] == c.shape[0] and p.shape[2] > 0, 'Prediction/CNF dimensions differ')
    require(np.issubdtype(p.dtype, np.integer) and np.issubdtype(c.dtype, np.integer), 'Non-integer predictions or clauses')
    require(np.all(c != 0) and np.max(np.abs(c)) <= p.shape[2], 'Invalid literal index')
    success = np.empty(p.shape[:2], dtype=bool)
    violations = np.empty(p.shape[:2], dtype=np.int32)
    for batch_index in range(p.shape[1]):
        q, cnf = p[:, batch_index], c[batch_index]
        clause_ok = (q[:, np.abs(cnf)-1] == np.where(cnf > 0, 2, 1)[None]).any(-1)
        violations[:, batch_index] = (~clause_ok).sum(1)
        success[:, batch_index] = clause_ok.all(1) & ((q == 1) | (q == 2)).all(1)
    return success, violations


def read_dimacs(path):
    """Read original variable-width clauses without rewriting the source file."""
    variables = expected = None
    clauses, pending = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('c'):
            continue
        if line.startswith('p'):
            words = line.split()
            require(len(words) == 4 and words[:2] == ['p', 'cnf'], f'Bad DIMACS header: {path}')
            variables, expected = map(int, words[2:])
            continue
        if line.startswith('%'):
            break
        for token in line.split():
            literal = int(token)
            if literal:
                pending.append(literal)
            else:
                clauses.append(pending)
                pending = []
    require(variables is not None and not pending and len(clauses) == expected, f'Incomplete DIMACS: {path}')
    require(all(c and all(0 < abs(x) <= variables for x in c) for c in clauses), f'Invalid DIMACS clause: {path}')
    return variables, clauses


def validate_original_cnf(cfg, clauses, n_variables):
    paths = cfg.get('original_cnf_files')
    if paths is None:
        return
    hashes = cfg['original_cnf_sha256']
    require(len(paths) == len(hashes) == len(clauses), 'Original CNF manifest length differs')
    for idx, (path, expected) in enumerate(zip(paths, hashes)):
        require(digest(path) == expected, f'Original CNF changed: {path}')
        variables, original = read_dimacs(path)
        require(variables == n_variables and len(original) == len(clauses[idx]), f'Original CNF shape differs: {path}')
        for source_clause, encoded_clause in zip(original, clauses[idx]):
            require(set(source_clause) == set(map(int, encoded_clause)), f'Encoded clause differs from original: {path}')



def wave_configuration(cfg):
    group, cases = cfg['starts_per_batch'], cfg['case_microbatch']
    require(type(group) is int and group > 0 and (group & (group - 1)) == 0 and cfg['starts'] % group == 0, 'Invalid start-wave size')
    require(type(cases) is int and 1 <= cases <= 13 and cfg['microbatch'] == cases * group, 'Invalid wave microbatch')
    return group, cases


def next_tasks(assigned, solved, attempts, starts, microbatch):
    """Fill a GPU batch with independent (start, case) pairs in fixed order."""
    pending = sorted((start, int(case)) for case in assigned if not solved[case]
                     for start in range(int(attempts[case]), starts))
    selected = pending[:microbatch]
    return (np.asarray([case for start, case in selected], dtype=np.int64),
            np.asarray([start for start, case in selected], dtype=np.int64))


def audit_rank(model_dir, rank, cfg, clauses, data_sha, worker_sha, metrics_sha):
    """Recheck atomic batches and reconstruct the exact independent-task queue."""
    model_dir = Path(model_dir)
    out = model_dir / f'rank{rank:02d}'
    n_cases, starts, cycles = len(clauses), cfg['starts'], cfg['inference_cycles']
    cfg_sha = digest(model_dir / 'config.json')
    assigned = np.arange(rank, n_cases, cfg['world_size'])
    solved = np.zeros(n_cases, dtype=bool)
    attempts = np.zeros(n_cases, dtype=np.int64)
    environments, records = {}, []
    for path in sorted(out.glob('b*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path, allow_pickle=False) as z:
            batch_index = int(z['batch_index'])
            require(batch_index == len(records) and path.name == f'b{batch_index:04d}.npz', f'Non-contiguous batch index: {path}')
            ids, task_starts = z['case_ids'], z['starts']
            expected_ids, expected_starts = next_tasks(assigned, solved, attempts, starts, cfg['microbatch'])
            require(len(expected_ids) > 0 and np.array_equal(ids, expected_ids)
                    and np.array_equal(task_starts, expected_starts), f'Batch does not match pending tasks: {path}')
            expected_hashes = dict(config_sha256=cfg_sha, data_sha256=data_sha,
                                   checkpoint_sha256=cfg['checkpoint_sha256'],
                                   worker_sha256=worker_sha, metrics_sha256=metrics_sha)
            for key, expected in expected_hashes.items():
                require(str(z[key]) == expected, f'Changed {key}: {path}')
            require(np.array_equal(z['seeds'], cfg['seed'] + task_starts*100000 + ids), f'Wrong seeds: {path}')
            predictions = z['predictions']
            require(predictions.dtype == np.uint8 and predictions.shape == (cycles, len(ids), cfg.get('variables', 600)), f'Wrong prediction format: {path}')
            require(int(z['executed_cycles']) == cycles, f'Incomplete batch trajectory: {path}')
            require(bool(z['all_logits_finite']) and not bool(z['preflight']), f'Invalid formal batch: {path}')
            checked, violations = check_predictions(predictions, clauses[ids])
            full_ok = checked.any(0)
            require(np.array_equal(checked, z['success']) and np.array_equal(full_ok, z['full_ok']), f'Incorrect saved metrics: {path}')
            require(np.array_equal(violations, z['violated_clauses']), f'Incorrect clause counts: {path}')
            require(np.isfinite(float(z['elapsed_seconds'])) and float(z['elapsed_seconds']) >= 0, f'Bad elapsed time: {path}')
            timing = z['cycle_cuda_seconds']
            require(timing.shape == (cycles,) and np.isfinite(timing).all() and (timing >= 0).all(), f'Bad cycle timing: {path}')
            environment_json = str(z['execution_environment_json'])
            environment_sha = hashlib.sha256(environment_json.encode()).hexdigest()
            require(environment_sha == str(z['execution_environment_sha256']), f'Runtime hash differs: {path}')
            environments[environment_sha] = json.loads(environment_json)
            np.add.at(attempts, ids, 1)
            np.logical_or.at(solved, ids, full_ok)
            records.append(str(path))
    require(np.all(attempts <= starts), 'Task count exceeds start budget')
    return dict(solved=solved, attempts=attempts, environments=environments, records=records)


def check_boolean_graph_predictions(predictions, clauses, edges):
    """Flat200 full-CNF and first-true-color validity, without repair."""
    predictions=np.asarray(predictions)
    require(predictions.dtype==np.bool_ and predictions.ndim==3 and predictions.shape[-1]==600,'Expected Boolean Flat200 predictions')
    full,_=check_predictions(predictions.astype(np.int32)+1,clauses)
    colors=predictions.reshape(*predictions.shape[:2],200,3).argmax(-1)
    graph=np.stack([(colors[:,j,edges[j,:,0]]!=colors[:,j,edges[j,:,1]]).all(-1) for j in range(predictions.shape[1])],axis=1)
    require(not np.any(full & ~graph),'CNF/graph validity disagreement')
    return np.stack([full,graph],axis=-1)

"""Evaluate frozen QuerySAT on the graph-coloring CNFs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from hrm.paths import repo_path, work_path
import hrm_querysat
SOURCE = Path(hrm_querysat.__file__).parent
HERE = work_path('sat/graph_coloring/querysat')
HERE.mkdir(parents=True,exist_ok=True)
DATA = repo_path('data/sat/data')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sparse(indices, shape):
    return tf.sparse.reorder(tf.SparseTensor(np.asarray(indices, np.int64),
                                           np.ones(len(indices), np.float32), shape))


def inputs(clauses):
    # The source pads binary clauses by repeating literals. A repeated literal
    # must remain one binary adjacency edge, as in the original HRM bool mask.
    deduplicated = [list(dict.fromkeys(map(int, row))) for row in clauses]
    edges = [[abs(lit)-1 + (1024 if lit < 0 else 0), ci]
             for ci, clause in enumerate(deduplicated) for lit in clause]
    return dict(adj_matrix=sparse(edges, [2048, len(clauses)]),
                clauses_graph=sparse([[0, i] for i in range(len(clauses))], [1, len(clauses)]),
                variables_graph=sparse([[0, i] for i in range(1024)], [1, 1024]),
                solutions=tf.ragged.constant([[0] * 1024], dtype=tf.int32,
                                             row_splits_dtype=tf.int32))


def verify_predictions(bits, clauses, edges, colors):
    """Independent NumPy checks for every round and head; no head selection."""
    full, original, graph = [], [], []
    for offset in range(0, len(bits), 64):
        chunk = bits[offset:offset+64]
        satisfied = (chunk[..., np.abs(clauses)-1] == (clauses > 0)).any(-1)
        full.append(satisfied.all(-1))
        original.append(satisfied[..., :1792 if colors == 3 else 2304].all(-1))
        decoded = chunk[..., :256*colors].reshape(len(chunk), 8, 256, colors).argmax(-1)
        graph.append((decoded[..., edges[:, 0]] != decoded[..., edges[:, 1]]).all(-1))
    return dict(full_cnf=np.concatenate(full), original_cnf=np.concatenate(original),
                decoded_graph=np.concatenate(graph))


def first_hits(flags, heads):
    result = {}
    for metric, values in flags.items():
        selected = values[np.arange(len(heads)), heads]
        found = np.flatnonzero(selected)
        anyhead = np.argwhere(values)
        result[metric] = dict(selected_round=int(found[0]+1) if len(found) else None,
                              selected_head=int(heads[found[0]]) if len(found) else None,
                              diagnostic_any_head_round=int(anyhead[0, 0]+1) if len(anyhead) else None,
                              per_head_first_round=[int(a[0]+1) if len(a := np.flatnonzero(values[:, h])) else None
                                                    for h in range(8)])
    return result


def original_equivalence(net, sample, seed, rounds, checkpoint):
    """Compare recurrence/readout using identical explicitly seeded query noise."""
    import importlib.util
    comparison_path = HERE/'reference_seeded.py'
    original = (SOURCE/'observed_model.py').read_text()
    original = original.replace('training=None, mask=None):', 'training=None, mask=None, query_seed=None):')
    original = original.replace('                                                                                                   variables_graph)',
                                '                                                                                                   variables_graph, query_seed)')
    original = original.replace('def loop(self, adj_matrix, clause_state, clauses_graph, rounds, training, variables, variables_graph):',
                                'def loop(self, adj_matrix, clause_state, clauses_graph, rounds, training, variables, variables_graph, query_seed=None):')
    original = original.replace('tf.random.normal([n_vars, 4])',
                                '(tf.random.stateless_normal([n_vars, 4], seed=tf.stack([query_seed, step])) if query_seed is not None else tf.random.normal([n_vars, 4]))')
    predict_at = original.index('    @tf.function', original.index('    def train_step'))
    original = original[:predict_at] + original[predict_at:].replace('row_splits_dtype=tf.int32)\n                                  ])',
                              'row_splits_dtype=tf.int32),\n                                  tf.TensorSpec(shape=[], dtype=tf.int32)\n                                  ])', 1)
    original = original.replace('def predict_step(self, adj_matrix, clauses_graph, variables_graph, solutions):',
                                'def predict_step(self, adj_matrix, clauses_graph, variables_graph, solutions, query_seed):')
    original = original.replace('self.call(adj_matrix, clauses_graph, variables_graph, training=False)',
                                'self.call(adj_matrix, clauses_graph, variables_graph, training=False, query_seed=query_seed)')
    # Expose only all-head logits and chosen heads; recurrence remains archived.
    original = original.replace('variables, first_logits, first_hit = self.loop(',
                                'variables, first_logits, first_hit, audit_logits, audit_heads = self.loop(')
    original = original.replace('return last_logits, unsupervised_loss + supervised_loss, step, first_logits, first_hit',
                                'return last_logits, unsupervised_loss + supervised_loss, step, first_logits, first_hit, audit_logits, audit_heads')
    original = original.replace('        for step in tf.range(rounds):',
                                '        audit_logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)\n'
                                '        audit_heads = tf.TensorArray(tf.int32, size=0, dynamic_size=True)\n'
                                '        for step in tf.range(rounds):')
    original = original.replace('            step_losses = step_losses.write(step, logit_loss)',
                                '            audit_logits = audit_logits.write(step, logits)\n'
                                '            audit_heads = audit_heads.write(step, best_logit_map[0])\n'
                                '            step_losses = step_losses.write(step, logit_loss)')
    original = original.replace('return out_logits, step, unsupervised_loss, supervised_loss, clause_state, variables, first_logits, first_hit',
                                'return out_logits, step, unsupervised_loss, supervised_loss, clause_state, variables, first_logits, first_hit, audit_logits.stack(), audit_heads.stack()')
    original = original.replace('predictions, loss, step, first_logits, first_hit = self.call(',
                                'predictions, loss, step, first_logits, first_hit, audit_logits, audit_heads = self.call(')
    original = original.replace('"first_hit": first_hit',
                                '"first_hit": first_hit, "trajectory_logits": audit_logits, "trajectory_heads": audit_heads')
    comparison_path.write_text(original)
    spec = importlib.util.spec_from_file_location('reference_seeded', comparison_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.ObservedQuerySAT(AdaBeliefOptimizer(.0002, beta_1=.6, clip_gradients=True), test_rounds=1)
    reference.call(**{k: sample[k] for k in ['adj_matrix', 'clauses_graph', 'variables_graph']}, training=False)
    restore = tf.train.Checkpoint(model=reference).restore(str(checkpoint))
    restore.assert_existing_objects_matched()
    restore.expect_partial()
    assert len(reference.trainable_variables) == len(net.trainable_variables)
    for expected_weight, actual_weight in zip(reference.trainable_variables, net.trainable_variables):
        np.testing.assert_array_equal(expected_weight.numpy(), actual_weight.numpy())
    reference.test_rounds = rounds
    expected = reference.predict_step(**sample, query_seed=tf.constant(seed, tf.int32))
    observed = net.predict_step(**sample, query_seed=tf.constant(seed, tf.int32))
    expected_logits = expected['trajectory_logits'].numpy()
    observed_logits = observed['trajectory_logits'].numpy()
    common = min(len(expected_logits), len(observed_logits))
    diagnostic = dict(expected_rounds=len(expected_logits), observed_rounds=len(observed_logits),
                      per_round_max_logit_error=np.max(np.abs(expected_logits[:common]-observed_logits[:common]), axis=(1, 2)).tolist(),
                      expected_heads=expected['trajectory_heads'].numpy().tolist(),
                      observed_heads=observed['trajectory_heads'].numpy().tolist())
    atomic(HERE/'equivalence_diagnostics.json', diagnostic)
    for key in ['prediction', 'first_logits', 'first_hit', 'steps_taken', 'trajectory_logits', 'trajectory_heads']:
        np.testing.assert_allclose(expected[key].numpy(), observed[key].numpy(), rtol=0, atol=0)
    return dict(exact_match=True, verification_device='CPU; GPU sparse reductions are nondeterministic in TensorFlow 2.15',
                query_noise='Identical stateless Gaussian, keyed by [trajectory_seed, round]',
                reference_change='Explicitly seeded same-distribution noise and read-only per-round logits/head capture',
                rounds_compared=len(observed['trajectory_heads']),
                keys=['prediction', 'first_logits', 'first_hit', 'steps_taken', 'trajectory_logits', 'trajectory_heads'])


def main():
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=repo_path('data/sat/graph_coloring/querysat.json'))
    parser.add_argument('--seed-base', type=int, required=True)
    parser.add_argument('--starts', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=2048)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--case-offset', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--colors', type=int, nargs='+', default=[3, 4], choices=[3, 4])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-label', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['status'] == 'ready', 'Run configuration is not finalized'
    DATA = repo_path(config['data_root']).resolve()
    assert DATA == repo_path('data/sat/data').resolve()
    assert 1 <= args.starts <= 64 and args.rounds > 0
    assert args.seed_base == config['seed_base'] and config['seed_stride'] == 100000
    assert 0 <= args.seed_base and args.seed_base + 100000*(args.starts-1) + 999 < 2**31
    if args.preflight:
        assert args.starts <= config['starts'] and args.rounds <= config['rounds']
    else:
        assert (args.starts, args.rounds, args.shards) == (config['starts'], config['rounds'], config['shards'])
    model_config = next(v for v in config['models'].values() if repo_path(v['checkpoint']).resolve() == args.checkpoint.resolve())
    assert 0 <= args.case_offset < 1000 and 1 <= args.limit <= 1000-args.case_offset
    assert 1 <= args.shards <= 1000 and 0 <= args.shard < args.shards
    case_ids = [i for i in range(args.case_offset, args.case_offset+args.limit) if i % args.shards == args.shard]
    assert case_ids
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=args.resume)
    gpus = tf.config.list_physical_devices('GPU')
    assert len(gpus) == 1, 'Expose exactly the allocated GPU via CUDA_VISIBLE_DEVICES'
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    files = [Path(str(args.checkpoint)+suffix) for suffix in ['.index', '.data-00000-of-00001']]
    assert all(path.exists() for path in files)
    assert digest(files[0]) == model_config['index_sha'] and digest(files[1]) == model_config['data_sha']
    for k in args.colors:
        source_manifest = json.loads((DATA/f'k{k}'/'data_manifest.json').read_text())
        assert source_manifest['sha256'] == digest(DATA/f'k{k}'/'instances.npz')
    manifest = dict(model=args.model_label, run_config=str(args.config.resolve()), run_config_sha256=digest(args.config),
                    seed_base=args.seed_base, seed_stride=config['seed_stride'],
                    checkpoint=str(args.checkpoint), checkpoint_sha256={p.name: digest(p) for p in files},
                    source_model=str(SOURCE / 'observed_model.py'), source_model_sha256=digest(SOURCE / 'observed_model.py'),
                    observer_sha256=digest(SOURCE / 'trajectory_model.py'), evaluator_sha256=digest(__file__),
                    inference_code_sha256={str(p.relative_to(SOURCE)): digest(p) for p in SOURCE.rglob('*.py')},
                    inputs={str(DATA/f'k{k}'/'instances.npz'): digest(DATA/f'k{k}'/'instances.npz') for k in args.colors},
                    source_data_manifests={str(DATA/f'k{k}'/'data_manifest.json'): digest(DATA/f'k{k}'/'data_manifest.json') for k in args.colors},
                    tensorflow=tf.__version__, python=sys.executable, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    starts=args.starts, rounds=args.rounds, cases_per_color=len(case_ids), case_ids=case_ids,
                    shard=args.shard, shards=args.shards, case_offset=args.case_offset,
                    seed_rule=f'{args.seed_base} + 100000*zero_based_start + zero_based_case; same for k3/k4',
                    query_noise='tf.random.stateless_normal keyed by [trajectory_seed, zero_based_round]; Gaussian distribution unchanged, explicit seeds avoid tf.function trace capture',
                    reproducibility='Explicit RNG mapping; original TensorFlow 2.15 GPU sparse matmul is not bitwise deterministic. Preserve all raw predictions for exact result audit.',
                    head_selection='Original QuerySAT argmin over per-head full-CNF softplus_mixed_loss; no reference labels and no discrete-success selection',
                    output_bits='round(sigmoid(logits)) as in original QuerySAT verifier',
                    early_stop='Original singleton selected-head full-CNF success ends trajectory; skip later starts after full-CNF success',
                    main_metrics='Any round of the original selected head; other heads are diagnostics only',
                    preprocessing='Deduplicate repeated literals within each source clause; preserve order, variables and clauses',
                    decoder='First true color, default zero; no repair or search',
                    raw_predictions='NPZ bits_packed[executed_round,head,128 bytes], numpy bitorder=big; selected_head stored per round',
                    preflight=args.preflight, started_unix=time.time())
    if (args.out/'manifest.json').exists():
        assert args.resume
        previous = json.loads((args.out/'manifest.json').read_text())
        for key in ['model', 'run_config', 'run_config_sha256', 'seed_base', 'seed_stride', 'checkpoint', 'checkpoint_sha256', 'source_model_sha256', 'observer_sha256',
                    'evaluator_sha256', 'inference_code_sha256', 'inputs', 'source_data_manifests', 'starts',
                    'rounds', 'case_ids', 'shard', 'shards', 'seed_rule', 'query_noise', 'head_selection', 'preflight']:
            assert previous[key] == manifest[key], f'Resume protocol mismatch: {key}'
        manifest['execution_sessions'] = previous.get('execution_sessions', [])
        manifest['started_unix'] = previous['started_unix']
    manifest.setdefault('execution_sessions', []).append(dict(pid=os.getpid(), started_unix=time.time(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    atomic(args.out/'manifest.json', manifest)
    sample_arrays = np.load(DATA/f'k{args.colors[0]}'/'instances.npz')
    sample = inputs(sample_arrays['clauses'][case_ids[0]])
    tf.keras.utils.set_random_seed(args.seed_base)
    if args.preflight:
        with tf.device('/CPU:0'):
            audit_sample = inputs(sample_arrays['clauses'][case_ids[0]])
            audit_net = ObservedQuerySAT(AdaBeliefOptimizer(.0002, beta_1=.6, clip_gradients=True), test_rounds=1)
            audit_net.audit_logits = True
            audit_net.call(**{k: audit_sample[k] for k in ['adj_matrix', 'clauses_graph', 'variables_graph']}, training=False)
            audit_restore = tf.train.Checkpoint(model=audit_net).restore(str(args.checkpoint))
            audit_restore.assert_existing_objects_matched()
            audit_restore.expect_partial()
            audit_net.test_rounds = min(args.rounds, 64)
            atomic(args.out/'observer_equivalence.json', original_equivalence(audit_net, audit_sample, args.seed_base, min(args.rounds, 64), args.checkpoint))
    net = ObservedQuerySAT(AdaBeliefOptimizer(.0002, beta_1=.6, clip_gradients=True), test_rounds=1)
    net.call(**{k: sample[k] for k in ['adj_matrix', 'clauses_graph', 'variables_graph']}, training=False)
    restore = tf.train.Checkpoint(model=net).restore(str(args.checkpoint))
    restore.assert_existing_objects_matched()
    restore.expect_partial()
    net.test_rounds = args.rounds
    trace_begin = time.perf_counter()
    net.predict_step.get_concrete_function()
    manifest['graph_trace_seconds'] = time.perf_counter()-trace_begin
    manifest['checkpoint_existing_objects_matched'] = True
    manifest['trainable_parameters'] = sum(int(np.prod(v.shape)) for v in net.trainable_variables)
    manifest['gpu_details'] = tf.config.experimental.get_device_details(gpus[0])
    atomic(args.out/'manifest.json', manifest)
    wall_start = time.perf_counter()
    all_cases = []
    for k in args.colors:
        destination = args.out/f'k{k}'
        destination.mkdir(exist_ok=args.resume)
        arrays = np.load(DATA/f'k{k}'/'instances.npz')
        for case in case_ids:
            clauses, edges = arrays['clauses'][case], arrays['edges'][case]
            sample = inputs(clauses)
            record = dict(colors=k, case=case, starts=[], first_hits={}, elapsed_seconds=None,
                          clauses_sha256=hashlib.sha256(clauses.tobytes()).hexdigest(),
                          graph_edges_sha256=hashlib.sha256(edges.tobytes()).hexdigest(), complete=False)
            case_path = destination/f'case{case:04d}.json'
            if case_path.exists():
                assert args.resume
                previous = json.loads(case_path.read_text())
                assert all(previous[key] == record[key] for key in ['colors', 'case', 'clauses_sha256', 'graph_edges_sha256'])
                for saved in previous['starts']:
                    assert digest(destination/saved['raw_predictions']) == saved['raw_sha256']
                record = previous
                if record['complete']:
                    all_cases.append(record)
                    continue
            case_begin = time.perf_counter()
            previous_elapsed = record['elapsed_seconds'] or 0.0
            for start in range(len(record['starts']), args.starts):
                if 'full_cnf' in record['first_hits']:
                    break
                seed = args.seed_base + 100000*start + case
                raw_path = destination/f'case{case:04d}_start{start:02d}.npz'
                recovered = raw_path.exists()
                if recovered:
                    assert args.resume
                    with np.load(raw_path) as saved:
                        assert int(saved['seed']) == seed
                        bits = np.unpackbits(saved['bits_packed'], axis=-1, bitorder='big').astype(bool)
                        heads, losses = saved['selected_head'], saved['head_selection_loss']
                        model_hit, inference_seconds = int(saved['model_first_hit']), float(saved['inference_seconds'])
                else:
                    tf.keras.utils.set_random_seed(seed)
                    begin = time.perf_counter()
                    output = net.predict_step(**sample, query_seed=tf.constant(seed, tf.int32))
                    # Converting results synchronizes GPU execution before timing.
                    bits = output['trajectory_bits'].numpy().transpose(0, 2, 1)
                    assert 'GPU:0' in output['trajectory_bits'].device, output['trajectory_bits'].device
                    heads = output['trajectory_heads'].numpy()
                    losses = output['trajectory_losses'].numpy()
                    inference_seconds = time.perf_counter()-begin
                    model_hit = int(output['first_hit'].numpy()[0])
                    assert len(bits) == int(output['steps_taken'].numpy())+1
                flags = verify_predictions(bits, clauses, edges, k)
                hits = first_hits(flags, heads)
                assert model_hit == (hits['full_cnf']['selected_round'] or 0)
                witnesses = {}
                for metric, hit in hits.items():
                    if hit['selected_round']:
                        assignment = bits[hit['selected_round']-1, hit['selected_head']]
                        witnesses[metric] = assignment.astype(np.uint8).tolist()
                        record['first_hits'].setdefault(metric, dict(start=start+1, round=hit['selected_round'], head=hit['selected_head']))
                if not recovered:
                    temporary_raw = raw_path.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary_raw, bits_packed=np.packbits(bits, axis=-1, bitorder='big'),
                                        selected_head=heads, head_selection_loss=losses, seed=seed,
                                        model_first_hit=model_hit, inference_seconds=inference_seconds, **flags)
                    temporary_raw.replace(raw_path)
                result = dict(start=start+1, seed=seed, rounds_executed=len(bits), inference_seconds=inference_seconds,
                              first_hits=hits, witnesses=witnesses, raw_predictions=raw_path.name,
                              recovered_committed_raw=recovered,
                              raw_sha256=digest(raw_path), nonfinite_head_losses=int((~np.isfinite(losses)).sum()),
                              nonfinite_selected_losses=int((~np.isfinite(losses[np.arange(len(heads)), heads])).sum()))
                record['starts'].append(result)
                record['elapsed_seconds'] = previous_elapsed+time.perf_counter()-case_begin
                atomic(case_path, record)
                atomic(args.out/'progress.json', dict(cases_done=len(all_cases), elapsed_seconds=time.perf_counter()-wall_start,
                       last_color=k, last_case=case, last_start=start+1,
                       rounds_executed=sum(s['rounds_executed'] for r in all_cases+[record] for s in r['starts']),
                       inference_seconds=sum(s['inference_seconds'] for r in all_cases+[record] for s in r['starts'])))
                if hits['full_cnf']['selected_round']:
                    break
            record['elapsed_seconds'] = previous_elapsed+time.perf_counter()-case_begin
            record['complete'] = True
            atomic(case_path, record)
            all_cases.append(record)
            progress = dict(cases_done=len(all_cases), elapsed_seconds=time.perf_counter()-wall_start,
                            last_color=k, last_case=case, selected_full_cnf_solved=sum('full_cnf' in r['first_hits'] for r in all_cases),
                            rounds_executed=sum(s['rounds_executed'] for r in all_cases for s in r['starts']),
                            inference_seconds=sum(s['inference_seconds'] for r in all_cases for s in r['starts']))
            atomic(args.out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    summary = dict(completed=True, preflight=args.preflight, elapsed_seconds=time.perf_counter()-wall_start,
                   rounds_executed=sum(s['rounds_executed'] for r in all_cases for s in r['starts']),
                   inference_seconds=sum(s['inference_seconds'] for r in all_cases for s in r['starts']),
                   results={str(k): dict(n=sum(r['colors']==k for r in all_cases),
                                        **{metric: sum(r['colors']==k and metric in r['first_hits'] for r in all_cases)
                                           for metric in ['full_cnf', 'original_cnf', 'decoded_graph']}) for k in args.colors})
    atomic(args.out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    # Resolve imports through archived TF2.15 site-packages in the launch command.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '4')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    import numpy as np
    import tensorflow as tf
    from hrm_querysat.optimization.AdaBelief import AdaBeliefOptimizer
    from hrm_querysat.trajectory_model import ObservedQuerySAT
    main()

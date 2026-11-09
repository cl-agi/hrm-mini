"""QuerySAT E43 evaluation with HF graph inputs and an explicit trained checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from hrm.paths import repository_root
from hrm.analyse.sat.graph_coloring.data import load_data


def inputs(clauses, variables):
    import tensorflow as tf
    def sparse(indices, shape):
        return tf.sparse.reorder(tf.SparseTensor(np.asarray(indices, np.int64), np.ones(len(indices), np.float32), shape))
    edges = [[abs(lit)-1 + (variables if lit<0 else 0), i]
             for i,row in enumerate(clauses) for lit in dict.fromkeys(map(int,row))]
    return dict(adj_matrix=sparse(edges,[2*variables,len(clauses)]),
                clauses_graph=sparse([[0,i] for i in range(len(clauses))],[1,len(clauses)]),
                variables_graph=sparse([[0,i] for i in range(variables)],[1,variables]),
                solutions=tf.ragged.constant([[0]*variables],dtype=tf.int32,row_splits_dtype=tf.int32))


def score_selected(bits, heads, clauses, edges, colors, flat=False):
    if bits.ndim!=3 or heads.shape!=(len(bits),) or np.any(heads<0) or np.any(heads>=bits.shape[1]):
        raise ValueError('Invalid QuerySAT trajectory/head dimensions')
    selected=bits[np.arange(len(bits)),heads]
    vertices=200 if flat else 256
    result={key:False for key in (('full_cnf','decoded_graph') if flat else ('full_cnf','original_cnf','decoded_graph'))}
    for offset in range(0,len(selected),64):
        chunk=selected[offset:offset+64]
        satisfied=(chunk[:,np.abs(clauses)-1]==(clauses>0)).any(-1)
        result['full_cnf'] |= bool(satisfied.all(-1).any())
        if not flat:result['original_cnf'] |= bool(satisfied[:,:1792 if colors==3 else 2304].all(-1).any())
        decoded=chunk[:,:vertices*colors].reshape(-1,vertices,colors).argmax(-1)
        result['decoded_graph'] |= bool((decoded[:,edges[:,0]]!=decoded[:,edges[:,1]]).all(-1).any())
    return result


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,help='TensorFlow prefix or directory; QuerySAT weights are not in the HRM release')
    p.add_argument('--dataset-path',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--benchmark',choices=('generated','flat200'),default='generated')
    p.add_argument('--colors',type=int,nargs='+',choices=(3,4))
    p.add_argument('--rounds',type=int,default=2048)
    p.add_argument('--starts',type=int,default=4)
    p.add_argument('--seed-base',type=int,default=1793018317,help='Archived same-seed evaluation; override for a new comparison')
    p.add_argument('--limit',type=int)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.checkpoint is None:p.error('Supply --checkpoint: this release includes HRM weights, not the separately trained QuerySAT baselines.')
    colors=a.colors or ([3] if a.benchmark=='flat200' else [3,4])
    if len(set(colors))!=len(colors) or (a.benchmark=='flat200' and colors!=[3]):p.error('Invalid color selection')
    if a.rounds<1 or not 1<=a.starts<=64 or (a.limit is not None and a.limit<1):p.error('Invalid inference budget')
    if not 0<=a.seed_base<2**31-1-100000*a.starts-1000:p.error('Seed range exceeds int32')
    import tensorflow as tf
    if a.device=='cpu':tf.config.set_visible_devices([], 'GPU')
    elif not tf.config.list_physical_devices('GPU'):raise RuntimeError('CUDA requested but no TensorFlow GPU is available')
    prefix=str(a.checkpoint.expanduser().resolve())
    if a.checkpoint.is_dir():prefix=tf.train.latest_checkpoint(prefix)
    if not prefix or not Path(prefix+'.index').is_file():raise FileNotFoundError(f'Missing TensorFlow checkpoint: {a.checkpoint}')
    files=[Path(prefix+'.index')]+sorted(Path(prefix).parent.glob(Path(prefix).name+'.data-*'))
    if len(files)<2:raise FileNotFoundError('Missing checkpoint data shards')
    data={k:load_data(k,a.dataset_path,a.benchmark) for k in colors}
    out=a.output or repository_root()/'exp/sat/records/graph_coloring/querysat'/a.benchmark
    import hrm_querysat
    source=Path(hrm_querysat.__file__).parent
    config=dict(checkpoint_sha256={f.name:digest(f) for f in files},checkpoint=prefix,
                source_sha256={str(f.relative_to(source)):digest(f) for f in source.rglob('*.py')},
                evaluator_sha256=digest(__file__),data_loader_sha256=digest(Path(__file__).with_name('data.py')),
                benchmark=a.benchmark,colors=colors,rounds=a.rounds,starts=a.starts,seed_base=a.seed_base,
                seed_rule='seed_base + 100000 * zero_based_start + zero_based_case; shared across colors',
                device=a.device,tensorflow=tf.__version__,
                inputs={str(k):dict(ordered_sha256=d['ordered_sha256'],ids=d['ids'][:a.limit]) for k,d in data.items()},
                head_selection='argmin full-CNF softplus_mixed_loss per round; no reference labels',
                stopping='Stop subsequent starts after selected-head full-CNF success',
                scope='Checkpoint evaluation; limited budgets/subsets do not reproduce full-dataset performance')
    out.mkdir(parents=True,exist_ok=True)
    path=out/'resolved_config.json'
    if path.exists():
        if json.loads(path.read_text())!=config:raise ValueError('Run configuration changed; choose a new output directory')
    else:path.write_text(json.dumps(config,indent=2)+'\n')
    if a.prepare_only:return
    with (out/'started.json').open('x') as stream:json.dump(dict(started_unix=time.time()),stream)
    from hrm_querysat.trajectory_model import ObservedQuerySAT
    from hrm_querysat.optimization.AdaBelief import AdaBeliefOptimizer
    net=ObservedQuerySAT(AdaBeliefOptimizer(.0002,beta_1=.6,clip_gradients=True),test_rounds=1)
    sample=inputs(data[colors[0]]['clauses'][0],data[colors[0]]['variables'])
    net.call(**{k:v for k,v in sample.items() if k!='solutions'},training=False)
    status=tf.train.Checkpoint(model=net).restore(prefix)
    status.assert_existing_objects_matched()
    status.expect_partial()  # Training optimizer slots are intentionally not used in inference.
    net.test_rounds=a.rounds
    results={}
    begin=time.perf_counter()
    for k,d in data.items():
        count=min(a.limit or len(d['ids']),len(d['ids']))
        totals=None
        executed=0
        for case in range(count):
            success=None
            sample=inputs(d['clauses'][case],d['variables'])
            for start in range(a.starts):
                seed=a.seed_base+100000*start+case
                tf.keras.utils.set_random_seed(seed)
                output=net.predict_step(**sample,query_seed=tf.constant(seed,tf.int32))
                bits=output['trajectory_bits'].numpy().transpose(0,2,1)
                heads=output['trajectory_heads'].numpy().astype(np.int64)
                flags=score_selected(bits,heads,d['clauses'][case],d['edges'][case],k,a.benchmark=='flat200')
                success=flags if success is None else {key:success[key] or flags[key] for key in flags}
                executed+=len(bits)
                if success['full_cnf']:break
            if totals is None:totals={key:0 for key in success}
            for key,value in success.items():totals[key]+=int(value)
        results[str(k)]=dict(n=count,solved=totals,accuracy={key:v/count for key,v in totals.items()},executed_rounds=executed)
    (out/'results.json').write_text(json.dumps(dict(results=results,wall_seconds=time.perf_counter()-begin),indent=2)+'\n')


if __name__=='__main__':main()

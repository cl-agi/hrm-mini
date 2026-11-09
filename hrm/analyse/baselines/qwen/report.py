"""Paired puzzle-cluster intervals from new E15 evaluation records, without model calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def read_condition(directory, budget):
    from hrm_qwen.cot.evaluate import load_plan, score
    from hrm_qwen.cot.run_config import file_hash
    directory=Path(directory)
    score(directory,write=False)  # validate full planned coverage and completion hashes first
    plan,questions=load_plan(directory)
    cfg=['test_hard',0,budget]
    if cfg not in plan['configs']:raise ValueError(f'Missing unhinted test condition {cfg}')
    inputs={r['index']:(r['question'],r['answer'],r['prompt']) for r in questions if r['config']==cfg}
    values={}
    hashes={}
    for rank in range(plan['dp_size']):
        path=directory/f'rank{rank}.jsonl'
        hashes[path.name]=file_hash(path)
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['config']==cfg:
                values[row['index']]=sum(int(s['exact']) for s in row['samples'])
    return inputs,values,plan['n'],dict(
        plan_sha256=file_hash(directory/'resolved_config.json'),
        questions_sha256=plan['questions_sha256'],shards_sha256=hashes,
        model=plan['model'],budget=budget,seed=plan['seed'],dp_size=plan['dp_size'],
        temperature=plan['temperature'],top_p=plan['top_p'],top_k=plan['top_k'])


def cluster_statistics(base,student,replicates,seed):
    base=np.asarray(base,dtype=float);student=np.asarray(student,dtype=float)
    if base.ndim!=1 or base.shape!=student.shape or not len(base):
        raise ValueError('Expected paired nonempty puzzle vectors')
    if not np.isfinite(base).all() or not np.isfinite(student).all() or np.any((base<0)|(base>1)|(student<0)|(student>1)):
        raise ValueError('Puzzle correctness fractions must be finite and within [0,1]')
    if replicates<2:raise ValueError('At least two bootstrap replicates required')
    n=len(base);rng=np.random.default_rng(seed);draws=[]
    vectors=np.stack([base,student,student-base],axis=1)
    # Bound memory while retaining the original multinomial cluster bootstrap.
    for start in range(0,replicates,256):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(256,replicates-start))/n
        draws.append(weights@vectors)
    ci=np.quantile(np.concatenate(draws),[.025,.975],axis=0)*100
    names=('instruct','teacher_cot_sft','paired_sft_minus_instruct')
    return {name:dict(mean_percent=float(vectors[:,i].mean()*100),
                     ci95_low_percent=float(ci[0,i]),ci95_high_percent=float(ci[1,i]))
            for i,name in enumerate(names)}


def report(base_dir,student_dir,output,base_budget=16384,student_budget=65536,replicates=10000,seed=None):
    bq,bv,bn,bmeta=read_condition(base_dir,base_budget)
    sq,sv,sn,smeta=read_condition(student_dir,student_budget)
    if bq!=sq:raise ValueError('Conditions do not contain the same indexed puzzles, prompts and reference answers')
    if bn!=sn:raise ValueError('Conditions use different numbers of outputs per puzzle')
    if set(bv)!=set(sv) or set(bv)!=set(bq):raise ValueError('Condition coverage mismatch')
    output=Path(output)
    old=json.loads(output.read_text()) if output.exists() else None
    seed=seed if seed is not None else (old['analysis_seed'] if old else int(time.time()))
    provenance=dict(analysis_seed=seed,bootstrap_replicates=replicates,
                    analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy_version=np.__version__,
                    inputs=dict(instruct=bmeta,teacher_cot_sft=smeta))
    if old and any(old[k]!=v for k,v in provenance.items()):
        raise ValueError('Analysis inputs/settings changed; select a new output file')
    indices=sorted(bq);base=np.array([bv[i]/bn for i in indices]);student=np.array([sv[i]/sn for i in indices])
    result=dict(**provenance,method='paired puzzle-cluster percentile bootstrap',
        n_puzzles=len(indices),outputs_per_puzzle_per_condition=bn,
        conditions=cluster_statistics(base,student,replicates,seed),
        counts=dict(instruct_correct=sum(bv.values()),sft_correct=sum(sv.values()),
                    outputs_per_condition=len(indices)*bn),
        difference_unit='percentage points')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    from hrm.paths import repository_root
    parser.add_argument('--records',type=Path,default=Path(os.environ.get('HRM_ANALYSIS_RECORDS',str(repository_root()/'exp/sudoku/records/qwen'))))
    parser.add_argument('--base-records',type=Path)
    parser.add_argument('--sft-records',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--base-budget',type=int,default=16384)
    parser.add_argument('--sft-budget',type=int,default=65536)
    parser.add_argument('--replicates',type=int,default=10000)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    result=report(args.base_records or args.records/'base-eval',args.sft_records or args.records/'sft-eval',
                  args.output or args.records/'summary.json',args.base_budget,args.sft_budget,args.replicates,args.seed)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

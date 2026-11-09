"""Prepare, execute and score the E15 teacher-CoT/base-model diagnostic."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from hrm_qwen.cot.run_config import file_hash
from hrm_qwen.cot.scoring import eval_rows, train_rows, grade, summarize

BASE_MODEL = 'Qwen/Qwen3.5-4B'
BASE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)


def source_hashes():
    root = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(root.glob('*.py'))}


def model_identity(model_path, model, revision):
    if model_path:
        root = Path(model_path).resolve(strict=True)
        if not root.is_dir() or not (root/'config.json').is_file():
            raise ValueError('Model path must be a saved Transformers checkpoint directory')
        files = sorted(p for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts)
        if not any(p.suffix in ('.safetensors', '.bin') for p in files):
            raise ValueError('No model weight files in checkpoint directory')
        return dict(path=str(root), files={str(p.relative_to(root)):file_hash(p) for p in files})
    revision = revision or (BASE_REVISION if model == BASE_MODEL else None)
    if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Use a full HF commit SHA for --revision')
    return dict(repo=model, revision=revision)


def prepare(args):
    root = Path(args.records)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Use a fresh records directory: {root}')
    if args.dp_size < 1 or args.n < 1 or args.tp < 1 or args.offset < 0:
        raise ValueError('Positive parallelism/sample counts and nonnegative offset required')
    if not 0 < args.top_p <= 1 or args.temperature < 0 or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError('Invalid sampling or memory settings')
    seed = args.seed if args.seed is not None else int(time.time())
    if seed < 0 or args.max_num_seqs < 1 or args.top_k == 0 or args.top_k < -1:
        raise ValueError('Invalid seed, max-num-seqs or top-k')
    jobs, configs = [], []
    for value in args.config:
        parts = value.split(':')
        if len(parts) != 4:
            raise ValueError('Config must be split:empties:budget:count')
        split = parts[0]; empties, budget, count = map(int,parts[1:])
        if split not in ('test_hard','train') or not 0 <= empties <= 81 or budget < 1 or count < 1:
            raise ValueError('Invalid diagnostic config')
        cfg = [split, empties, budget]
        if cfg in configs:
            raise ValueError('Duplicate diagnostic config')
        configs.append(cfg)
        if split == 'test_hard':
            if empties != 0:
                raise ValueError('test_hard requires empties=0 (unmodified test puzzles)')
            rows = eval_rows(args.data_dir, count)
        else:
            rows = train_rows(args.data_dir,seed,range(args.offset,args.offset+count),empties)
        if len(rows) != count:
            raise ValueError('Requested cohort exceeds available split')
        if args.dp_size > count:
            raise ValueError('dp-size exceeds the number of puzzles in a config')
        for i,row in enumerate(rows):
            jobs.append(dict(config=cfg, rank=i % args.dp_size, **row))
    model = model_identity(args.model_path,args.model,args.revision)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'questions.jsonl').open('x') as stream:
        for row in jobs:stream.write(json.dumps(row)+'\n')
    from hrm import hub
    data_source = ({'directory': str(Path(args.data_dir).resolve()), 'kind': 'explicit CSV'}
                   if args.data_dir else {'repo': 'cl-agi/sudoku', 'revision': hub.RELEASES['sudoku'][1]})
    plan = dict(data_source=data_source,seed=seed,model=model,dp_size=args.dp_size,tp=args.tp,n=args.n,
                temperature=args.temperature,top_p=args.top_p,top_k=args.top_k,
                gpu_memory_utilization=args.gpu_memory_utilization,max_num_seqs=args.max_num_seqs,
                configs=configs,rows=len(jobs),questions_sha256=file_hash(root/'questions.jsonl'),
                code_sha256=source_hashes(),data_dir=args.data_dir,
                selection='first count test rows; indexed augmentation and hint curriculum for train')
    write_json(root/'resolved_config.json',plan)
    return plan


def load_plan(root):
    root = Path(root)
    plan = json.loads((root/'resolved_config.json').read_text())
    if file_hash(root/'questions.jsonl') != plan['questions_sha256']:
        raise ValueError('Prepared questions changed')
    if source_hashes() != plan['code_sha256']:
        raise ValueError('Evaluation source changed; prepare a new run')
    rows = [json.loads(line) for line in (root/'questions.jsonl').read_text().splitlines()]
    if len(rows) != plan['rows']:
        raise ValueError('Prepared cohort count mismatch')
    return plan,rows


def execute(records,rank):
    root=Path(records);plan,rows=load_plan(root)
    if not 0 <= rank < plan['dp_size']:raise ValueError('Rank outside prepared dp-size')
    identity=plan['model']
    if 'path' in identity and model_identity(identity['path'],None,None) != identity:
        raise ValueError('Checkpoint contents changed')
    with (root/f'rank{rank}.started').open('x') as stream:
        stream.write(json.dumps({'plan_sha256':file_hash(root/'resolved_config.json')}))
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    model=identity.get('path',identity.get('repo'))
    revision=identity.get('revision')
    tok=AutoTokenizer.from_pretrained(model,revision=revision)
    jobs=[]
    for row in rows:
        if row['rank'] != rank:continue
        text=tok.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        jobs.append((row,tok(text,add_special_tokens=False)['input_ids']))
    llm=LLM(model=model,revision=revision,tokenizer_revision=revision,
            tensor_parallel_size=plan['tp'],dtype='bfloat16',
            max_model_len=max(len(p) for _,p in jobs)+max(c[2] for c in plan['configs'])+8,
            gpu_memory_utilization=plan['gpu_memory_utilization'],seed=plan['seed']+rank,
            enable_prefix_caching=True,max_num_seqs=plan['max_num_seqs'])
    with (root/f'rank{rank}.jsonl').open('x') as stream:
        for cfg in plan['configs']:
            mine=[(r,p) for r,p in jobs if r['config']==cfg]
            params=SamplingParams(n=plan['n'],max_tokens=cfg[2],temperature=plan['temperature'],
                                  top_p=plan['top_p'],top_k=plan['top_k'],seed=plan['seed']+rank)
            outs=llm.generate([{'prompt_token_ids':p} for _,p in mine],params,use_tqdm=True)
            if len(outs)!=len(mine):raise ValueError('Inference dropped puzzles')
            for (row,_),out in zip(mine,outs):
                if len(out.outputs)!=plan['n']:raise ValueError('Inference dropped samples')
                samples=[]
                for sample in out.outputs:
                    g=grade(sample.text,row['question'],row['answer'])
                    g.update(n_tokens=len(sample.token_ids),truncated=sample.finish_reason=='length')
                    samples.append(g)
                stream.write(json.dumps(dict(config=cfg,index=row['index'],samples=samples))+'\n')
    write_json(root/f'rank{rank}.done.json',dict(plan_sha256=file_hash(root/'resolved_config.json'),
                result_sha256=file_hash(root/f'rank{rank}.jsonl')))


def score(records, *, write=True):
    root=Path(records);plan,questions=load_plan(root)
    grouped={}
    for rank in range(plan['dp_size']):
        path=root/f'rank{rank}.jsonl'
        done=json.loads((root/f'rank{rank}.done.json').read_text())
        if done != dict(plan_sha256=file_hash(root/'resolved_config.json'),result_sha256=file_hash(path)):
            raise ValueError('Shard or plan changed after completion')
        expected={(tuple(r['config']),r['index']) for r in questions if r['rank']==rank}
        seen=set()
        for line in path.read_text().splitlines():
            row=json.loads(line);key=(tuple(row['config']),row['index'])
            if key not in expected or key in seen:raise ValueError('Duplicate or unexpected puzzle')
            seen.add(key)
            if len(row['samples'])!=plan['n']:raise ValueError('Sample count mismatch')
            for s in row['samples']:
                for k in ('exact','closed','parsed','legal','truncated'):
                    if type(s[k]) is not bool:raise ValueError('Malformed grade')
                if type(s['n_tokens']) is not int or s['n_tokens']<0:raise ValueError('Invalid token count')
                for k in ('cell_acc','clues_kept'):
                    if not 0 <= s[k] <= 1:raise ValueError('Invalid grade fraction')
            grouped.setdefault(key[0],[]).append(row)
        if seen!=expected:raise ValueError('Incomplete puzzle coverage')
    report={}
    for cfg,rows in sorted(grouped.items()):
        flat=[s for r in rows for s in r['samples']]
        metrics=summarize(flat,[s['n_tokens'] for s in flat])
        metrics['pass@1']=metrics.pop('exact_match')
        metrics[f"pass@{plan['n']}"]=float(np.mean([any(s['exact'] for s in r['samples']) for r in rows]))
        metrics['n_puzzles']=len(rows)
        metrics['truncated_rate']=float(np.mean([s['truncated'] for s in flat]))
        report[f'{cfg[0]}/empties{cfg[1]}/budget{cfg[2]}']=metrics
    # Aggregates can be regenerated; source shards are immutable.
    if write:
        (root/'metrics.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--records',default='exp/sudoku/records/qwen/evaluation')
    prep.add_argument('--data-dir',default=None)
    group=prep.add_mutually_exclusive_group()
    group.add_argument('--model',default=BASE_MODEL)
    group.add_argument('--model-path')
    prep.add_argument('--revision')
    prep.add_argument('--config',action='append',required=True,help='split:empties:budget:count')
    prep.add_argument('--seed',type=int)
    prep.add_argument('--offset',type=int,default=0)
    prep.add_argument('--dp-size',type=int,default=1)
    prep.add_argument('--tp',type=int,default=1)
    prep.add_argument('--n',type=int,default=4)
    prep.add_argument('--temperature',type=float,default=1.)
    prep.add_argument('--top-p',type=float,default=.95)
    prep.add_argument('--top-k',type=int,default=20)
    prep.add_argument('--gpu-memory-utilization',type=float,default=.85)
    prep.add_argument('--max-num-seqs',type=int,default=128)
    for name in ('run','score'):
        parser=sub.add_parser(name);parser.add_argument('--records',default='exp/sudoku/records/qwen/evaluation')
        if name=='run':parser.add_argument('--rank',type=int,required=True)
    args=p.parse_args()
    if args.action=='prepare':print(json.dumps(prepare(args),indent=2))
    elif args.action=='run':execute(args.records,args.rank)
    else:print(json.dumps(score(args.records),indent=2))


if __name__=='__main__':main()

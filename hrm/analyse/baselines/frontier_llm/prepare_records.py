"""Prepare newly sampled public E14 inputs and frozen requests without API calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from hrm import hub
from hrm.paths import repository_root
from hrm.analyse.baselines.frontier_llm import frozen_code
from hrm.analyse.baselines.frontier_llm.engine import build_payload, context_admission


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(output,profiles_path,count=100,seed=None,paths=None):
    output=Path(output);profiles_path=Path(profiles_path);paths=paths or {}
    if count<1:raise ValueError('count must be positive')
    if output.exists():raise FileExistsError('Use a new output directory; prepared inputs are immutable')
    seed=int(time.time()) if seed is None else seed
    if not 0<=seed<2**63:raise ValueError('seed must be a nonnegative int64')
    profiles=json.loads(profiles_path.read_text())
    if not profiles:raise ValueError('No model profiles')
    # Only known transport fields are accepted; do not copy arbitrary metadata/credentials.
    allowed={'max_completion_tokens','reasoning','provider','temperature','top_p','seed'}
    for model,profile in profiles.items():
        if not isinstance(model,str) or not model or set(profile['call_settings'])-allowed:raise ValueError('Invalid profile/request fields')
        build_payload('preflight',model,profile)
    questions=[];requests=[];sources={}
    for task in ('sudoku','maze'):
        path=hub.puzzle_data(task,paths.get(task),split='test',size=30)
        with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
        if count>len(rows):raise ValueError(f'{task}: requested {count}, only {len(rows)} available')
        indices=random.Random(f'{seed}:{task}').sample(range(len(rows)),count)
        formatter=frozen_code.definitions('output_contract/dataset/'+task+'.py',{'get_prompt','format_prompt'}).format_prompt
        for case,index in enumerate(indices):
            question=rows[index]['question']
            if task=='sudoku':question=question.replace('0','.')
            prompt=formatter(question)
            for model,profile in profiles.items():
                admission=context_admission(prompt,profile)
                if not admission['admission_passed']:raise ValueError(f'Context budget exceeded for {model}/{task}')
                payload=build_payload(prompt,model,profile)
                job=f'{task}/{model}/{case:04d}'
                settings=dict(profile['call_settings'],historical_version=profile.get('historical_version'),
                              expected_provider_names=profile.get('expected_provider_names',[]))
                questions.append(dict(job_id=job,task=task,model=model,settings=settings,
                                      problem={'question':question},source_index=index))
                requests.append(dict(job_id=job,payload=payload,context_admission=admission))
        sources[task]=dict(source='explicit_csv' if task in paths else 'pinned_hf',
                           revision=None if task in paths else hub.RELEASES[task][1],
                           dataset=None if task in paths else f'cl-agi/{task}',split='test',size=30 if task=='maze' else 9,
                           csv_sha256=digest(path),population=len(rows),selected_indices=indices,
                           formatter_sha256=digest(frozen_code.source_path('output_contract/dataset/'+task+'.py')))
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('questions.jsonl',questions),('requests.jsonl',requests)]:
        (output/name).write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows))
    config=dict(seed=seed,count_per_task=count,profiles=profiles,sources=sources,
                python=sys.version,preparation_sha256=digest(__file__),profiles_sha256=digest(profiles_path),
                sampling='Python Random(str(seed)+":"+task).sample without replacement; same rows across models',
                scope='New sample, historical request settings; provider availability/version not verified',
                api_called=False,input_hashes={name:digest(output/name) for name in ('questions.jsonl','requests.jsonl')})
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=repository_root()/'exp/sudoku/records/frontier')
    p.add_argument('--profiles',type=Path,default=repository_root()/'exp/sudoku/configs/frontier/profiles.json')
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--seed',type=int)
    p.add_argument('--sudoku-path',type=Path)
    p.add_argument('--maze-path',type=Path)
    a=p.parse_args()
    paths={task:path for task,path in [('sudoku',a.sudoku_path),('maze',a.maze_path)] if path is not None}
    c=prepare(a.output,a.profiles,a.count,a.seed,paths)
    print(json.dumps(dict(seed=c['seed'],requests=2*a.count*len(c['profiles']),output=str(a.output),api_called=False)))


if __name__=='__main__':main()

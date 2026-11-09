"""Validate new E14 requests offline; --execute explicitly enables paid calls."""
import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path, PurePosixPath
from hrm.paths import repository_root
from hrm.analyse.baselines.frontier_llm import frozen_code
from hrm.analyse.baselines.frontier_llm.prepare_records import digest
from hrm.analyse.baselines.frontier_llm.score_records import read_index
from hrm.analyse.baselines.frontier_llm import run


def prepare(records):
    records=Path(records)
    config=json.loads((records/'resolved_config.json').read_text())
    for name in ('questions.jsonl','requests.jsonl'):
        if digest(records/name)!=config['input_hashes'][name]:raise ValueError(f'Prepared input changed: {name}')
    questions=read_index(records/'questions.jsonl');requests=read_index(records/'requests.jsonl')
    if questions.keys()!=requests.keys():raise ValueError('Question/request job IDs differ')
    original=frozen_code.load_original()
    hashes=frozen_code.verify_sources()
    jobs=[]
    for job_id,q in questions.items():
        path=PurePosixPath(job_id)
        if path.is_absolute() or '..' in path.parts or '\\' in job_id:raise ValueError('Unsafe job ID')
        task=q['task'];model=q['model'];profile=config['profiles'][model]
        if task not in ('sudoku','maze'):raise ValueError('Unsupported task')
        if not profile.get('expected_response_models') or not profile.get('upstream_requirements'):
            raise ValueError('Profile must freeze response-model and upstream parameter checks')
        settings=dict(profile['call_settings'],historical_version=profile.get('historical_version'),expected_provider_names=profile.get('expected_provider_names',[]))
        if q['settings']!=settings:raise ValueError('Question settings differ from the frozen profile')
        dimensions=dict(height=30,width=30) if task=='maze' else {}
        prompt=original.formatters[task].format_prompt(q['problem']['question'],**dimensions)
        payload=original.engine.build_payload(prompt,model,profile)
        admission=original.engine.context_admission(prompt,profile)
        if not admission['admission_passed'] or payload!=requests[job_id]['payload'] or admission!=requests[job_id]['context_admission']:
            raise ValueError('Prepared payload or context admission differs')
        problem_sha=original.transport.problem_hash(q['problem'])
        payload_sha=original.transport.problem_hash(payload)
        grader = "protocol/maze_shortest_v003.py" if task=="maze" else "original/dataset/sudoku.py"
        frozen=dict(model=model,provider=payload["provider"],payload=payload,problem_sha256=problem_sha,
                    template_sha256=digest(frozen_code.source_path("output_contract/dataset/"+task+".py")),
                    grader_sha256=hashes[grader],repeat_index=1,
                    profile_sha256=original.transport.problem_hash(profile),source_sha256=hashes)
        jobs.append(dict(job_id=job_id,task=task,variant='fresh_public_cohort',model=model,
                         case_id=q['source_index'],scale='9x9' if task=='sudoku' else '30x30',
                         problem=q['problem'],problem_sha256=problem_sha,payload=payload,payload_sha256=payload_sha,
                         call_settings=profile['call_settings'],model_profile=profile,
                         expected_provider_names=profile['expected_provider_names'],validator=original.validators[task],
                         maze_dimensions=dimensions or None,maze_diagnose=original.maze_diagnose if task=='maze' else None,
                         frozen_job=frozen,context_admission=admission))
    manifest=dict(adapter_version=1,group_semantics='Newly sampled E14 cohort; all prepared jobs, no historical archive dependency',
                  data_sha256={name:digest(records/name) for name in ('resolved_config.json','questions.jsonl','requests.jsonl')},
                  source_sha256=hashes,adapter_sha256={str(Path(f).name):digest(f) for f in (__file__,run.__file__,frozen_code.__file__)},
                  socket_timeout_seconds=run.SOCKET_TIMEOUT_SECONDS,
                  retry_policy=original.retry.RetryPolicy(None,run.BACKOFF_SECONDS).as_dict())
    return jobs,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=repository_root()/'exp/sudoku/records/frontier')
    p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true',help='Make paid API calls for every prepared job')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-attempts',type=int,default=5,help='Maximum transport attempts per job; unsuccessful answers are not retried')
    a=p.parse_args()
    if a.max_attempts<1:p.error('--max-attempts must be positive')
    if a.resume and not a.execute:p.error('--resume requires --execute')
    jobs,manifest=prepare(a.records)
    run.execution_manifest(jobs,manifest,a.workers)
    if not a.execute:
        print(json.dumps(dict(api_called=False,requests=len(jobs),tasks=dict(Counter(j['task'] for j in jobs)),models=sorted({j['model'] for j in jobs})),indent=2))
        return
    output=a.output or a.records/'execution'
    summary=run.execute(jobs,manifest,output,workers=a.workers,resume=a.resume,max_attempts=a.max_attempts)
    # The same enumerated input list gives the offline scorer an explicit denominator.
    source=(a.records/'questions.jsonl').read_bytes()
    if hashlib.sha256(source).hexdigest()!=manifest['data_sha256']['questions.jsonl']:
        raise ValueError('Prepared questions changed during execution')
    dest=output/'questions.jsonl'
    if dest.exists() and dest.read_bytes()!=source:raise ValueError('Execution question list changed')
    temporary=dest.with_suffix(".jsonl.tmp")
    temporary.write_bytes(source)
    temporary.replace(dest)
    if summary['unresolved']:
        raise SystemExit('Execution paused/incomplete; inspect summary.json and resume the same frozen plan')
    print(json.dumps(dict(status=summary['status'],finalized=summary['finalized'],output=str(output))))


if __name__=='__main__':main()

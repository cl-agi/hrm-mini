"""Freeze the source trace snapshot before a repair stage."""
import json
from pathlib import Path
from hrm_qwen.cot.run_config import file_hash
from hrm_qwen.cot.generation_config import read_records, trace_id


def freeze_repair(args):
    source=Path(args.source).resolve(strict=True)
    out=Path(args.out)
    if source==out.resolve():raise ValueError('Repair output must differ from source')
    if args.workers<1 or args.seed<0 or args.timeout<=0 or args.temperature<0:
        raise ValueError('Invalid workers, seed, timeout or temperature')
    if args.max_total_tokens<4096 or args.max_prefix_tokens<0 or (args.limit is not None and args.limit<1):
        raise ValueError('Invalid repair token budget or limit')
    identities={}
    for row in read_records(source):
        q,a=row['question'],row['answer']
        if len(q)!=81 or len(a)!=81 or any(c not in '0123456789' for c in q) or any(c not in '123456789' for c in a):
            raise ValueError('Malformed Sudoku in trace source')
        identity=trace_id(q,a)
        if row['index'] in identities and identities[row['index']]!=identity:
            raise ValueError('One source index refers to different puzzle variants')
        identities[row['index']]=identity
    config={k:v for k,v in vars(args).items() if k!='execute'}
    config['source_sha256']=file_hash(source)
    config['code_sha256']={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    path=Path(args.out+'.config.json')
    if path.exists():
        if json.loads(path.read_text())!=config:
            raise ValueError('Repair source or configuration changed; use a new output path')
    else:
        if out.exists():raise FileExistsError('Existing repairs have no frozen configuration')
        out.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(config,stream,indent=2)
    if out.exists():
        for row in read_records(out):
            if identities.get(row['index'])!=trace_id(row['question'],row['answer']):
                raise ValueError('Repair output does not match source puzzle')
    return config

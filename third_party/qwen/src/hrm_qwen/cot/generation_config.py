"""Public-input and frozen-run bookkeeping for teacher trace generation."""
import fcntl
import hashlib
import json
import random
from pathlib import Path
import time

from hrm import hub
from hrm_qwen.cot.run_config import file_hash
from hrm_qwen.cot.scoring import puzzle_with_empties
from hrm_qwen.cot.ratings import ratings_for


def trace_id(question, answer):
    # Identify the actual variant while retaining source-index selection priority.
    return hashlib.sha256((question+'\n'+answer).encode()).hexdigest()


def load_seed(args):
    path=Path(args.out+'.config.json')
    if args.seed is None:
        args.seed=json.loads(path.read_text())['seed'] if path.exists() else int(time.time())


def prepare_run(args,puzzles):
    if args.offset<0 or args.count<1 or args.seed<0 or args.workers<1:
        raise ValueError('Invalid offset/count/seed/workers')
    if args.empties is not None and not 0<=args.empties<=81:
        raise ValueError('empties must be between 0 and 81')
    if args.timeout<=0 or args.max_tokens<1 or args.reasoning_tokens<1 or args.temperature<0 or args.continuations<0:
        raise ValueError('Invalid request budget/settings')
    if args.max_cost is not None and args.max_cost<=0:
        raise ValueError('max-cost must be positive')
    if args.augment and args.min_rating is not None:
        raise ValueError('min-rating filtering is defined only for unaugmented source rows')
    if args.select_count is not None and args.select_count<1:
        raise ValueError('select-count must be positive')
    ratings=None
    rating_identity=None
    if args.min_rating is not None or args.order!='shuffle':
        ratings,rating_identity=ratings_for(args.data_dir,with_metadata=True)
    inputs={}
    excluded=set()
    for name in (args.only_unsolved_in or '').split(','):
        if not name:continue
        path=Path(name).resolve(strict=True)
        inputs[str(path)]=file_hash(path)
        for record in read_records(path):
            if record.get('exact'):excluded.add(record.get('source_index',record['index']))
    resolved=[]
    for index,q,a in puzzles:
        if args.empties is not None:q=puzzle_with_empties(q,a,args.empties,args.seed,index)
        resolved.append(dict(source_index=index,index=index,variant_id=trace_id(q,a),question=q,answer=a))
    eligible=[i for i,_,_ in puzzles if i not in excluded
              and (args.min_rating is None or ratings[i]>=args.min_rating)]
    random.Random(args.seed).shuffle(eligible)
    if args.order!='shuffle':eligible.sort(key=lambda i:ratings[i%len(ratings)],reverse=args.order=='rating_desc')
    if args.select_count is not None:eligible=eligible[:args.select_count]
    config={k:v for k,v in vars(args).items() if k not in ('execute','retry_errors','retry_wrong')}
    config['questions_sha256']=hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()
    config['other_traces_sha256']=inputs
    config['rating_source']=rating_identity
    config['selected_indices']=eligible
    config['source_sha256']={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    config['data_source']=({'repo':'cl-agi/sudoku','revision':hub.RELEASES['sudoku'][1]}
        if args.data_dir is None else {'train_csv_sha256':file_hash(hub.puzzle_data('sudoku',args.data_dir,split='train'))})
    out=Path(args.out);path=Path(args.out+'.config.json')
    if path.exists():
        if json.loads(path.read_text())!=config:
            raise ValueError('Teacher configuration, input or source changed; use a new output path')
    else:
        if out.exists():raise FileExistsError('Existing traces have no frozen configuration')
        out.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(config,stream,indent=2)
    expected={r['source_index']:r for r in resolved if r['source_index'] in set(eligible)}
    if out.exists():
        with out.open() as stream:
            for line in stream:
                rec=json.loads(line)
                row=expected.get(rec.get('source_index'))
                if row is None or any(rec.get(k)!=row[k] for k in ('index','variant_id','question','answer')):
                    raise ValueError('Existing trace belongs to a different puzzle/variant')
    return config


def execution_lock(out):
    handle=Path(out+'.lock').open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another generation process is using this output')
    return handle


def read_records(path):
    with open(path) as stream:
        for line in stream:
            yield json.loads(line)

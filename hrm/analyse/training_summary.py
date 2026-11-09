"""Summarize local scalar training evaluations without mixing selection rules."""
import argparse
import json
import os
from pathlib import Path


def summarize(root,split,cycles=None):
    runs=[]
    for path in sorted(Path(root).rglob('evaluation.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows=[r for r in rows if r['split']==split and (cycles is None or r['inference_cycles']==cycles)]
        if not rows:continue
        identity={(r['seed'],r['total'],r['inference_cycles'],r['weights'],r.get('stable_cycles',0)) for r in rows}
        if len(identity)!=1:raise ValueError(f'Inconsistent seed/denominator/budget/weights/stopping rule (use --cycles for a budget sweep): {path}')
        if len({r['optimizer_step'] for r in rows})!=len(rows):raise ValueError(f'Duplicate evaluation steps: {path}')
        for row in rows:
            if row['total']<=0 or not 0<=row['correct']<=row['total'] or abs(row['accuracy']-row['correct']/row['total'])>1e-12:
                raise ValueError(f'Invalid scalar record: {path}')
        ordered=sorted(rows,key=lambda r:r['optimizer_step'])
        best=max(rows,key=lambda r:(r['accuracy'],-r['optimizer_step']))
        runs.append({'records':str(path.relative_to(root)),'evaluations':len(rows),'peak':best,'last':ordered[-1]})
    if not runs:raise FileNotFoundError(f'No evaluation.jsonl records for split {split!r} under {root}')
    return {'split':split,'inference_cycles':cycles,'runs':runs,'selection':'Peak uses earliest tie; last uses maximum recorded optimizer step. No cross-run or cross-architecture aggregation. Last recorded evaluation is not proof of completed training.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=os.environ.get('HRM_ANALYSIS_RECORDS'))
    p.add_argument('--split',default='test_hard');p.add_argument('--cycles',type=int);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.records is None:p.error('Use hrm analyse with --experiment or supply --records')
    root=Path(args.records).resolve();result=summarize(root,args.split,args.cycles)
    output=args.output or root/'training_summary.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

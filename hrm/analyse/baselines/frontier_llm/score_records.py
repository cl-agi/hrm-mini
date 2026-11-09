"""Score a new, explicitly enumerated Sudoku/Maze response cohort offline."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from hrm.paths import repository_root
from hrm.analyse.baselines.llm_sat import frozen_grading as grading


def read_index(path):
    rows={}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            key=row['job_id']
            if not isinstance(key,str) or not key or key in rows:raise ValueError('Invalid or duplicate job_id')
            rows[key]=row
    if not rows:raise ValueError(f'Empty cohort: {path}')
    return rows


def score(records):
    records=Path(records)
    questions=read_index(records/'questions.jsonl')
    answers=read_index(records/'final_answers.jsonl')
    if questions.keys()!=answers.keys():raise ValueError('Question/answer IDs differ; missing attempts must not be dropped')
    groups=defaultdict(list)
    for job,q in questions.items():
        task=q['task'];model=q['model'];settings=q['settings'];problem=q['problem']
        if task not in ('sudoku','maze') or not isinstance(model,str) or not model or not isinstance(settings,dict) or not settings:
            raise ValueError('Each question requires supported task, model and nonempty frozen settings')
        question=problem['question']
        if not isinstance(question,str) or len(question)!=(81 if task=='sudoku' else 900):raise ValueError('Invalid input shape')
        if any(c not in ('.123456789' if task=='sudoku' else '# SG') for c in question):raise ValueError('Invalid input alphabet')
        if task=='maze' and (question.count('S')!=1 or question.count('G')!=1):raise ValueError('Maze requires one start and goal')
        a=answers[job]
        status=a['status'];final=a['final_content']
        if status not in ('completed','length_limited','no_final_content','failed','content_filtered'):raise ValueError('Unknown terminal status')
        if not isinstance(final,str):raise ValueError('final_content must be a string; use empty string for no response')
        if status in ('no_final_content','failed') and final:raise ValueError('Failed/missing response cannot contain final content')
        flags=grading.regrade(task,problem,final,grading.robust_json_parse)
        key=(task,model,json.dumps(settings,sort_keys=True,allow_nan=False))
        groups[key].append((flags,status))
    result=[]
    for (task,model,settings),rows in sorted(groups.items()):
        n=len(rows)
        counts={name:sum(bool(flags[name]) for flags,_ in rows) for name in grading.FLAGS}
        result.append(dict(task=task,model=model,settings=json.loads(settings),n=n,counts=counts,
                           rates={name:v/n for name,v in counts.items()},
                           terminal_status_counts={status:sum(s==status for _,s in rows) for status in sorted({s for _,s in rows})}))
    return dict(api_called=False,scope='All supplied planned jobs; not a claim of the historical 800-call cohort',
                n=len(questions),groups=result,
                input_sha256={name:hashlib.sha256((records/name).read_bytes()).hexdigest() for name in ('questions.jsonl','final_answers.jsonl')},
                grader_sha256=hashlib.sha256(Path(grading.__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=repository_root()/'exp/sudoku/records/frontier')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=score(a.records)
    output=a.output or a.records/'scores.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(n=result['n'],groups=len(result['groups']),output=str(output))))


if __name__=='__main__':main()

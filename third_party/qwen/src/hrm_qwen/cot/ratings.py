"""Recover public difficulty ratings without changing the published puzzle schema."""
import csv
import hashlib
import json
from pathlib import Path

from hrm import hub
from hrm.training.data import SUDOKU_UPSTREAM, SUDOKU_UPSTREAM_REVISION
from hrm_qwen.cot.run_config import file_hash


def key(row):
    return row['question'].replace('.','0'),row['answer']


def match_ratings(rows,source):
    wanted={key(row) for row in rows}
    found={}
    with open(source,newline='') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or 'rating' not in reader.fieldnames:
            raise ValueError('Upstream CSV has no rating column')
        for row in reader:
            identity=key(row)
            if identity not in wanted:continue
            rating=int(row['rating'] or 0)
            if identity in found and found[identity]!=rating:
                raise ValueError('Conflicting upstream ratings for the same puzzle and solution')
            found[identity]=rating
    if set(found)!=wanted:
        raise ValueError(f'Public ratings missing for {len(wanted-set(found))} selected puzzles')
    return [found[key(row)] for row in rows]


def ratings_for(data_dir,*,with_metadata=False):
    train=hub.puzzle_data('sudoku',data_dir,split='train')
    with open(train,newline='') as stream:rows=list(csv.DictReader(stream))
    if not rows:raise ValueError('Empty training CSV')
    identity={'training_csv_sha256':file_hash(train)}
    if data_dir is not None:
        if 'rating' not in rows[0]:
            raise ValueError('Explicit training CSV must contain rating for rating-based selection')
        ratings=[int(row['rating'] or 0) for row in rows]
        identity['source']='explicit CSV'
    else:
        from huggingface_hub import hf_hub_download
        identity.update(source=SUDOKU_UPSTREAM,revision=SUDOKU_UPSTREAM_REVISION,
                        selected_dataset='cl-agi/sudoku',selected_revision=hub.RELEASES['sudoku'][1])
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        cache=hub.cache_path('sudoku',f'ratings/{digest}.json')
        if cache.exists():
            payload=json.loads(cache.read_text())
            if payload['identity']!=identity:raise ValueError('Cached rating source mismatch')
            ratings=payload['ratings']
        else:
            source=hf_hub_download(SUDOKU_UPSTREAM,'train.csv',repo_type='dataset',revision=SUDOKU_UPSTREAM_REVISION)
            ratings=match_ratings(rows,source)
            hub._write(cache,lambda tmp:tmp.write_text(json.dumps({'identity':identity,'ratings':ratings})+'\n'))
    if len(ratings)!=len(rows) or any(type(v) is not int or v<0 for v in ratings):
        raise ValueError('Invalid rating vector')
    identity['ratings_sha256']=hashlib.sha256(json.dumps(ratings).encode()).hexdigest()
    return (ratings,identity) if with_metadata else ratings

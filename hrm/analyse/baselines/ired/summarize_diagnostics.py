"""Summarize saved IRED diagnostics without importing historical conclusions."""
import argparse
import json
from pathlib import Path
import numpy as np
from hrm.paths import analysis_directory


def summarize(manifest, rrn_dir=None):
    manifest=Path(manifest)
    spec=json.loads(manifest.read_text())
    variants=spec['variants'];baseline=spec['baseline']
    if not variants or baseline not in variants:raise ValueError('A named baseline and nonempty variants are required')
    records={}
    # Require every declared output before loading any public reference data.
    for name,relative in variants.items():
        folder=Path(relative)
        if not folder.is_absolute():folder=manifest.parent/folder
        config=json.loads((folder/'config.json').read_text())
        splits=config['splits']
        if not splits or len(splits)!=len(set(splits)) or set(splits)-{'train','satnet','rrn'}:
            raise ValueError(f'Invalid split list for {name}')
        metrics={}
        for split in splits:
            metric=json.loads((folder/f'{split}.json').read_text())
            ids=np.load(folder/f'{split}.indices.npy',allow_pickle=False)
            pred=np.load(folder/f'{split}.predictions.npy',allow_pickle=False)
            if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids):
                raise ValueError(f'Invalid indices for {name}/{split}')
            if len(ids)==0 or pred.shape!=(len(ids),81) or metric['n']!=len(ids):
                raise ValueError(f'Denominator/shape mismatch for {name}/{split}')
            if pred.dtype.kind not in 'iu' or np.any((pred<1)|(pred>9)):
                raise ValueError(f'Invalid predicted digits for {name}/{split}')
            for key in ('exact_match','strict_valid_and_clues','clues_preserved'):
                if not np.isfinite(metric[key]) or not 0<=metric[key]<=1:
                    raise ValueError(f'Invalid metric {name}/{split}/{key}')
            metrics[split]=dict(metrics=metric,indices=ids,predictions=pred)
        records[name]=dict(config=config,splits=metrics)
    if 'rrn' not in records[baseline]['splits']:raise ValueError('Baseline must contain RRN diagnostics')
    from hrm_ired.sat_dataset import SudokuRRNDataset
    ds=SudokuRRNDataset('sudoku-rrn','test',rrn_dir)
    reference=records[baseline]['splits']['rrn']
    ids=reference['indices']
    if np.any(ids<0) or np.any(ids>=len(ds)):raise ValueError('RRN indices outside reference dataset')
    target=np.stack([ds[int(i)][1].reshape(81,9).argmax(-1).numpy()+1 for i in ids])
    base_correct=(reference['predictions']==target).all(1)
    output={}
    for name,record in records.items():
        item=dict(configuration=record['config'],splits={s:r['metrics'] for s,r in record['splits'].items()})
        if 'rrn' in record['splits']:
            rrn=record['splits']['rrn']
            if not np.array_equal(ids,rrn['indices']):raise ValueError(f'RRN input order differs: {name}')
            if rrn['metrics']['seed']!=reference['metrics']['seed']:raise ValueError(f'RRN sampling seed differs: {name}')
            correct=(rrn['predictions']==target).all(1)
            if not np.isclose(correct.mean(),rrn['metrics']['exact_match'],rtol=0,atol=1e-7):
                raise ValueError(f'RRN exact-match metric disagrees with predictions: {name}')
            item['paired_rrn_vs_baseline']=dict(n=len(ids),exact_correct=int(correct.sum()),
                newly_solved=int((correct & ~base_correct).sum()),newly_failed=int((~correct & base_correct).sum()),
                changed_boards=int((rrn['predictions']!=reference['predictions']).any(1).sum()),
                delta_percentage_points=float((correct.mean()-base_correct.mean())*100))
        output[name]=item
    return dict(baseline=baseline,results=output,
                scope='Saved diagnostic subsets only; paired RRN comparison requires identical ordered indices and sampling seed. No full-test or multi-seed claim is inferred.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,help='JSON with baseline name and variants mapping names to directories')
    parser.add_argument('--rrn-dir',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest=args.manifest or analysis_directory(__file__)/'diagnostics/variants.json'
    result=summarize(manifest,args.rrn_dir)
    output=args.output or manifest.with_name('summary.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:record.get('paired_rrn_vs_baseline') for name,record in result['results'].items()},indent=2))


if __name__=='__main__':main()

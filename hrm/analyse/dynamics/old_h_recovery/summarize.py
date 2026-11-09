"""Pool E30 trained no-HH decoder metrics across four models and sampled phases."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from hrm.paths import repository_root
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


def pool(rows):
    """Equal condition/phase weight as in S13; ratios are formed after pooling."""
    if not rows:raise ValueError('No reconstruction rows')
    fields=('mse','baseline_mse','target_energy','mae','target_abs','cosine','centered_cosine')
    values=np.array([[r[k] for k in fields] for r in rows],dtype=np.float64)
    if not np.isfinite(values).all() or (values[:,:5]<0).any():raise ValueError('Invalid metric sufficient statistics')
    means=dict(zip(fields,values.mean(0).tolist()))
    ratio=lambda a,b:a/b if b>0 else None
    l2sq=ratio(means['mse'],means['target_energy'])
    return dict(means,relative_l1=ratio(means['mae'],means['target_abs']),
        relative_l2=float(np.sqrt(l2sq)) if l2sq is not None else None,
        r2_pooled_position_baseline=1-means['mse']/means['baseline_mse'] if means['baseline_mse']>0 else None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/memory/records/old_h'))
    p.add_argument('--records',type=Path,default=root/'conditions')
    p.add_argument('--cases',nargs=4,default=['continued_1','continued_2','continued_3','scratch'])
    p.add_argument('--output-dir',type=Path,default=root/'summary');a=p.parse_args(argv)
    if len(set(a.cases))!=4:raise ValueError('Four distinct model conditions required')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    rows=[];provenance=[];checkpoints=set();reference=None
    for case in a.cases:
        folder=a.records/case;features=folder/'features';decoders=folder/'decoders'
        plan=json.loads((features/'run_config.json').read_text());fm=json.loads((features/'manifest.json').read_text())
        dm=json.loads((decoders/'manifest.json').read_text());dp=json.loads((decoders/'run_config.json').read_text())
        audit=json.loads((folder/'audit/audit.json').read_text())
        if fm.get('status')!='complete' or dm.get('status')!='complete' or audit.get('status')!='complete':raise ValueError('Incomplete condition: '+case)
        if any(fm.get(k)!=v for k,v in plan.items()) or any(dm.get(k)!=v for k,v in dp.items()):raise ValueError('Manifest/config mismatch')
        if plan['trained_hh_link'] or plan['evaluation_hh_link']:raise ValueError('Summary requires trained no-HH models, not inference-only deletion')
        if audit['feature_manifest_sha256']!=sha(features/'manifest.json') or audit['decoder_manifest_sha256']!=sha(decoders/'manifest.json') or dp['feature_manifest_sha256']!=sha(features/'manifest.json'):raise ValueError('Stale source or audit')
        for directory,manifest in [(features,fm),(decoders,dm)]:
            for name,digest in manifest['artifacts'].items():
                if sha(directory/name)!=digest:raise ValueError('Changed artifact: '+name)
        shared={k:plan[k] for k in ('counts','tokens','steps','board_ids','positions','train_dataset_sha256','test_dataset_sha256')}
        if reference is not None and shared!=reference:raise ValueError('Conditions must share puzzle/cell cohorts and phases')
        reference=shared
        if plan['checkpoint_sha256'] in checkpoints:raise ValueError('Duplicate backbone checkpoint')
        checkpoints.add(plan['checkpoint_sha256'])
        selected=[r for r in json.loads((decoders/'results.json').read_text()) if r['probe']=='next_l_mlp']
        if sorted(r['step'] for r in selected)!=plan['steps']:raise ValueError('Missing or duplicate next-L MLP phases')
        if not {(r['step'],'next_l_mlp') for r in selected}<={(r['step'],r['probe']) for r in audit['checks']}:raise ValueError('Unaudited result')
        rows.extend(dict(r,case=case) for r in selected)
        provenance.append(dict(case=case,checkpoint_sha256=plan['checkpoint_sha256'],audit_sha256=sha(folder/'audit/audit.json')))
    output=dict(metrics=pool(rows),conditions=provenance,rows=rows,
        weighting='Equal model/phase weights with equal held-out puzzle and cell counts; form relative norms after pooling numerator and denominator.',
        reference_sampling=reference['counts']==[1024,256,512] and reference['tokens']==16 and reference['steps']==[1,8,24],
        note='No confidence interval across independently trained backbones is inferred from this pooled point estimate. Training lineage must establish three continuations and one from-scratch model.',code_sha256=sha(Path(__file__)))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())

"""Score projected CREMP candidates against reference-only HF conformations."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from hrm import hub


def reference_molecule(payload, cap=0):
    from rdkit import Chem
    mol=Chem.MolFromMolBlock(payload['molblock'],removeHs=False)
    if mol is None:raise ValueError('Invalid reference molblock')
    mol.RemoveAllConformers()
    coordinates=payload['coordinates'][:cap or None]
    for xyz in coordinates:
        if len(xyz)!=mol.GetNumAtoms():raise ValueError('Reference atom/coordinate count differs')
        conf=Chem.Conformer(mol.GetNumAtoms())
        for i,point in enumerate(xyz):conf.SetAtomPosition(i,tuple(map(float,point)))
        mol.AddConformer(conf,assignId=True)
    if not mol.GetNumConformers():raise ValueError('Empty reference ensemble')
    return mol


def score(projected, output, reference_path=None, filter_mode='none', reference_cap=0, cpus=1):
    from rdkit import Chem
    from ringer.utils import evaluation
    from hrm.analyse.cremp.classical.evaluate_conformers import scores
    import pyarrow.parquet as pq
    projected=Path(projected)
    path=hub.local_file(reference_path) if reference_path is not None else hub.data_file('cremp','data/default/validation.parquet')
    references=pq.read_table(path,columns=['id','reference_conformers']).to_pylist()
    refs={r['id']:r['reference_conformers'] for r in references}
    candidates=json.loads((projected/'candidates.json').read_text())
    if not candidates:raise ValueError('No projected candidate records')
    keys=[r['candidate'] for r in candidates]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate candidate IDs')
    # This file is the user's local output from project_angles, not a downloaded pickle.
    with (projected/'molecules.pickle').open('rb') as f:molecules=pickle.load(f)
    if set(molecules)-set(keys):raise ValueError('Molecules without candidate records')
    flags={'none':None,'drift':'original_drift_filter','physical':'physical_valid','combined':'filter_accept'}
    if filter_mode not in flags:raise ValueError('Unknown filter mode')
    groups={}
    for item in candidates:
        if item['id'] not in refs:raise ValueError(f"Unknown reference molecule: {item['id']}")
        if item.get('success') and item['candidate'] not in molecules:raise ValueError('Successful candidate is missing its molecule')
        groups.setdefault(item['id'],[]).append(item)
    functions=[('all_heavy_rmsd',evaluation.compute_rmsd_matrix,.75),
               ('ring_rmsd',evaluation.compute_ring_rmsd_matrix,.1),
               ('ring_tfd',evaluation.compute_ring_tfd_matrix,.05)]
    results=[]
    for mid,items in groups.items():
        reference=reference_molecule(refs[mid],reference_cap)
        accepted=[r for r in items if r.get('success') and (flags[filter_mode] is None or r.get(flags[filter_mode],False))]
        predicted=None
        for r in accepted:
            mol=molecules[r['candidate']]
            if mol.GetNumConformers()!=1:raise ValueError('Expected exactly one conformer per candidate')
            if predicted is None:
                predicted=Chem.Mol(mol);predicted.RemoveAllConformers()
            elif mol.GetNumAtoms()!=predicted.GetNumAtoms():raise ValueError('Candidate atom counts differ')
            predicted.AddConformer(Chem.Conformer(mol.GetConformer()),assignId=True)
        row={'id':mid,'requested_candidates':len(items),'accepted_candidates':len(accepted),
             'num_reference':reference.GetNumConformers(),'metrics':{}}
        for name,fn,threshold in functions:
            if predicted is None:
                metric=dict(threshold=threshold,cov_recall=0.,cov_precision=0.,cov_f1=0.,mat_recall=None,mat_precision=None,cov_precision_per_requested=0.)
            else:
                matrix=fn(Chem.Mol(predicted),Chem.Mol(reference),ncpu=cpus)
                if matrix.shape!=(reference.GetNumConformers(),len(accepted)) or not np.isfinite(matrix).all():raise ValueError('Invalid distance matrix')
                metric=scores(matrix,threshold)
                metric['cov_precision_per_requested']=float(np.sum(matrix.min(axis=0)<=threshold)/len(items))
            row['metrics'][name]=metric
        results.append(row)
    means={}
    for name,_,_ in functions:
        means[name]={}
        for key in ['cov_recall','cov_precision','cov_f1','mat_recall','mat_precision','cov_precision_per_requested']:
            values=[r['metrics'][name][key] for r in results]
            means[name][key]=float(np.mean(values)) if all(v is not None for v in values) else None
    report={'molecules':len(results),'reference_split_molecules':len(refs),'complete_split':set(groups)==set(refs),
            'filter':filter_mode,'reference_cap':reference_cap,'reference_source':str(path),
            'requested_candidates':len(candidates),'accepted_candidates':sum(r['accepted_candidates'] for r in results),
            'aggregation':'Unweighted molecule means; failed/filtered candidates remain in requested counts. MAT is null if any molecule has no retained candidates.',
            'reference_usage':'Scoring only; no generation or projection is performed here.',
            'means':means,'results':results}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-path',type=Path,help='Override the HF validation Parquet')
    p.add_argument('--filter',choices=['none','drift','physical','combined'],default='none')
    p.add_argument('--reference-cap',type=int,default=0,help='0 uses all reference conformers; positive caps are smoke checks')
    p.add_argument('--cpus',type=int,default=1)
    a=p.parse_args()
    if a.reference_cap<0 or a.cpus<1:p.error('reference-cap must be nonnegative and cpus positive')
    result=score(a.projected,a.output,a.reference_path,a.filter,a.reference_cap,a.cpus)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))


if __name__=='__main__':main()

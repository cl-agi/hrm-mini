"""Shared no-reference closure/NeRF/filter path for every learned arm.

All candidates are retained with explicit failure/filter status. K is never
increased until K candidates pass. Public geometry constants are an external
prior shared by all arms, not estimated from validation targets.
"""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from hrm_cremp.ringer_support import ROOT, graph_from_smiles, conditioning, embed_2d, reconstruct_full, enable_geometry_repair, wrap
from ringer.utils import internal_coords
from hrm_cremp.angle_data import sha


def chirality_check(mol, graph):
    expected=Chem.FindMolChiralCenters(graph,includeUnassigned=True,useLegacyImplementation=False)
    expected={int(i):c for i,c in expected if c!='?'}
    inferred=Chem.Mol(mol)
    for atom in inferred.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if atom.HasProp('_CIPCode'):atom.ClearProp('_CIPCode')
    Chem.AssignAtomChiralTagsFromStructure(inferred,confId=0,replaceExistingTags=True)
    Chem.AssignStereochemistry(inferred,cleanIt=True,force=True)
    actual=dict(Chem.FindMolChiralCenters(inferred,includeUnassigned=True,useLegacyImplementation=False))
    mismatches={str(i):dict(expected=c,actual=actual.get(i)) for i,c in expected.items() if actual.get(i)!=c}
    return not mismatches,mismatches


def project(samples_path, output, train_geometry, dummy_seed=20261020):
    enable_geometry_repair()
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config_path=ROOT/'vendor/ringer/ringer/sidechain_reconstruction/data/reconstruction_data.pickle'
    with config_path.open('rb') as f:config=pickle.load(f)
    from hrm.hub import cremp_geometry
    statistics=cremp_geometry(train_geometry)
    if statistics['source_split']!='train':raise ValueError('Projection geometry must be train-only')
    means=statistics['mean_bond_distances']
    phase_distances=[float(means[k]) for k in internal_coords.BACKBONE_ATOM_LABELS]
    for residues in (4,5,6):config['macrocycle_ring_distances'][residues]=phase_distances*residues
    with np.load(samples_path,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    # A scaling run can predeclare fixed geometry from a strict training subset.
    geometry_source=str(arrays.get('geometry_source_cache_sha256',arrays['train_cache_sha256']))
    if geometry_source != statistics['source_cache_sha256']:
        raise ValueError('Projection geometry is not bound to the model training cache')
    records=[];mols={};dummy_cache={}
    for i,(mid,text,length,values) in enumerate(zip(arrays['ids'],arrays['graph_smiles'],arrays['lengths'],arrays['angles'])):
        row=dict(candidate=i,id=str(mid),filter_accept=False,success=False,
                 physical_valid=False,original_drift_filter=False,chirality_correct=None)
        tick=time.perf_counter()
        try:
            if str(mid) not in dummy_cache:
                graph=graph_from_smiles(str(text));_,mask,ring=conditioning(graph)
                # Exactly one graph-only dummy initialization per molecule, shared by all K candidates.
                dummy=embed_2d(graph,dummy_seed)
                dummy_cache[str(mid)]=(graph,dummy,mask,ring,str(text))
            graph,dummy,mask,ring,cached_text=dummy_cache[str(mid)]
            assert cached_text==str(text) and len(ring)==int(length)
            values=np.array(values[:length],copy=True)
            if not np.isfinite(values[mask]).all():raise ValueError('Nonfinite predicted angles')
            values[~mask]=np.nan
            distances=pd.Series(config['macrocycle_ring_distances'][len(ring)//3],index=ring)
            optimizer=internal_coords.RingInternalCoordinates(ring)
            opt_start=time.perf_counter()
            result,status=optimizer.to_cartesian(dummy,distances,
                pd.Series(values[:,0],index=ring),pd.Series(values[:,1],index=ring),
                opt_init='average',return_result_obj=True,print_warning=False)
            row['closure_seconds']=time.perf_counter()-opt_start
            full,xyz=reconstruct_full(graph,ring,result,values,config)
            if not np.isfinite(xyz).all():raise ValueError('Nonfinite reconstructed coordinates')
            torsions=internal_coords.get_macrocycle_dihedrals(full,ring).iloc[0].to_numpy()
            drift=float(np.abs(wrap(torsions-values[:,1])).mean()/np.pi)
            error=float(np.abs(optimizer.compute_distances(result[['x','y','z']].to_numpy().ravel())-distances).max())
            full_error=float(np.abs(internal_coords.get_macrocycle_bond_distances(full,ring).iloc[0].to_numpy()-distances).max())
            chiral_ok,mismatches=chirality_check(full,graph)
            physical_valid=bool(status.success) and max(error,full_error)<=1e-4 and chiral_ok
            row.update(success=True,optimizer_success=bool(status.success),optimizer_message=str(status.message),
                       sample_to_projected_rtfd=drift,original_drift_filter=drift<.01,
                       filter_accept=physical_valid and drift<.01,physical_valid=physical_valid,
                       chirality_correct=chiral_ok,chirality_mismatches=mismatches,
                       max_ring_bond_error=error,max_full_ring_bond_error=full_error)
            mols[i]=full
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
        row['wall_seconds']=time.perf_counter()-tick;records.append(row)
    with (out/'molecules.pickle').open('wb') as f:pickle.dump(mols,f)
    (out/'candidates.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=dict(input_sha256=sha(samples_path),candidate_count=len(records),
                 reconstructed=sum(r['success'] for r in records),accepted=sum(r['filter_accept'] for r in records),
                 wall_seconds=time.perf_counter()-start,dummy_seed=dummy_seed,
                 reference_coordinates_used=False,graph_only_initialization=True,
                 validity_scope='Finite coordinates, optimizer success, ring bond errors <=1e-4 A, graph-specified chirality; does not test clashes or energetic stability',
                 external_repairs=['graph-based stereo normalization','proper PCA rotation'],
                 reconstruction_config_sha256=sha(config_path),
                 train_geometry_sha256=sha(train_geometry),train_geometry_source=statistics,
                 model_training_cache_sha256=str(arrays['train_cache_sha256']),geometry_source_cache_sha256=geometry_source,
                 reconstruction_constants='Train-only ring bond distances; pinned public sidechain/rigid geometry asset, upstream estimation provenance not independently established',
                 output_schema='molecules.pickle maps candidate integer to full heavy-atom RDKit Mol; candidates.json retains every attempted candidate')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-geometry')
    a=p.parse_args()
    if a.train_geometry is None:
        from hrm.hub import data_file
        adjacent=Path(a.samples).with_suffix('.geometry.json')
        a.train_geometry=str(adjacent if adjacent.is_file() else data_file('cremp','geometry.json'))
    RDLogger.DisableLog('rdApp.warning');print(json.dumps(project(a.samples,a.output,a.train_geometry)),flush=True)

"""Read-only artifact audit and compact report, never a new solver/model run."""
from hrm.paths import analysis_directory
import csv
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
from rdkit import Chem
from hrm.analyse.cremp.classical.template_core import sha, load_training, transport, graph_from_smiles, sequence_for_graph
from ringer.utils import internal_coords


def read(path):return json.loads(Path(path).read_text())


def main():
    start=time.perf_counter();cpu=time.process_time();own=analysis_directory(__file__);root=own.parent
    probe=Chem.MolFromSmiles('C');probe.AddConformer(Chem.Conformer(1),assignId=True)
    point=np.asarray([.123456789,1.23456789,np.pi]);probe.GetConformer().SetAtomPosition(0,point)
    restored=pickle.loads(pickle.dumps(probe,protocol=4)).GetConformer().GetPositions()[0]
    assert np.array_equal(restored,point.astype(np.float32).astype(np.float64))
    generation=read(own/'val32_k64_v1/summary.json')
    assert generation['script_sha256']==sha(own/'generate_templates.py')
    assert generation['core_sha256']==sha(own/'template_core.py')
    train_path=root/'processed/train_1000_angles.npz';train,meta=load_training(train_path)
    assert sha(train_path)==generation['training_cache_sha256']
    assert sha(root/'processed/train_1000_angles.geometry.json')==generation['training_geometry_sha256']
    source=read(own/'val32_conditions.json');conditions=source['conditions']
    assert sha(own/'val32_conditions.json')==generation['conditions_sha256']
    refs=read(own/'val32_reference_manifest.json')['entries']
    assert [r['id'] for r in refs]==[r['id'] for r in conditions]
    allrefs=read(root/'data/split_v1/reference_manifest_ready_v1.json')['entries']
    assert [r['id'] for r in refs]==[r['id'] for r in allrefs if r['split']=='val'][:32]
    refs_checked=[]
    for ref in refs:
        expected=ref.get('reference_sha256',ref.get('sha256'));assert expected
        assert sha(ref['reference_path'])==expected
        refs_checked.append(dict(id=ref['id'],reference_sha256=expected))
    raw=read(own/'val32_k64_v1/raw_results.json');selected=read(own/'val32_k64_v1/selected_results.json')
    assert [r['id'] for r in raw]==[r['id'] for r in conditions]
    assert [r['id'] for r in selected]==[r['id'] for r in conditions]
    assert not set(r['id'] for r in raw)&set(train['ids'])
    candidates_checked=0;max_drift_disagreement=0.;records=[]
    for query,rawrow,selrow in zip(conditions,raw,selected):
        assert sha(rawrow['output_molecule'])==rawrow['output_sha256']
        assert sha(selrow['output_molecule'])==selrow['output_sha256']
        mol=pickle.loads(Path(rawrow['output_molecule']).read_bytes())
        kept=pickle.loads(Path(selrow['output_molecule']).read_bytes())
        _,ring=sequence_for_graph(graph_from_smiles(query['graph_smiles']))
        rows=read(rawrow['source_records']);assert len(rows)==64
        assert len({r['train_molecule_id'] for r in rows})==4
        assert all(sum(r['neighbor_rank']==rank for r in rows)==16 for rank in range(4))
        for row in rows:
            tr=row['train_row'];ci=row['train_global_conformer']
            assert train['ids'][tr]==row['train_molecule_id'] and int(train['conf_molecule'][ci])==tr
            sourceconfs=np.flatnonzero(train['conf_molecule']==tr)
            assert int(sourceconfs[row['train_local_conformer']])==ci
            assert int(train['lengths'][tr])==len(ring)
            values=transport(train['angles'][ci,:len(ring)],row['residue_shift'])
            checksum=hashlib.sha256(np.ascontiguousarray(values,dtype='<f4').tobytes()).hexdigest()
            assert checksum==row['transported_backbone_sha256']
            if row['success']:
                conf_id=row['raw_conformer_id'];single=type(mol)(mol,quickCopy=True)
                single.AddConformer(type(mol.GetConformer(conf_id))(mol.GetConformer(conf_id)),assignId=True)
                actual=internal_coords.get_macrocycle_dihedrals(single,ring).iloc[0].to_numpy()
                diff=(actual-values[:,1]+np.pi)%(2*np.pi)-np.pi
                drift=float(np.mean(np.abs(diff))/np.pi)
                max_drift_disagreement=max(max_drift_disagreement,abs(drift-row['sample_to_projected_rtfd']))
                # RDKit's default binary pickle stores coordinates in float32. Check a
                # bounded reload discrepancy and the exact selection decision, not 1e-12.
                assert abs(drift-row['sample_to_projected_rtfd'])<1e-6
                assert (drift<.01)==row['original_drift_filter']
                if row['filter_accept']:
                    assert np.array_equal(kept.GetConformer(row['selected_conformer_id']).GetPositions(),
                                          mol.GetConformer(conf_id).GetPositions())
            candidates_checked+=1
        assert mol.GetNumConformers()==sum(r['success'] for r in rows)
        assert kept.GetNumConformers()==sum(r['filter_accept'] for r in rows)
        records.append(dict(id=query['id'],residues=len(ring)//3,
                            unique_templates=rawrow['unique_sampled_training_conformers'],
                            requested=64,reconstructed=mol.GetNumConformers(),selected=kept.GetNumConformers(),
                            generation_wall_s=rawrow['generation_wall_s']))
    summaries={};matrix_count=0
    for view in ('raw','selected'):
        directory=own/f'val32_k64_v1_{view}_metrics'
        summary=read(directory/'summary.json');results=read(directory/'results.json')
        assert summary['reference_cap']==0 and len(results)==32
        assert summary['script_sha256']==sha(own/'evaluate_ring_run_v1_snapshot.py')
        for j,record in enumerate(results):
            assert record['id']==records[j]['id']
            for metric,stats in record['metrics'].items():
                with np.load(directory/stats['distance_matrix_file']) as f:matrix=f['distances']
                assert matrix.shape==(record['num_reference'],record['num_generated'])
                assert np.isfinite(matrix).all()
                if matrix.shape[1]:
                    r=matrix.min(axis=1);p=matrix.min(axis=0);cut=stats['threshold']
                    recalculated=dict(cov_recall=float(np.mean(r<=cut)),cov_precision=float(np.mean(p<=cut)),
                                      cov_precision_per_requested=float(np.sum(p<=cut)/64),
                                      mat_recall=float(np.mean(r)),mat_precision=float(np.mean(p)))
                    for key,value in recalculated.items():assert abs(value-stats[key])<1e-12
                matrix_count+=1
                for key in ('cov_recall','cov_precision','mat_recall','mat_precision'):
                    records[j][f'{view}_{metric}_{key}']=stats[key]
            records[j]['reference_conformers']=record['num_reference']
        for metric,stats in summary['means'].items():
            for key,value in stats.items():
                values=[r['metrics'][metric][key] for r in results]
                assert abs(float(np.mean(values))-value)<1e-12
        summaries[view]=summary['means']
    checks=read(own/'checks.json');receipts=[read(own/f'{name}_receipt.json') for name in ('checks','formal')]
    assert all(r['status']=='complete' and r['GPU_seconds']==0 for r in receipts)
    assert receipts[0]['script_sha256']==sha(own/'launch_checks_snapshot.py')
    assert receipts[1]['script_sha256']==sha(own/'launch.py')
    audit_wall=time.perf_counter()-start;audit_cpu=time.process_time()-cpu
    audit=dict(status='pass',candidates_source_and_transport_verified=candidates_checked,
               metric_matrices_regrouped=matrix_count,reference_hashes_verified=len(refs_checked),
               references=refs_checked,max_saved_drift_disagreement=max_drift_disagreement,
               coordinate_serialization='RDKit pickle float32 coordinate roundtrip independently checked',
               drift_reload_tolerance=1e-6,drift_filter_decisions_unchanged=True,
               audit_wall_s=audit_wall,audit_CPU_s=audit_cpu,GPU_seconds=0,
               note='Reference SHA field alias was absent in original metric script; every exact reference hash is independently verified here. Original run source is retained. First audit used an overly strict 1e-12 reload tolerance and stopped; observed 1.71e-8 discrepancy is bounded float32 serialization, not a changed filter result.')
    (own/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    costs=dict(checks_and_formal_wall_s=sum(r['wall_s'] for r in receipts),
               checks_and_formal_CPU_s=sum(r['CPU_seconds'] for r in receipts),
               audit_wall_s=audit_wall,audit_CPU_s=audit_cpu,GPU_seconds=0,
               generation_process_wall_s=receipts[1]['steps'][1]['wall_s'],
               generation_parent_and_worker_CPU_s=generation['parent_CPU_seconds']+generation['worker_process_CPU_seconds'])
    report=dict(title='Frozen training-only template backbone baseline, CREMP val32',
       schema='Top4 training-only residue-Morgan neighbors, 16 iid with-replacement first30 conformers each, K64; cyclic N-to-C alignment, no reversal',
       generation=generation,metrics=summaries,audit=audit,costs=costs,
       unique_sources=dict(mean=float(np.mean([r['unique_templates'] for r in records])),
                           min=min(r['unique_templates'] for r in records),max=max(r['unique_templates'] for r in records)),
       length_counts={str(n):sum(r['residues']==n for r in records) for n in (4,5,6)},
       limitations=['Only ring RMSD/rTFD are meaningful: sidechains are deliberately placeholder coordinates.',
                    'No all-heavy conformation generation, energetic stability, or stereochemical 3D validity claim.',
                    'Validation32 pilot, fixed seed and hyperparameters; no final test or broad SOTA claim.',
                    'K64 includes repeated source conformers; report unique counts and retain duplicates in precision.',
                    'Comparison with ETKDG K256 selected uses a different candidate/selection budget, not matched-cost dominance.'])
    import rdkit,scipy,pandas
    report['versions']=dict(Python=platform.python_version(),NumPy=np.__version__,RDKit=rdkit.__version__,
                            SciPy=scipy.__version__,pandas=pandas.__version__)
    report['shared_dependency_sha256']={name:sha(root/'model_tools'/name) for name in (
        'ringer_support.py','isolation_gate.py','vendor/ringer/ringer/utils/internal_coords.py',
        'vendor/ringer/ringer/utils/chem.py','vendor/ringer/ringer/utils/peptides.py',
        'vendor/ringer/ringer/utils/evaluation.py','vendor/ringer/ringer/utils/data/amino_acids.csv')}
    classical=root/'classical_tools/val32_k256_seed1_selected_fullref_eval'
    if (classical/'summary.json').exists():
        old=read(classical/'results.json');assert [r['id'] for r in old]==[r['id'] for r in records]
        assert [r['num_reference'] for r in old]==[r['reference_conformers'] for r in records]
        classical_summary=read(classical/'summary.json');assert classical_summary['reference_cap']==0
        report['comparison_ETKDG_K256_selected']=dict(source_summary_sha256=sha(classical/'summary.json'),
            means={k:v for k,v in classical_summary['means'].items() if k in ('ring_rmsd','ring_tfd')},
            interpretation='Template COV-R is higher, but COV-P and MAT-R/P are worse; not overall dominance.')
    (own/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (own/'per_molecule.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=[report['title'],'',report['schema'],
           'All 2048/2048 candidates reconstructed and passed frozen closure+drift filter; raw and selected are identical.',
           f"Unique source conformers per molecule: mean {report['unique_sources']['mean']:.3f}, min {report['unique_sources']['min']}, max {report['unique_sources']['max']}.",
           '', 'metric COV-R COV-P MAT-R MAT-P']
    for key,v in summaries['raw'].items():lines.append(f"{key} {v['cov_recall']:.9f} {v['cov_precision']:.9f} {v['mat_recall']:.9f} {v['mat_precision']:.9f}")
    lines += ['',f"Checks+formal: {costs['checks_and_formal_wall_s']:.6f} wall s, {costs['checks_and_formal_CPU_s']:.6f} CPU s, 0 GPU.",
              f"Generation process only (includes imports/setup/workers): {costs['generation_process_wall_s']:.6f} wall s.",
              f"Audit: {audit_wall:.6f} wall s, {audit_cpu:.6f} CPU s. 2048 sources/transports,128 matrices,32 reference hashes verified.",
              '',*report['limitations']]
    (own/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    hashes={str(p.relative_to(own)):sha(p) for p in sorted(own.rglob('*')) if p.is_file() and p.name!='checksums.json' and '__pycache__' not in p.parts}
    (own/'checksums.json').write_text(json.dumps(hashes,indent=2)+'\n');print(json.dumps(dict(costs=costs,metrics=summaries,audit=audit),indent=2))


if __name__=='__main__':main()

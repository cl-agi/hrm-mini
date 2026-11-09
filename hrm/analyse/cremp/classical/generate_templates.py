"""Training-only top4 x16 template backbone candidates; CPU, fixed K64."""
import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import resource
import time
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from hrm.analyse.cremp.classical.template_core import sha, load_training, graph_from_smiles, sequence_for_graph, SequenceSimilarity, transport, embed_2d, projected_container


def generate_one(job):
    row, candidates, phase_distances, output, dummy_seed = job
    started=time.perf_counter();cpu0=time.process_time()
    graph=graph_from_smiles(row['graph_smiles']); sequence,ring=sequence_for_graph(graph)
    assert len(ring)==row['length']
    raw=Chem.Mol(graph);raw.RemoveAllConformers()
    selected=Chem.Mol(raw)
    records=[]
    try:
        dummy=embed_2d(graph,dummy_seed)
        dummy_error=None
    except Exception as exc:
        dummy=None;dummy_error=f'{type(exc).__name__}: {exc}'
    for i,candidate in enumerate(candidates):
        tick=time.perf_counter()
        values=candidate['values'];record={k:v for k,v in candidate.items() if k!='values'}
        record.update(candidate=i,success=False,filter_accept=False)
        try:
            if dummy_error:raise ValueError(dummy_error)
            mol,info=projected_container(graph,dummy,ring,values,phase_distances)
            record.update(info,success=True,raw_conformer_id=raw.GetNumConformers())
            raw.AddConformer(Chem.Conformer(mol.GetConformer()),assignId=True)
            if info['filter_accept']:
                record['selected_conformer_id']=selected.GetNumConformers()
                selected.AddConformer(Chem.Conformer(mol.GetConformer()),assignId=True)
        except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'
        record['projection_wall_s']=time.perf_counter()-tick;records.append(record)
    stem=hashlib.sha256(row['id'].encode()).hexdigest()[:20]
    outputs={}
    for view,mol in [('raw',raw),('selected',selected)]:
        path=Path(output)/f'{stem}_{view}.pickle'
        path.write_bytes(pickle.dumps(mol,protocol=4));outputs[view]=str(path.resolve())
    record_path=Path(output)/f'{stem}_candidates.json'
    record_path.write_text(json.dumps(records,indent=2)+'\n')
    unique_sources=len({(c['train_molecule_id'],c['train_global_conformer']) for c in candidates})
    entry=dict(id=row['id'],input_smiles=row['graph_smiles'],ring_residues=sequence,
               requested_candidates=64,raw_candidates=raw.GetNumConformers(),accepted_candidates=selected.GetNumConformers(),
               unique_sampled_training_conformers=unique_sources,duplicate_samples=64-unique_sources,
               generation_wall_s=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu0,
               sidechains_are_placeholders=True,allowed_metrics=['ring_rmsd','ring_tfd'],
               full_molecule_physical_validity_tested=False,source_records=str(record_path.resolve()))
    views={view:dict(entry,view=view,output_molecule=path,output_sha256=sha(path)) for view,path in outputs.items()}
    print(json.dumps({k:v for k,v in entry.items() if k!='input_smiles'}),flush=True)
    return views


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-cache',type=Path,required=True)
    p.add_argument('--conditions',type=Path,required=True);p.add_argument('--train-geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261021);p.add_argument('--dummy-seed',type=int,default=20261020)
    a=p.parse_args();start=time.perf_counter();cpu0=time.process_time();RDLogger.DisableLog('rdApp.warning')
    a.output.mkdir(parents=True,exist_ok=False)
    train,metadata=load_training(a.train_cache)
    if len(train['ids']) != 1000:raise ValueError('Formal gate requires frozen 1000-molecule training bank')
    source=json.loads(a.conditions.read_text());conditions=source['conditions']
    assert len({q['id'] for q in conditions}) == len(conditions)
    assert not set(q['id'] for q in conditions) & set(train['ids'])
    geometry=json.loads(a.train_geometry.read_text())
    assert geometry['source_split']=='train' and geometry['source_cache_sha256']==sha(a.train_cache)
    phase=[geometry['mean_bond_distances'][key] for key in ('N','Calpha','CO')]
    similarity=SequenceSimilarity()
    sequences=[sequence_for_graph(graph_from_smiles(str(text)))[0] for text in train['graph_smiles']]
    confs=[np.flatnonzero(train['conf_molecule']==i) for i in range(len(sequences))]
    jobs=[];retrieval=[]
    for query in conditions:
        seq,ring=sequence_for_graph(graph_from_smiles(query['graph_smiles']))
        assert len(ring)==query['length']
        ranked=[]
        for i,trainseq in enumerate(sequences):
            if len(seq)!=len(trainseq):continue
            score,identical,shift=similarity.align(seq,trainseq)
            ranked.append(dict(train_row=i,train_id=str(train['ids'][i]),similarity=score,
                               exact_residue_matches=identical,shift=shift,sequence=trainseq))
        ranked.sort(key=lambda x:(-x['similarity'],-x['exact_residue_matches'],x['train_id']))
        assert len(ranked)>=4
        neighbors=ranked[:4];retrieval.append(dict(id=query['id'],query_sequence=seq,neighbors=neighbors))
        seed_material=f'{a.seed}|{query["id"]}'.encode()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(seed_material).digest()[:8],'little'))
        candidates=[]
        for rank,neighbor in enumerate(neighbors):
            i=neighbor['train_row'];chosen=rng.choice(confs[i],size=16,replace=True)
            for global_conf in chosen:
                values=transport(train['angles'][global_conf,:len(ring)],neighbor['shift'])
                candidates.append(dict(neighbor_rank=rank,train_row=i,train_molecule_id=neighbor['train_id'],
                    train_global_conformer=int(global_conf),train_local_conformer=int(np.flatnonzero(confs[i]==global_conf)[0]),
                    residue_shift=neighbor['shift'],similarity=neighbor['similarity'],
                    transported_backbone_sha256=hashlib.sha256(np.ascontiguousarray(values,dtype='<f4').tobytes()).hexdigest(),values=values))
        jobs.append((query,candidates,phase,str(a.output.resolve()),a.dummy_seed))
    (a.output/'retrieval.json').write_text(json.dumps(retrieval,indent=2)+'\n')
    prep=time.perf_counter()-start
    with mp.get_context('spawn').Pool(a.workers) as pool:
        results=list(pool.imap(generate_one,jobs))
    for view in ('raw','selected'):
        (a.output/f'{view}_results.json').write_text(json.dumps([r[view] for r in results],indent=2)+'\n')
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    summary=dict(status='complete',GPU_seconds=0,wall_s=time.perf_counter()-start,
        parent_CPU_seconds=time.process_time()-cpu0,worker_process_CPU_seconds=child.ru_utime+child.ru_stime,
        worker_generation_CPU_seconds=sum(r['raw']['CPU_seconds'] for r in results),preparation_wall_s=prep,
        molecules=len(results),candidate_budget=64,neighbor_count=4,samples_per_neighbor=16,sampling_with_replacement=True,
        requested=64*len(results),reconstructed=sum(r['raw']['raw_candidates'] for r in results),
        selected=sum(r['raw']['accepted_candidates'] for r in results),
        ranking='Max mean per-residue radius2 2048-bit chirality-aware Morgan Tanimoto over cyclic residue shifts; ties exact residue matches then train ID. No reversal.',
        sampling_seed=a.seed,dummy_seed=a.dummy_seed,training_cache_sha256=sha(a.train_cache),
        training_geometry_sha256=sha(a.train_geometry),conditions_sha256=sha(a.conditions),
        reference_coordinates_read=False,reference_angles_read=False,training_conformers_used=True,
        only_channels=[0,1],sidechains_are_placeholders=True,all_heavy_metrics_forbidden=True,
        filter='Optimizer success, maximum ring bond error <=1e-4 A, sample-to-projected rTFD<0.01. No full-molecule validity claim.',
        script_sha256=sha(__file__),core_sha256=sha(Path(__file__).with_name('template_core.py')))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

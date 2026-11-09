"""Training-only engineering checks; does not tune any retrieval/selection rule."""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
from rdkit import Chem
from hrm.analyse.cremp.classical.template_core import sha, load_training, graph_from_smiles, sequence_for_graph, SequenceSimilarity, transport, embed_2d, projected_container
from ringer.utils import evaluation,internal_coords


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-cache',type=Path,required=True)
    p.add_argument('--train-geometry',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();start=time.perf_counter();train,metadata=load_training(a.train_cache)
    geometry=json.loads(a.train_geometry.read_text());assert geometry['source_cache_sha256']==sha(a.train_cache)
    phase=[geometry['mean_bond_distances'][key] for key in ('N','Calpha','CO')]
    similarity=SequenceSimilarity();rows=[]
    for i in range(8):
        graph=graph_from_smiles(str(train['graph_smiles'][i]));sequence,ring=sequence_for_graph(graph)
        first=int(np.flatnonzero(train['conf_molecule']==i)[0]);values=train['angles'][first,:len(ring)]
        assert similarity.align(sequence,sequence)[0] == 1.
        for shift in range(len(sequence)):
            assert np.array_equal(transport(transport(values,shift),-shift),values)
            shifted_sequence=sequence[shift:]+sequence[:shift]
            assert similarity.align(sequence,shifted_sequence)[0] == 1.
        source=metadata['records'][i];path=Path(source['reference_path']);assert sha(path)==source['sha256']
        original=pickle.loads(path.read_bytes())['rd_mol'];reference=Chem.RemoveHs(original)
        one=Chem.Mol(reference,quickCopy=True);one.AddConformer(Chem.Conformer(reference.GetConformer(0)),assignId=True)
        extracted=np.stack([internal_coords.get_macrocycle_bond_angles(one,ring).iloc[0].to_numpy(),
                            internal_coords.get_macrocycle_dihedrals(one,ring).iloc[0].to_numpy()],axis=-1)
        assert np.max(np.abs(extracted-values)) < 3e-7
        dummy=embed_2d(graph,20261020);projected,info=projected_container(graph,dummy,ring,values,phase)
        assert info['ring_closure_valid'],info
        # Dummy sidechains never contribute to ring distances. Test with deliberately arbitrary
        # non-ring coordinates while keeping the same graph and ring coordinates.
        container=Chem.Mol(graph);container.AddConformer(Chem.Conformer(one.GetConformer()),assignId=True)
        for atom in set(range(container.GetNumAtoms()))-set(ring):
            container.GetConformer().SetAtomPosition(atom,[100.+atom,200.-atom,-300.])
        changed=Chem.Mol(projected)
        for atom in set(range(changed.GetNumAtoms()))-set(ring):
            changed.GetConformer().SetAtomPosition(atom,[-400.,500.,atom])
        measurements={}
        for name,fn in [('ring_rmsd',evaluation.compute_ring_rmsd_matrix),('ring_tfd',evaluation.compute_ring_tfd_matrix)]:
            identity=float(fn(Chem.Mol(one),Chem.Mol(one),ncpu=1)[0,0])
            dummy_identity=float(fn(Chem.Mol(container),Chem.Mol(one),ncpu=1)[0,0])
            actual=float(fn(Chem.Mol(projected),Chem.Mol(one),ncpu=1)[0,0])
            altered=float(fn(Chem.Mol(changed),Chem.Mol(one),ncpu=1)[0,0])
            assert max(identity,dummy_identity)<1e-6 and abs(actual-altered)<1e-6
            measurements[name]=dict(identity=identity,placeholder_identity=dummy_identity,
                                    self_reconstruction=actual,nonring_perturbation_error=abs(actual-altered))
        rows.append(dict(id=str(train['ids'][i]),ring_sequence=sequence,closure=info,metrics=measurements))
    result=dict(status='pass',GPU_seconds=0,wall_s=time.perf_counter()-start,
                source='First eight molecules in frozen training cache; not validation or a representative quality estimate',
                training_cache_sha256=sha(a.train_cache),training_geometry_sha256=sha(a.train_geometry),
                checks=['Residue membership and N/Calpha/CO phase','Cyclic transport inversion',
                        'Rotation alignment score invariance','Train first-conformer label/order equality',
                        'Shared closure succeeds using train means','Official identity metric equality',
                        'Arbitrary non-ring coordinates do not affect official ring RMSD/rTFD'],results=rows)
    if a.out.exists():raise ValueError('Refusing to replace checks')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

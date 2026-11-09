"""Small-molecule-grouped cache; graph conditioning is separate from conformer labels."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem, RDLogger
from hrm_cremp.ringer_support import FEATURES, conditioning, normalize_graph, mapped_smiles, graph_from_smiles, extract_fixture, wrap
from ringer.utils import internal_coords


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(manifest, output, split=None, max_conf=30, limit=None, seed=20261020):
    started=time.perf_counter()
    if Path(output).exists():raise ValueError('Refusing to replace an existing angle cache')
    source = json.loads(Path(manifest).read_text())
    entries = source['entries']
    entries = [e for e in entries if split is None or e['split']==split]
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise ValueError('No source molecules match requested split')
    if len({e['id'] for e in entries})!=len(entries):raise ValueError('Duplicated source molecule IDs')
    ids=[]; graphs=[]; fields=[]; lengths=[]; graph_text=[]
    angles=[]; conf_molecule=[]; records=[]; distance_sums=np.zeros(3);distance_counts=np.zeros(3,dtype=np.int64)
    for index, entry in enumerate(entries):
        path = Path(entry['reference_path'])
        source_sha=sha(path)
        expected_sha=entry.get('reference_sha256',entry.get('sha256',source_sha))
        if expected_sha!=source_sha:raise ValueError(f'Source file SHA mismatch: {path}')
        with path.open('rb') as f:
            ensemble=pickle.load(f)
            mol = Chem.RemoveHs(ensemble['rd_mol'])
        graph = normalize_graph(mol)
        features, mask, ring = conditioning(graph)
        # Faithful official cap: macrocycle.py takes DataFrame.iloc[:num_conf].
        # Preserve source conformer order, with all labels grouped by molecule.
        conf_ids = np.arange(min(max_conf,mol.GetNumConformers()))
        selected = Chem.Mol(mol, quickCopy=True)
        for c in conf_ids:
            selected.AddConformer(Chem.Conformer(mol.GetConformer(int(c))), assignId=True)
        extracted = extract_fixture(selected, ring)
        distance_values=extracted['distance'].to_numpy()
        for phase in range(3):
            distance_sums[phase]+=distance_values[:,phase::3].sum()
            distance_counts[phase]+=distance_values[:,phase::3].size
        values = np.stack([extracted[f].to_numpy() for f in FEATURES], -1)
        if not np.array_equal(np.isfinite(values), np.broadcast_to(mask, values.shape)):
            raise ValueError(f'Graph/label feature support mismatch: {entry["id"]}')
        padded = np.full((len(conf_ids),18,12), np.nan, dtype=np.float32)
        padded[:,:len(ring)] = values
        angles.append(padded); conf_molecule.extend([index]*len(conf_ids))
        features = np.pad(features, ((0,18-len(ring)),(0,0)))
        graphs.append(features); fields.append(np.pad(mask, ((0,18-len(ring)),(0,0))))
        lengths.append(len(ring)); ids.append(entry['id']); graph_text.append(mapped_smiles(graph))
        rebuilt_features, rebuilt_mask, rebuilt_ring = conditioning(graph_from_smiles(graph_text[-1]))
        assert rebuilt_ring == ring
        assert np.array_equal(rebuilt_features,features[:len(ring)])
        assert np.array_equal(rebuilt_mask,mask)
        conformer_info=ensemble.get('conformers')
        energies=np.asarray([c['totalenergy'] for c in conformer_info]) if conformer_info is not None else None
        records.append(dict(id=entry['id'], reference_path=str(path), sha256=source_sha,
                            selected_conformer_ids=conf_ids.tolist(), split=entry['split'],
                            source_energy_monotone_nondecreasing=bool(np.all(np.diff(energies)>=0)) if energies is not None else None,
                            source_adjacent_energy_inversions=int(np.sum(np.diff(energies)<0)) if energies is not None else None))
        if (index+1)%100==0:
            print(json.dumps(dict(prepared_molecules=index+1,selected_conformers=len(conf_molecule),
                                  elapsed_seconds=time.perf_counter()-started)),flush=True)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(ids), graph_smiles=np.asarray(graph_text),
                        graph_features=np.asarray(graphs,np.float32), field_mask=np.asarray(fields),
                        lengths=np.asarray(lengths), angles=np.concatenate(angles),
                        conf_molecule=np.asarray(conf_molecule,np.int64))
    meta=dict(schema='CREMP angles cache v1', source_manifest=str(manifest), source_manifest_sha256=sha(manifest),
              split=split, molecules=len(ids), conformers=len(conf_molecule), max_conf=max_conf,
              selection='Official first max_conf conformers in source order; no added energy filtering',
              cache_sha256=sha(output), features=list(FEATURES), graph_only_conditioning=True,
              records=records,wall_seconds=time.perf_counter()-started,
              molecule_order='Exactly the filtered source manifest entry order',
              graph_feature_dimensions=int(graphs[0].shape[-1]))
    output.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    if split=='train' or (split is None and all(e['split']=='train' for e in entries)):
        statistics=dict(source_split='train',source_cache_sha256=sha(output),molecules=len(ids),
                        conformers=len(conf_molecule),labels=internal_coords.BACKBONE_ATOM_LABELS,
                        mean_bond_distances=dict(zip(internal_coords.BACKBONE_ATOM_LABELS,
                                                    (distance_sums/distance_counts).tolist())),
                        counts=distance_counts.tolist(),method='Official atom-type means over selected train conformers, N-to-C ring order')
        output.with_suffix('.geometry.json').write_text(json.dumps(statistics,indent=2)+'\n')
    return meta


class AngleDataset(Dataset):
    feature_names = FEATURES
    feature_is_angular = [True]*12
    pad = 18

    def __init__(self, path, means=None):
        self.path=Path(path)
        metadata=self.path.with_suffix('.manifest.json')
        if metadata.exists() and json.loads(metadata.read_text())['cache_sha256']!=sha(path):
            raise ValueError('Angle cache SHA disagrees with its manifest')
        with np.load(path,allow_pickle=False) as f:
            self.arrays={k:f[k] for k in f.files}
        a=self.arrays
        self.ids=a['ids']; self.lengths=a['lengths']
        self.means=np.asarray(means,dtype=np.float32) if means is not None else None
        self.all_lengths=self.lengths.tolist()
        self.structures=None; self.atom_features=None

    def training_means(self):
        # Call only on the frozen train cache. All selected conformers contribute,
        # matching the official channel-wise circular mean definition.
        x=self.arrays['angles']
        means=np.arctan2(np.nanmean(np.sin(x),axis=(0,1)),np.nanmean(np.cos(x),axis=(0,1)))
        return np.nan_to_num(means,nan=0).astype(np.float32)

    def __len__(self):
        return len(self.arrays['conf_molecule'])

    def condition(self, molecule):
        a=self.arrays; length=int(a['lengths'][molecule])
        return dict(atom_features=torch.from_numpy(a['graph_features'][molecule].copy()),
                    attn_mask=(torch.arange(18)<length).float(),
                    feat_mask=torch.from_numpy(a['field_mask'][molecule].copy()),
                    lengths=torch.tensor(length), position_ids=torch.arange(18),
                    atom_ids=torch.tensor([0,1,2]*6))

    def __getitem__(self,index,ignore_zero_center=False):
        a=self.arrays; m=int(a['conf_molecule'][index]); result=self.condition(m)
        x=a['angles'][index].copy()
        if self.means is not None and not ignore_zero_center:
            x=wrap(x-self.means)
        result['angles']=torch.from_numpy(np.nan_to_num(x,nan=0))
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split');p.add_argument('--limit',type=int);p.add_argument('--max-conf',type=int,default=30)
    a=p.parse_args(); RDLogger.DisableLog('rdApp.warning')
    meta=prepare(a.manifest,a.output,a.split,a.max_conf,a.limit)
    print(json.dumps({k:v for k,v in meta.items() if k!='records'},indent=2))

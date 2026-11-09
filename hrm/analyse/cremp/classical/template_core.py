"""Fixed 2D-only retrieval and backbone transport. No reference coordinates."""
from hrm.paths import analysis_directory
import sys
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
MODEL_TOOLS = analysis_directory(__file__).parents[0] / 'model_tools'
pass
from hrm_cremp.ringer_support import graph_from_smiles, embed_2d, enable_geometry_repair, wrap
from ringer.utils import chem, peptides, internal_coords

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sequence_for_graph(graph):
    ring = chem.get_macrocycle_idxs(graph)
    assert len(ring) in (12, 15, 18)
    residue_map = peptides.get_residues(graph, macrocycle_idxs=ring)
    sequence = []
    for offset in range(0, len(ring), 3):
        triple = set(ring[offset:offset + 3])
        matches = [label for atoms, label in residue_map.items() if triple <= set(atoms)]
        if len(matches) != 1:
            raise ValueError(f'Backbone triple has {len(matches)} residue assignments: {triple}')
        sequence.append(matches[0])
    return (sequence, ring)

class SequenceSimilarity:

    def __init__(self):
        labels = sorted(peptides.AMINO_ACID_DATA.index)
        fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        self.fingerprints = {name: fingerprint.GetFingerprint(peptides.AMINO_ACID_DATA.loc[name, 'residue_mol']) for name in labels}
        self.scores = {(a, b): DataStructs.TanimotoSimilarity(self.fingerprints[a], self.fingerprints[b]) for a in labels for b in labels}

    def align(self, query, source):
        """Source residue (i + shift) % R maps to query residue i; never reverse."""
        assert len(query) == len(source)
        alternatives = []
        for shift in range(len(query)):
            aligned = [source[(i + shift) % len(source)] for i in range(len(query))]
            score = float(np.mean([self.scores[q, s] for q, s in zip(query, aligned)]))
            identical = sum((q == s for q, s in zip(query, aligned)))
            alternatives.append((score, identical, -shift))
        score, identical, negative_shift = max(alternatives)
        return (score, identical, -negative_shift)

def transport(values, shift):
    """Preserve N/Calpha/CO phase and both angle definitions under cyclic shift."""
    values = np.asarray(values)
    assert values.ndim == 2 and values.shape[1] == 2 and (values.shape[0] in (12, 15, 18))
    return values[(np.arange(len(values)) + 3 * shift) % len(values)].copy()

def projected_container(graph, dummy, ring, values, phase_distances):
    enable_geometry_repair()
    values = np.asarray(values, dtype=float)
    assert values.shape == (len(ring), 2) and np.isfinite(values).all()
    distances = pd.Series(list(phase_distances) * (len(ring) // 3), index=ring)
    optimizer = internal_coords.RingInternalCoordinates(ring)
    result, status = optimizer.to_cartesian(dummy, distances, pd.Series(values[:, 0], index=ring), pd.Series(values[:, 1], index=ring), opt_init='average', return_result_obj=True, print_warning=False)
    xyz = result.loc[ring, ['x', 'y', 'z']].to_numpy()
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite projected ring coordinates')
    container = Chem.Mol(dummy)
    conf = container.GetConformer()
    for atom, position in zip(ring, xyz):
        conf.SetAtomPosition(int(atom), position.tolist())
    actual = internal_coords.get_macrocycle_dihedrals(container, ring).iloc[0].to_numpy()
    drift = float(np.mean(np.abs(wrap(actual - values[:, 1]))) / np.pi)
    distance_error = float(np.max(np.abs(optimizer.compute_distances(xyz.ravel()) - distances.to_numpy())))
    closure_ok = bool(status.success) and distance_error <= 0.0001
    info = dict(optimizer_success=bool(status.success), optimizer_message=str(status.message), max_ring_bond_error=distance_error, sample_to_projected_rtfd=drift, ring_closure_valid=closure_ok, original_drift_filter=drift < 0.01, filter_accept=closure_ok and drift < 0.01, sidechains_are_placeholders=True, full_molecule_physical_validity_tested=False)
    assert Chem.MolToSmiles(container) == Chem.MolToSmiles(graph)
    return (container, info)

def load_training(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.manifest.json').read_text())
    if metadata['split'] != 'train' or metadata['cache_sha256'] != sha(path):
        raise ValueError('A hash-bound training cache is required')
    if metadata['max_conf'] != 30 or any((r['split'] != 'train' for r in metadata['records'])):
        raise ValueError('Require official first <=30 training conformers')
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in ('ids', 'graph_smiles', 'lengths', 'angles', 'conf_molecule')}
    arrays['angles'] = arrays['angles'][:, :, :2]
    assert len(set(arrays['ids'])) == len(arrays['ids'])
    assert all((0 < np.sum(arrays['conf_molecule'] == i) <= 30 for i in range(len(arrays['ids']))))
    return (arrays, metadata)

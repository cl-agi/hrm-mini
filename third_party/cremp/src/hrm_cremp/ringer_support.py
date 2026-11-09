"""Shared graph-only conditioning and repaired official geometry operations.

The vendor source is unchanged. Repairs are externally applied for every arm.
"""
from pathlib import Path
import sys
import numpy as np
from rdkit import Chem
ROOT = Path(__file__).resolve().parent
pass
from hrm_cremp.isolation_gate import FEATURES, graph_field_mask, embed_2d, extract_fixture, reconstruct_full, proper_canonical_transform, wrap
from ringer.utils import chem, featurization, internal_coords

def normalize_graph(mol):
    graph = Chem.RemoveHs(Chem.Mol(mol, quickCopy=True))
    graph.RemoveAllConformers()
    Chem.AssignStereochemistry(graph, cleanIt=True, force=True)
    return graph

def mapped_smiles(graph):
    graph = normalize_graph(graph)
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(graph, isomericSmiles=True)

def graph_from_smiles(text):
    graph = Chem.MolFromSmiles(text)
    if graph is None:
        raise ValueError('Invalid graph SMILES')
    maps = [a.GetAtomMapNum() for a in graph.GetAtoms()]
    if any(maps):
        if sorted(maps) != list(range(1, len(maps) + 1)):
            raise ValueError('Atom maps must encode a complete heavy-atom permutation')
        graph = Chem.RenumberAtoms(graph, np.argsort(maps).tolist())
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
    return normalize_graph(graph)

def conditioning(graph):
    graph = normalize_graph(graph)
    ring = chem.get_macrocycle_idxs(graph)
    if len(ring) not in (12, 15, 18):
        raise ValueError(f'Unsupported CREMP backbone length {len(ring)}')
    feature_graph = Chem.AddHs(graph, addCoords=False)
    assert feature_graph.GetNumConformers() == 0
    features = featurization.featurize_macrocycle_atoms(feature_graph, radius=3, size=32)
    features = features.loc[ring].to_numpy(dtype=np.float32)
    return (features, graph_field_mask(graph, ring), ring)

def enable_geometry_repair():
    internal_coords.compute_canonical_transform = proper_canonical_transform

"""Reference-independent energy, convergence, chirality and diversity selection."""
import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generation-results', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--energy-window-kcal', type=float, default=6.)
    ap.add_argument('--rmsd-prune-angstrom', type=float, default=.1)
    ap.add_argument('--max-kept', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    raw_results = json.loads(args.generation_results.read_text())
    results = []
    start = time.perf_counter()
    for raw in raw_results:
        t0 = time.perf_counter()
        mol = pickle.loads(Path(raw['output_molecule']).read_bytes())
        heavy = Chem.RemoveHs(mol)
        valid = [c for c in raw['conformers'] if c['mmff_status'] == 0
                 and c['specified_chirality_ok'] and c['mmff_energy_kcal_mol'] is not None
                 and math.isfinite(c['mmff_energy_kcal_mol'])]
        valid.sort(key=lambda c: (c['mmff_energy_kcal_mol'], c['id']))
        minimum = valid[0]['mmff_energy_kcal_mol'] if valid else None
        window = [c for c in valid if c['mmff_energy_kcal_mol'] <= minimum + args.energy_window_kcal]
        kept = []
        for candidate in window:
            conf_id = candidate['id']
            duplicate = any(AllChem.GetBestRMS(Chem.Mol(heavy), Chem.Mol(heavy),
                                               prbId=conf_id, refId=k['id']) < args.rmsd_prune_angstrom
                            for k in kept)
            if not duplicate:
                kept.append(candidate)
                if args.max_kept and len(kept) >= args.max_kept:
                    break
        selected = Chem.Mol(mol)
        selected.RemoveAllConformers()
        for c in kept:
            selected.AddConformer(Chem.Conformer(mol.GetConformer(c['id'])), assignId=True)
        p = args.out/Path(raw['output_molecule']).name
        p.write_bytes(pickle.dumps(selected, protocol=4))
        elapsed = time.perf_counter()-t0
        result = dict(raw)
        result.update(output_molecule=str(p.resolve()), output_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                      source_generation_molecule=raw['output_molecule'],
                      raw_conformer_records=raw['conformers'],
                      conformers=[dict(c, id=i, source_conformer_id=c['id']) for i,c in enumerate(kept)],
                      selected_candidates=len(kept), selection_wall_s=elapsed,
                      generation_plus_selection_wall_s=raw['generation_wall_s']+elapsed,
                      valid_converged_chiral_candidates=len(valid), candidates_within_energy_window=len(window),
                      minimum_valid_MMFF_energy=minimum,
                      selected_source_conformer_ids=[c['id'] for c in kept],
                      selected_yield_per_requested=len(kept)/raw['requested_candidates'],
                      selection={'MMFF_status_required':0, 'specified_chirality_required':True,
                                 'relative_MMFF94_window_kcal':args.energy_window_kcal,
                                 'heavy_atom_symmetry_RMSD_prune_A':args.rmsd_prune_angstrom,
                                 'order':'ascending MMFF energy; source ID tie break', 'max_kept':args.max_kept,
                                 'reference_conformers_or_metrics_used':False,
                                 'energy_note':'MMFF94 energy is not the GFN2-xTB/ALPB reference energy.'})
        results.append(result)
    (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.out/'completion.json').write_text(json.dumps({
        'status':'complete','GPU_seconds':0,'selection_wall_s':time.perf_counter()-start,
        'source_results_sha256':hashlib.sha256(args.generation_results.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'molecules':len(results),'selected_counts':[r['selected_candidates'] for r in results]
    },indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Replay a screened Mixer cohort and retain the selected case's full L states."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hrm.paths import repository_root
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .screen import rank_candidates


def read_screen(source):
    manifest = json.loads((source/'manifest.json').read_text())
    plan = json.loads((source/'run_config.json').read_text())
    if manifest.get('status') != 'complete' or any(manifest.get(k) != v for k, v in plan.items()):
        raise ValueError('Screen is incomplete or differs from its frozen plan')
    expected = {'cohort.npz': manifest['cohort_sha256']}
    for name in ('screen.json', 'summary.json', 'ranked_candidates.json'):
        expected[name] = manifest['record_sha256'][name]
    for hc in range(1, manifest['h_steps']+1):
        name = f'H{hc:03d}.pt'; expected[name] = manifest['boundary_sha256'][name]
    for name, digest in expected.items():
        if sha(source/name) != digest:
            raise ValueError(f'Screen record changed: {name}')
    rows = json.loads((source/'screen.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    candidates = rank_candidates(summary, rows)
    if candidates != json.loads((source/'ranked_candidates.json').read_text()):
        raise ValueError('Candidate ranking differs from screening records')
    with np.load(source/'cohort.npz', allow_pickle=False) as f:
        cohort = {k: f[k] for k in f.files}
    ids = cohort['ids'].tolist()
    if ids != manifest['board_ids'] or len(ids) != len(set(ids)):
        raise ValueError('Cohort IDs/order differ from the plan')
    if cohort['inputs'].shape != cohort['targets'].shape or cohort['inputs'].shape != (len(ids), 82):
        raise ValueError('Invalid cohort shape')
    return manifest, cohort, candidates


def analysis_h_steps(manifest):
    """Display/probe stops at first correctness; replay retains two extra H updates."""
    first = manifest.get('selected', {}).get('first_correct_H')
    return int(first) if first is not None else int(manifest['h_steps'])


@torch.inference_mode()
def replay(source, output, checkpoint=None, case=None, device=None):
    source, output = Path(source), Path(output)
    manifest, cohort, candidates = read_screen(source)
    if case is None:
        if not candidates:
            raise ValueError('No eligible case in this cohort; no selected trajectory exists. '
                             'Use --case only for an explicitly labelled diagnostic replay.')
        selected = candidates[0]
        case = selected['case']
        stop = max(selected['first_correct_H']+2, 4)
        selection = 'archived eligibility and ranking rule'
    else:
        selected = {'case': case}
        stop = manifest['h_steps']
        selection = 'explicit diagnostic case; not automatic ranked selection'
    if case not in cohort['ids'] or stop > manifest['h_steps']:
        raise ValueError('Case or requested boundaries are absent from the screen')
    checkpoint = Path(checkpoint or manifest['checkpoint'])
    if sha(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from screening checkpoint')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    device = device or manifest['device']
    arch = analysis_architecture(manifest['config']); arch.pop('name', None)
    model = HRM(arch).to(device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)), strict=True)
    x = torch.tensor(cohort['inputs'], device=device)
    index = cohort['ids'].tolist().index(case)
    emb = model.embed(x)
    h = model.zH_init.expand_as(emb).clone(); l = model.zL_init.expand_as(emb).clone()
    if l.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f'Unsupported trajectory storage dtype: {l.dtype}')
    bits = l.dtype == torch.bfloat16
    output.mkdir(parents=True)
    plan = dict(screen_records=str(source.resolve()), checkpoint=str(checkpoint.resolve()),
                config=manifest['config'], source_manifest_sha256=sha(source/'manifest.json'),
                checkpoint_sha256=manifest['checkpoint_sha256'], cohort_sha256=manifest['cohort_sha256'],
                seed=manifest['seed'], case=int(case), selection=selection, selected=selected,
                h_steps=stop, l_steps=manifest['l_steps'], board_ids=manifest['board_ids'],
                device=device, precision=str(l.dtype), storage='uint16 BF16 bits' if bits else 'float32',
                code_sha256=sha(Path(__file__)))
    (output/'run_config.json').write_text(json.dumps(plan, indent=2))
    np.savez_compressed(output/'cohort.npz', **cohort)
    hashes = {}; started = time.time()
    for hc in range(1, stop+1):
        reference = torch.load(source/f'H{hc:03d}.pt', map_location='cpu', weights_only=True)
        if not torch.equal(h.cpu(), reference['H_before']):
            raise RuntimeError(f'H{hc}: whole-batch H-before differs')
        filename = f'H{hc:03d}_case{case}' + ('_bits.npy' if bits else '.npy')
        trace = np.lib.format.open_memmap(output/filename, mode='w+',
            dtype=np.uint16 if bits else np.float32,
            shape=(manifest['l_steps']+1, x.shape[1], l.shape[-1]))
        def store(step):
            value = l[index].contiguous()
            trace[step] = (value.view(torch.uint16) if bits else value).cpu().numpy()
        store(0)
        for step in range(1, manifest['l_steps']+1):
            l = model.L_level(l+h+emb); store(step)
            if step > manifest['l_steps']-manifest['tail_steps']:
                tail_index = step-(manifest['l_steps']-manifest['tail_steps'])-1
                if not torch.equal(l.cpu(), reference['tail'][tail_index]):
                    raise RuntimeError(f'H{hc}: whole-batch tail differs at L{step}')
        trace.flush(); del trace
        h = model.H_level(h+l)
        for name, actual in [('H_after', h), ('L_after', l), ('predictions', model.lm_head(h).argmax(-1))]:
            if not torch.equal(actual.cpu(), reference[name]):
                raise RuntimeError(f'H{hc}: whole-batch {name} differs')
        hashes[filename] = sha(output/filename)
    result = dict(plan, status='complete', whole_batch_boundary_parity='bitwise H/L/predictions and tails',
                  trajectory_sha256=hashes, elapsed_seconds=time.time()-started)
    (output/'manifest.json').write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    root = Path(os.environ.get('HRM_ANALYSIS_RECORDS', repository_root()/'exp/dynamics/records/mixer'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, default=root/'screen')
    p.add_argument('--output-dir', type=Path, default=root/'selected')
    p.add_argument('--checkpoint', type=Path, help='Relocated copy of the exact screened weights')
    p.add_argument('--case', type=int, help='Diagnostic override; does not claim ranked selection')
    p.add_argument('--device', help='Defaults to screening device; bitwise checks remain mandatory')
    a = p.parse_args(argv)
    torch.set_num_threads(2)
    print(json.dumps(replay(a.records, a.output_dir, a.checkpoint, a.case, a.device), indent=2))


if __name__ == '__main__':
    main()

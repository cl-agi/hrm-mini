from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
"""Repeat the frozen L-only intervention, substituting only the fitted target.

The original intervention implementation is imported unchanged. Its `error`
field remains the original constraint-excess auxiliary measurement; answer
errors are computed from the saved diagnostic logits in a separate NPZ.
"""
from hrm.paths import repo_path
from pathlib import Path
import argparse
import json
import sys
import time
from types import SimpleNamespace, ModuleType
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
BASE = analysis_directory(__file__)
HEX = Path(str(work_path('hrm/analyse/sudoku/core')))
ORIGINAL = HEX / 'artifacts/critic_Lonly_20261004'
pass
scripts_package = ModuleType('scripts')
scripts_package.__path__ = [str(HEX / 'scripts')]
sys.modules['scripts'] = scripts_package
from hrm.analyse.dynamics.answer_error import critic_target_dose as dose
from hrm.analyse.dynamics.answer_error import intervene_internal_critic as intervention

def verify_freeze():
    frozen = json.loads((BASE / 'source_freeze_v2.json').read_text())
    for name, expected in frozen['sha256'].items():
        assert dose.common.sha(name) == expected, f'Frozen source changed: {name}'

def prepare(out, probe_name='linear_L', modules=('L',)):
    assert probe_name == 'linear_L' and modules == ('L',)
    protocol = json.loads((ORIGINAL / 'protocol.json').read_text())
    protocol.update(status='frozen before this target intervention', criterion=TARGET, criterion_definition=json.loads((BASE / 'protocol.json').read_text())['targets'][TARGET], new_cohort_selection=False, source='Unchanged 81 IDs selected by the ORIGINAL joint excess critic >=1; no CE/MSE re-selection.', inference='CPU FP32, math SDPA, TF32 disabled, 4 threads, batch size 8; unchanged from original L-only execution.', plot_y='Absolute update L2 norm; unchanged original L-only four plots and their resampling rules.', auxiliary_error_field='batch initial.error is the original constraint-excess measurement. answer_errors.npz contains CE and MSE from the same saved diagnostic logits.', inherited_protocol_sha256=dose.common.sha(ORIGINAL / 'protocol.json'), sha256={str(p): dose.common.sha(p) for p in [Path(__file__), intervention.SOURCE / 'probes.pt', dose.OLD / 'setup.pt', dose.OLD / 'analysis_arrays.npz', Path(dose.__file__), Path(intervention.__file__), BASE / 'source_freeze_v2.json']})
    dose.common.js(out / 'protocol.json', protocol)
    return protocol

@torch.inference_mode()
def precheck(out):
    """Validate the original protocol on ALL fixed states before intervention.

    Failed positivity/reachability is reported, never repaired by filtering,
    shifting scores, changing doses, or expanding the displacement limit.
    """
    setup = torch.load(dose.OLD / 'setup.pt', weights_only=True)
    old = np.load(dose.OLD / 'analysis_arrays.npz')
    selected = np.flatnonzero(old['critic_immediate'][0] >= 1.0) + 32
    expected_ids = json.loads((ORIGINAL / 'protocol.json').read_text())['included_ids']
    assert setup['ids'][selected].tolist() == expected_ids
    probe = intervention.Probe()
    fit = probe.fits['linear_L']
    w = fit['weight']
    bias = fit['bias']
    model, identity = dose.common.model_load('hrm', 1, device='cpu')
    observations = []
    for first in range(0, len(selected), 8):
        positions = selected[first:first + 8]
        x = setup['x'][positions]
        carry = dose.common.initial(model, len(x))
        for _ in range(2):
            carry = dose.common.step(model, carry, x, 1)
        h, l = (carry['z_H'], carry['z_L'])
        feature = torch.cat([l.mean(1), h.mean(1)], -1)
        old_batch = torch.load(ORIGINAL / f'batch_{first:03d}.pt', weights_only=True)
        assert torch.equal(feature, old_batch['initial']['feature'][0]), 'Native prefix changed'
        assert torch.equal(x, old_batch['x'])
        assert torch.equal(setup['y'][positions], old_batch['y'])
        scores = l.mean(1).double() @ w + bias
        vec = dose.directions(l, w, setup['covariance']['L'])
        a, b = dose.coefficients(l, vec['critic'], w)
        limit = 2 * np.arcsin(0.2 / 2)
        for k, idx in enumerate(positions):
            row = dict(id=int(setup['ids'][idx]), score=float(scores[k]), positive=bool(scores[k] > 0), doses=[])
            for q in dose.DOSES:
                delta = scores[k] * q
                sign = torch.sign(delta)
                grid = torch.linspace(0, limit, 128, dtype=torch.double) * sign
                monotone = bool(torch.all(b[k] * grid.cos() - a[k] * grid.sin() > 0))
                reached = a[k] * (np.cos(limit) - 1) + sign * b[k] * np.sin(limit)
                reachable = bool(sign * (reached - delta) >= 0)
                row['doses'].append(dict(q=q, monotone=monotone, reachable=reachable))
            observations.append(row)
    failures = [r for r in observations if not r['positive'] or not all((d['monotone'] and d['reachable'] for d in r['doses']))]
    report = dict(passed=not failures, criterion=TARGET, n=81, batch_size=8, threads=4, baseline_prefix_bitwise_equal=True, checkpoint=identity, score_min=min((r['score'] for r in observations)), score_max=max((r['score'] for r in observations)), nonpositive_count=sum((not r['positive'] for r in observations)), failures=failures, observations=observations)
    dose.common.js(out / 'precheck.json', report)
    if failures:
        raise RuntimeError(f'{TARGET}: fixed protocol incompatible on {len(failures)}/81 states; see precheck.json. No samples or doses changed.')
    return report

def record_answer_errors(out):
    ce_parts, mse_parts, ids = ([], [], [])
    for path in sorted(out.glob('batch_*.pt')):
        batch = torch.load(path, weights_only=True)
        logits = batch['initial']['decoded_logits'][:, :, 1:, 1:10].float()
        target = batch['y'][None, :, 1:] - 1
        target = target.expand(logits.shape[:-1])
        ce = -logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1).mean(-1)
        mse = (logits.softmax(-1) - F.one_hot(target, 9)).square().mean((-2, -1))
        ce_parts.append(ce.numpy())
        mse_parts.append(mse.numpy())
        ids.extend(batch['ids'].tolist())
    np.savez_compressed(out / 'answer_errors.npz', ids=ids, ce=np.concatenate(ce_parts, 1), mse=np.concatenate(mse_parts, 1))
    dose.common.js(out / 'answer_errors_metadata.json', dict(definition=json.loads((BASE / 'protocol.json').read_text())['targets'], axes=['arm', 'puzzle'], saved_diagnostic_readout=True, sha256=dose.common.sha(out / 'answer_errors.npz')))

def main():
    global TARGET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['ce', 'mse'], required=True)
    args = parser.parse_args()
    TARGET = args.target
    verify_freeze()
    out = BASE / 'interventions' / TARGET
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob('batch_*.pt')):
        raise FileExistsError('Existing observations must be preserved.')
    intervention.SOURCE = BASE / 'probes' / TARGET
    dose.SOURCE = intervention.SOURCE
    dose.prepare = prepare
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            precheck(out)
            dose.run(SimpleNamespace(output=str(out), probe='linear_L', device='cpu', batch_size=8))
        record_answer_errors(out)
        verify_freeze()
        dose.common.js(out / 'wrapper_completion.json', dict(status='complete', criterion=TARGET, seconds=time.monotonic() - started, wrapper_sha256=dose.common.sha(__file__)))
    except Exception as exc:
        dose.common.js(out / 'wrapper_status.json', dict(status='blocked', criterion=TARGET, error=str(exc), exception_type=type(exc).__name__, elapsed_seconds=time.monotonic() - started))
        raise
if __name__ == '__main__':
    main()

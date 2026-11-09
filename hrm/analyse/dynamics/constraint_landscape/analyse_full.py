"""Audit saved full-plane HRM score grids; no model calls or modifications."""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def summary(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return dict(n=len(a), min=float(a.min()), mean=float(a.mean()),
                median=float(np.median(a)), max=float(a.max()),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95))) if len(a) else None


def interpolate(x, y, z, q):
    """Bilinear grid interpolation, with NaN for queries outside the grid."""
    q = np.asarray(q).reshape(-1, 2)
    inside = (q[:, 0] >= x[0]) & (q[:, 0] <= x[-1]) & (q[:, 1] >= y[0]) & (q[:, 1] <= y[-1])
    ix = np.clip(np.searchsorted(x, q[:, 0], side='right')-1, 0, len(x)-2)
    iy = np.clip(np.searchsorted(y, q[:, 1], side='right')-1, 0, len(y)-2)
    tx = (q[:, 0]-x[ix])/(x[ix+1]-x[ix])
    ty = (q[:, 1]-y[iy])/(y[iy+1]-y[iy])
    value = ((1-tx)*(1-ty)*z[iy, ix] + tx*(1-ty)*z[iy, ix+1]
             + (1-tx)*ty*z[iy+1, ix] + tx*ty*z[iy+1, ix+1])
    return np.where(inside, value, np.nan)


def curve_distance(q, curve):
    """Exact distance to the open piecewise-linear tail; never closes a ring."""
    a, d = curve[:-1], np.diff(curve, axis=0)
    t = np.clip(((np.asarray(q)-a)*d).sum(1)/np.maximum((d*d).sum(1), 1e-30), 0, 1)
    return float(np.linalg.norm(np.asarray(q)-a-t[:, None]*d, axis=1).min())


def extrema(x, y, z):
    middle = z[1:-1, 1:-1]
    low, high = np.ones(middle.shape, bool), np.ones(middle.shape, bool)
    low_strict, high_strict = np.zeros(middle.shape, bool), np.zeros(middle.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                neighbor = z[1+dy:z.shape[0]-1+dy, 1+dx:z.shape[1]-1+dx]
                low &= middle <= neighbor
                high &= middle >= neighbor
                low_strict |= middle < neighbor
                high_strict |= middle > neighbor
    answer = {}
    for name, mask, sign in [('minima', low & low_strict, 1), ('maxima', high & high_strict, -1)]:
        rows, cols = np.nonzero(mask)
        rows, cols = rows+1, cols+1
        selected = []
        for j in np.argsort(sign*z[rows, cols]):
            r, c = int(rows[j]), int(cols[j])
            if all((r-v['row'])**2+(c-v['col'])**2 >= 25 for v in selected):
                selected.append(dict(row=r, col=c, q=[float(x[c]), float(y[r])], value=float(z[r, c])))
            if len(selected) == 12:
                break
        answer[name] = dict(count_before_separation=len(rows), selected_separated_by_5_grid_cells=selected)
    return answer


def audit_grid(path, reference):
    d = np.load(path)
    x, y, z = d['x'][0], d['y'][:, 0], d['expected_errors']
    assert z.shape == (len(y), len(x)) and np.isfinite(z).all()
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
    dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
    q = d['trajectory_q'] if 'trajectory_q' in d else reference['trajectory_q']
    phase = d['trajectory_phase'] if 'trajectory_phase' in d else reference['trajectory_phase']
    step = d['trajectory_step'] if 'trajectory_step' in d else reference['trajectory_step']
    method = path.stem.split('_')[0]
    result = dict(file=str(path), shape=list(z.shape), spacing=[dx, dy],
                  bounds=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
                  expected_errors=summary(z), soft_conflict=summary(d['soft_conflict']),
                  delta_rms=summary(d['delta_rms']), discrete_extrema=extrema(x, y, z))
    for name, flat in [('minimum', np.argmin(z)), ('maximum', np.argmax(z))]:
        r, c = np.unravel_index(flat, z.shape)
        point = [float(x[c]), float(y[r])]
        nearest = int(np.argmin(np.linalg.norm(q-point, axis=1)))
        result[name] = dict(q=point, value=float(z[r, c]),
                            on_rectangle_edge=bool(r in (0, len(y)-1) or c in (0, len(x)-1)),
                            distance_to_trajectory_by_phase={str(h): curve_distance(point, q[phase == h]) for h in range(1, 6)},
                            nearest_observed_index=nearest, nearest_observed_phase=int(phase[nearest]),
                            nearest_observed_step=int(step[nearest]),
                            distance_to_nearest_observed_point=float(np.linalg.norm(q[nearest]-point)))
    match = re.search(r'_H([234])$', path.stem)
    phases = [int(match.group(1))] if match else [2, 3, 4]
    result['tails'] = {}
    for h in phases:
        mask = (phase == h) & (step >= 3073)
        curve = q[mask]
        lo, hi = curve.min(0), curve.max(0)
        center = (lo+hi)/2
        ring_values = interpolate(x, y, z, curve)
        report = dict(n_points=len(curve), step_min=int(step[mask].min()), step_max=int(step[mask].max()),
                      bounds=[lo.tolist(), hi.tolist()], span=(hi-lo).tolist(),
                      span_in_this_grid_cells=((hi-lo)/[dx, dy]).tolist(),
                      bbox_center_q=center.tolist(), bbox_center_score=float(interpolate(x, y, z, center)[0]),
                      bbox_center_distance_to_tail=curve_distance(center, curve),
                      interpolated_score_on_tail=summary(ring_values))
        old_x, old_y = reference['x'][0], reference['y'][:, 0]
        report['span_in_previous_global_grid_cells'] = ((hi-lo)/[np.diff(old_x).mean(), np.diff(old_y).mean()]).tolist()
        old_values = interpolate(old_x, old_y, reference[method+'_expected_errors'], curve)
        for name in ('minimum', 'maximum'):
            report[name+'_distance_to_tail'] = curve_distance(result[name]['q'], curve)
        boxmask = ((d['x'] >= lo[0]) & (d['x'] <= hi[0]) & (d['y'] >= lo[1]) & (d['y'] <= hi[1]))
        if boxmask.any():
            flat = np.argmin(np.where(boxmask, z, np.inf))
            r, c = np.unravel_index(flat, z.shape)
            point = [float(x[c]), float(y[r])]
            report['minimum_in_tail_bbox'] = dict(q=point, value=float(z[r, c]), distance_to_tail=curve_distance(point, curve))
        for source in ['true', method]:
            values = reference[f'trajectory_{source}_expected_errors'][mask]
            report[source+'_direct_score_on_tail'] = summary(values)
            report['grid_interpolation_MAE_vs_'+source+'_direct_on_tail'] = float(np.nanmean(np.abs(ring_values-values)))
            report['previous_coarse_grid_interpolation_MAE_vs_'+source+'_direct_on_tail'] = float(np.nanmean(np.abs(old_values-values)))
        for name in ('minima', 'maxima'):
            report[name+'_representative'] = [dict(v, distance_to_tail=curve_distance(v['q'], curve))
                for v in result['discrete_extrema'][name]['selected_separated_by_5_grid_cells']]
        result['tails'][str(h)] = report
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).parent/'results/full_plane')
    parser.add_argument('--reference', type=Path, default=Path(__file__).parent/'results')
    parser.add_argument('--output', type=Path, default=Path('reports/case6566_global_lift_20261021/full_plane_numerical_audit.json'))
    args = parser.parse_args()
    paths = sorted(args.input.glob('*.npz'))
    assert paths, f'No grids found in {args.input}'
    grids = {}
    for path in paths:
        if not path.stem.startswith(('kernel_', 'fourier_')):
            continue
        reference = np.load(args.reference/(path.stem.split('_')[0]+'_fields.npz'))
        grids[path.stem] = audit_grid(path, reference)
    report = dict(definitions={
        'tail':'Last1024 observed points of each phase, steps3073..4096 inclusive.',
        'center':'Midpoint of axis-aligned tail bounding box. Not a topological ring interior.',
        'extrema':'8-neighbor discrete extrema, at least one strict inequality, excluding the grid edge. Report12 lowest/highest separated by5 grid cells; precision-dependent.',
        'curve_distance':'Euclidean distance to open consecutive segments of the observed terminal tail; no artificial ring closing.',
        'score_on_curve':'Bilinear interpolation of evaluated score grid, compared with previous direct HRM evaluation at trajectory coordinates.',
        'interpretation':'Learned global lift scored by HRM. Peaks/valleys in this scalar field do not alone establish dynamical attraction or explain a limit cycle.',
    }, grids=grids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(args.output)

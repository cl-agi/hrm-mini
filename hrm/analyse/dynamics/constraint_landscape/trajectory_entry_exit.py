"""Clip actual trajectory approaches/departures to a local rectangular window.

trajectory_entry_exit(path, nextpath, bounds) is projection-independent. Bounds
are [[xmin,ymin],[xmax,ymax]], or [xmin,xmax,ymin,ymax]. Fractional steps only
locate intersections along drawn consecutive-point segments, not native states.
"""
from hrm.paths import analysis_directory
import argparse
import json
from pathlib import Path

import numpy as np


def rectangle(bounds):
    b = np.asarray(bounds, dtype=float)
    if b.shape == (4,):
        b = b[[0, 2, 1, 3]].reshape(2, 2)
    if b.shape != (2, 2) or not np.isfinite(b).all() or (b[0] > b[1]).any():
        raise ValueError('bounds must specify finite lower/upper 2D corners')
    return b


def segment_clip(a, b, bounds):
    """Return the exact parameter interval of segment a+t*(b-a) in bounds."""
    lower, upper = rectangle(bounds)
    delta = b-a
    begin, end = 0., 1.
    for axis in (0, 1):
        if delta[axis] == 0:
            if not lower[axis] <= a[axis] <= upper[axis]:
                return None
        else:
            first, last = sorted(((lower[axis]-a[axis])/delta[axis],
                                  (upper[axis]-a[axis])/delta[axis]))
            begin, end = max(begin, first), min(end, last)
            if begin > end:
                return None
    return begin, end


def clip_polyline(path, bounds):
    """List connected fragments; never bridge an out-of-window excursion."""
    path = np.asarray(path, dtype=float)
    bounds = rectangle(bounds)
    fragments = []
    current = None
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        interval = segment_clip(a, b, bounds)
        if interval is None:
            current = None
            continue
        first, last = interval
        p, q = a+first*(b-a), a+last*(b-a)
        s, t = i+first, i+last
        if current is None or abs(current['steps'][-1]-s) > 1e-10:
            current = dict(xy=[p.tolist()], steps=[float(s)])
            fragments.append(current)
        if t > current['steps'][-1]:
            current['xy'].append(q.tolist())
            current['steps'].append(float(t))
    if len(path) == 1 and np.all((path[0] >= bounds[0]) & (path[0] <= bounds[1])):
        fragments.append(dict(xy=path.tolist(), steps=[0.]))
    for fragment in fragments:
        fragment['integer_step_indices'] = [int(round(s)) for s in fragment['steps']
                                            if abs(s-round(s)) < 1e-10]
    return fragments


def arrow_on_fragment(fragment, fraction):
    """Arrow stays on one actual plotted segment, near a chosen arclength."""
    points = np.asarray(fragment['xy'])
    if len(points) < 2:
        return None
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if lengths.sum() == 0:
        return None
    cumulative = np.cumsum(lengths)
    k = min(int(np.searchsorted(cumulative, fraction*cumulative[-1])), len(lengths)-1)
    while lengths[k] == 0:
        k += 1
    center = np.clip((fraction*cumulative[-1]-(cumulative[k]-lengths[k]))/lengths[k], .25, .75)
    left, right = max(0., center-.25), min(1., center+.25)
    a, b = points[k], points[k+1]
    s, t = fragment['steps'][k:k+2]
    return dict(tail=(a+left*(b-a)).tolist(), head=(a+right*(b-a)).tolist(),
                source_segment_index=k, source_step_tail=float(s+left*(t-s)),
                source_step_head=float(s+right*(t-s)), arclength_fraction=fraction)


def _visits(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [[int(a), int(b-1)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def trajectory_entry_exit(path, nextpath, bounds, tail_points=1024):
    """Return entry/exit fragments, step metadata and directed-arrow positions.

    Entry: the connected in-window approach ending at the FIRST RECORDED point
    in the axis-aligned envelope of the last tail_points states. This is first
    arrival, not a claim of permanent capture. Later excursions are recorded.

    Exit: nextpath[0] (which must equal path[-1]) through the FIRST RECORDED
    next-phase point outside bounds, clipped at its boundary crossing. All
    intervening next-phase states are retained, including winding paths.
    If no exit is observed, exit and exit_arrow are empty; the continuation is
    reported separately rather than treating a terminal point as an exit.
    """
    path, nextpath = np.asarray(path, dtype=float), np.asarray(nextpath, dtype=float)
    bounds = rectangle(bounds)
    for p in (path, nextpath):
        if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
            raise ValueError('Both paths must be nonempty finite [N,2] arrays')
    if not 1 <= tail_points <= len(path):
        raise ValueError('tail_points must be between1 and len(path)')
    gap = float(np.linalg.norm(path[-1]-nextpath[0]))
    if gap > 1e-9:
        raise ValueError('Current endpoint and next-phase start differ; cannot invent a joining path')
    inside = lambda points, box: np.all((points >= box[0]) & (points <= box[1]), axis=1)
    if not inside(path[-1:], bounds)[0]:
        raise ValueError('The current endpoint must be inside the local window')
    tail = path[-tail_points:]
    envelope = np.array([tail.min(0), tail.max(0)])
    arrival = int(np.flatnonzero(inside(path, envelope))[0])
    prefix_fragments = clip_polyline(path[:arrival+1], bounds)
    if not prefix_fragments:
        raise ValueError('First terminal-envelope arrival is outside the supplied window')
    entry = prefix_fragments[-1]
    window_visits = _visits(inside(path, bounds))
    window_fragments = clip_polyline(path, bounds)
    core_fragments = clip_polyline(path[:arrival+1], envelope)
    next_inside = inside(nextpath, bounds)
    outside = np.flatnonzero(~next_inside)
    exit_step = int(outside[0]) if len(outside) else None
    departure_prefix = nextpath[:exit_step+1] if exit_step is not None else nextpath
    departure_fragments = clip_polyline(departure_prefix, bounds)
    exit_fragments = departure_fragments if exit_step is not None else []
    later_exits = np.flatnonzero(~inside(path, bounds) & (np.arange(len(path)) > arrival))
    return dict(
        bounds=bounds.tolist(), tail_envelope=envelope.tolist(), tail_points=tail_points,
        entry=[entry], prior_window_pass_throughs=prefix_fragments[:-1], exit=exit_fragments,
        next_phase_continuation_without_observed_exit=departure_fragments if exit_step is None else [],
        entry_arrow=arrow_on_fragment(entry, .30),
        exit_arrow=arrow_on_fragment(exit_fragments[-1], 1.) if exit_fragments else None,
        entry_source='current_path', exit_source='next_path',
        first_window_recorded_step=int(np.flatnonzero(inside(path, bounds))[0]),
        first_window_segment_hit=float(window_fragments[0]['steps'][0]),
        first_tail_envelope_recorded_step=arrival,
        first_tail_envelope_segment_hit=float(core_fragments[0]['steps'][0]),
        selected_entry_window_crossing=float(entry['steps'][0]),
        current_phase_window_visits=window_visits,
        first_current_phase_recorded_exit_after_arrival=int(later_exits[0]) if len(later_exits) else None,
        final_window_visit_start=window_visits[-1][0],
        next_phase_first_recorded_outside_step=exit_step,
        next_phase_boundary_crossing_step=float(exit_fragments[-1]['steps'][-1]) if exit_step is not None else None,
        junction=dict(current_step=len(path)-1, next_step=0, coordinate_gap=gap),
        fractional_steps='Linear interpolation on the displayed segment; not a model-evaluated fractional-time state',
        capture_claim=False)


def verify_fragments(result, path, nextpath):
    bounds = np.asarray(result['bounds'])
    for name, source in [('entry', path), ('prior_window_pass_throughs', path), ('exit', nextpath),
                         ('next_phase_continuation_without_observed_exit', nextpath)]:
        for fragment in result[name]:
            points, steps = np.asarray(fragment['xy']), np.asarray(fragment['steps'])
            assert np.all(np.diff(steps) >= 0)
            assert np.all(points >= bounds[0]-1e-12) and np.all(points <= bounds[1]+1e-12)
            i = np.minimum(np.floor(steps).astype(int), len(source)-2)
            reconstructed = source[i]+(steps-i)[:, None]*(source[i+1]-source[i])
            assert np.max(np.abs(points-reconstructed)) < 1e-12


if __name__ == '__main__':
    root = analysis_directory(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', type=Path, default=root/'data/L4096_case_search_20260928/case6566_L4096_arrays.npz')
    parser.add_argument('--grids', type=Path, default=Path(__file__).parent/'results/existing_L_probe')
    parser.add_argument('--output', type=Path, default=root/'reports/case6566_global_lift_20261021/trajectory_entry_exit_audit.json')
    args = parser.parse_args()
    archive = np.load(args.arrays)
    paths = {h: archive[f'H{h}_global_xy'] for h in range(1, 6)}
    audit = dict(definition='First recorded terminal-envelope arrival and first recorded next-phase window exit; actual consecutive segments clipped to rectangle',
                 arrays=str(args.arrays), panels={})
    for name, h in [('H1_context', 1), ('H1', 1), ('H2', 2), ('H3', 3), ('H4', 4)]:
        g = np.load(args.grids/f'existing_L_probe_{name}.npz')
        bounds = [[g['x'].min(), g['y'].min()], [g['x'].max(), g['y'].max()]]
        result = trajectory_entry_exit(paths[h], paths[h+1], bounds)
        verify_fragments(result, paths[h], paths[h+1])
        result.update(current_phase=h, next_phase=h+1)
        audit['panels'][name] = result
        print(name, 'entry', result['selected_entry_window_crossing'], '->', result['first_tail_envelope_recorded_step'],
              'exit-next-step', result['next_phase_boundary_crossing_step'],
              'first-outside', result['next_phase_first_recorded_outside_step'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2)+'\n')
    print(args.output)

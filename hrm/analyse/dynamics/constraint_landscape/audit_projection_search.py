"""Independent geometric audit and common-plane orientation search.

No training, scalar-field fitting, or per-phase transforms. Uses its own
monotone-chain hull and separating-axis implementation rather than the search
script's geometry helpers. Optional native replay checks sampled archived L.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def hull(points):
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    points = sorted(map(tuple, points))
    if len(points) < 3:
        return np.array(points)

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []
    for sequence, chain in [(points, lower), (points[::-1], upper)]:
        for p in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
    return np.array(lower[:-1]+upper[:-1])


def point_edge_distances(points, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edge = b-a
    difference = points[:, None, :]-a[None]
    t = np.clip(np.sum(difference*edge[None], axis=2)/np.sum(edge*edge, axis=1), 0, 1)
    return np.linalg.norm(difference-t[:, :, None]*edge[None], axis=2)


def separation(a, b):
    edges = np.concatenate([np.roll(a, -1, axis=0)-a, np.roll(b, -1, axis=0)-b])
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    pa, pb = a@normals.T, b@normals.T
    gaps = np.maximum(pb.min(0)-pa.max(0), pa.min(0)-pb.max(0))
    best = int(gaps.argmax())
    disjoint = bool(gaps[best] > 1e-12)
    distance = min(point_edge_distances(a, b).min(), point_edge_distances(b, a).min()) if disjoint else 0.
    return dict(disjoint=disjoint, separating_axis=normals[best].tolist(),
                projected_gap=float(gaps[best]), euclidean_hull_gap=float(distance))


def point_to_hull(point, polygon):
    e = np.roll(polygon, -1, axis=0)-polygon
    d = point-polygon
    cross = e[:, 0]*d[:, 1]-e[:, 1]*d[:, 0]
    outside = not bool(np.all(cross >= -1e-12))
    return dict(outside=outside, distance=float(point_edge_distances(point[None], polygon).min()) if outside else 0.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', type=Path, required=True)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d = np.load(args.view)
    xy = d['xy']
    basis = d['basis']
    paths = [d[f'H{h}_xy'] for h in range(1, 6)]
    polygons = [hull(p[-1024:]) for p in paths[:4]]
    pairs = {f'H{i+1}_H{j+1}': separation(polygons[i], polygons[j])
             for i in range(4) for j in range(i+1, 4)}
    fraction = (xy[-1]-xy.min(0))/np.ptp(xy, axis=0)
    result = dict(view=str(args.view), terminal_definition='Each H1-H4 last 1024 recorded states, L=3073..4096.',
                  algorithm='Independent monotone-chain convex hull, separating axis theorem, point-segment distance.',
                  orthonormality_max_error=float(np.max(np.abs(basis.T@basis-np.eye(2)))),
                  paths_concatenate_exactly=bool(np.array_equal(np.concatenate(paths), xy)),
                  duplicate_boundary_max_coordinate_difference=[float(np.max(np.abs(paths[h][-1]-paths[h+1][0]))) for h in range(4)],
                  endpoint_bbox_fraction=fraction.tolist(), pairwise_terminal_hulls=pairs,
                  all_six_terminal_hull_pairs_disjoint=all(p['disjoint'] for p in pairs.values()),
                  minimum_hull_gap=min(p['euclidean_hull_gap'] for p in pairs.values()),
                  start_end_hull_checks={name: {f'H{h+1}': point_to_hull(pt, poly) for h, poly in enumerate(polygons)}
                                         for name, pt in [('start', xy[0]), ('end', xy[-1])]},
                  scope='Non-overlap is checked for terminal hulls, not for transient approach/exit paths. No claim of dynamical stability follows from this projection.')
    # Searching only common SO(2) rotations cannot change lengths, areas or hull intersections.
    rotations = []
    all_hull = hull(xy)
    for angle in np.arange(-90., 90., .25):
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points = all_hull@R
        span = np.ptp(points, axis=0)
        f = (xy[-1]@R-points.min(0))/span
        start = (xy[0]@R-points.min(0))/span
        if start[0] <= .16:
            rotations.append(dict(angle_degrees=float(angle), endpoint_bbox_fraction=f.tolist(),
                                  start_bbox_fraction=start.tolist(), maximum_center_deviation=float(np.max(np.abs(f-.5)))))
    result['best_common_rotation_with_start_fraction_x_le_016'] = min(rotations, key=lambda r: r['maximum_center_deviation'])
    result['rotation_recommendation'] = 'Keep the supplied orientation; the small centrality improvement from common rotation is cosmetic.'
    if args.original:
        original = np.load(args.original)
        oldxy = np.concatenate([original[f'H{h}_global_xy'] for h in range(1, 6)])
        oldfraction = (oldxy[-1]-oldxy.min(0))/np.ptp(oldxy, axis=0)
        result['original_endpoint_bbox_fraction'] = oldfraction.tolist()
        result['fixed_original_x_obstruction'] = 'Keeping the original x axis fixes the endpoint x fraction at %.9f; thus a .35-.65 x fraction is impossible without changing x or a common in-plane rotation.' % oldfraction[0]
    if args.archive:
        indices = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 1024, 2048, 3073, 3584, 4095, 4096])
        errors = {}
        for h, path in enumerate(paths, 1):
            bits = np.load(args.archive/f'H{h:03d}_L_bits.npy', mmap_mode='r')[indices, 1]
            native = (bits.astype(np.uint32) << 16).view(np.float32).reshape(len(indices), -1).astype(np.float64)
            exact = (native-d['origin'])@basis/np.sqrt(len(d['origin']))
            errors[f'H{h}'] = float(np.max(np.abs(exact-path[indices])))
        result['native_replay'] = dict(definition='Original bf16 L promoted to float64, (L-origin) @ basis / sqrt(41984).',
                                       sample_steps=indices.tolist(), max_abs_coordinate_error_by_phase=errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

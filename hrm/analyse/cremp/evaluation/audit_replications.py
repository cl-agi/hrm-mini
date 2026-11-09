"""Read-only verification of the seed2/3 CREMP replication evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--with-generation', action='store_true')
    a = p.parse_args()
    release = a.base / 'releases/pilot_v1'
    frozen = read(release / 'release_manifest.json')['files_sha256']
    assert all(sha(release / name) == value for name, value in frozen.items())
    train_sha = sha(a.base / 'processed/train_1000_angles.npz')
    val_sha = sha(a.base / 'processed/val_256_angles.npz')
    validation = np.load(a.base / 'processed/val_256_angles.npz', allow_pickle=False)
    report = dict(frozen_source_files_verified=len(frozen), runs=[], noise_pairing={},
                  receipts=[], test_data_accessed=False)
    for seed in (2, 3):
        hashes = []
        for family in ('hrm', 'rt', 'ringer'):
            name = f'pilot_v1_{family}_seed{seed}'
            run = a.base / 'runs' / name
            manifest, summary = read(run / 'manifest.json'), read(run / 'summary.json')
            reference = read(a.base / 'runs' / f'pilot_v1_{family}_seed1' / 'manifest.json')
            expected_args = dict(reference['args'], seed=seed, output=str(run))
            assert manifest['args'] == expected_args
            for key in ('model_config', 'budget', 'train_sha256', 'val_sha256', 'train_means'):
                assert manifest[key] == reference[key], key
            assert summary['completed_updates'] == 20000
            assert manifest['train_sha256'] == train_sha and manifest['val_sha256'] == val_sha
            rows = read(run / 'metrics.json')
            assert len(rows) == 20000 and [r['step'] for r in rows] == list(range(1, 20001))
            assert all(np.isfinite(r['loss']) for r in rows)
            noise = {r['step']: r['corruption_sha256'] for r in rows if 'corruption_sha256' in r}
            assert sorted(noise) == [1] + list(range(1000, 20001, 1000))
            hashes.append(noise)
            candidates = [r for r in rows if 'val_loss' in r]
            best = min(candidates, key=lambda r: (r['val_loss'], -r['step']))
            result = dict(run_id=name, final_val_loss=rows[-1]['val_loss'], best_step=best['step'],
                          best_val_loss=best['val_loss'], completed_updates=20000,
                          conformer_samples=rows[-1]['conformer_samples'],
                          nominal_conformer_epochs=rows[-1]['nominal_conformer_epochs'])
            receipt_path = run / 'receipt.json'
            receipt = read(receipt_path)
            assert receipt['exit_code'] == 0
            report['receipts'].append(dict(path=str(receipt_path), sha256=sha(receipt_path),
                                           gpu_seconds=receipt['gpu_seconds_conservative']))
            if a.with_generation:
                inference = a.base / 'inference' / name
                gen_summary = read(inference / 'summary.json')
                assert gen_summary['best_step'] == best['step']
                result['generation'] = []
                for choice in gen_summary['choices']:
                    directory = inference / choice['name']
                    m = read(directory / 'manifest.json')
                    assert m['checkpoint_sha256'] == sha(Path(m['args']['checkpoint']))
                    assert m['data_sha256'] == manifest['val_sha256']
                    assert not m['reference_angles_used']
                    assert (m['molecules'], m['k'], m['candidates']) == (32, 64, 2048)
                    assert m['args']['seed'] == 327681 and m['args']['batch_molecules'] == 16
                    with np.load(directory / 'generated_angles.npz', allow_pickle=False) as samples:
                        assert len(samples['angles']) == 2048
                        assert np.array_equal(samples['ids'], np.repeat(validation['ids'][:32], 64))
                        assert np.array_equal(samples['graph_smiles'], np.repeat(validation['graph_smiles'][:32], 64))
                        assert np.array_equal(samples['lengths'], np.repeat(validation['lengths'][:32], 64))
                        assert np.array_equal(samples['candidate_index'], np.tile(np.arange(64), 32))
                        assert str(samples['train_cache_sha256']) == train_sha
                        assert all(np.isfinite(x[:int(n)]).all()
                                   for x, n in zip(samples['angles'], samples['lengths']))
                    result['generation'].append(dict(directory=str(directory),
                        checkpoint_step=m['checkpoint_step'], checkpoint_sha256=m['checkpoint_sha256']))
                for path in sorted(inference.glob('*_receipt.json')):
                    if path.name.endswith('_launch_receipt.json'):
                        continue
                    r = read(path)
                    assert r['exit_code'] == 0
                    report['receipts'].append(dict(path=str(path), sha256=sha(path),
                                                   gpu_seconds=r['gpu_seconds_conservative']))
            report['runs'].append(result)
        assert hashes[0] == hashes[1] == hashes[2]
        report['noise_pairing'][str(seed)] = dict(all_three_arms_identical=True,
                                                 checkpoints=len(hashes[0]), hashes=hashes[0])
    report['total_gpu_seconds_conservative'] = sum(r['gpu_seconds'] for r in report['receipts'])
    report['total_gpu_hours_conservative'] = report['total_gpu_seconds_conservative'] / 3600
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('noise_pairing', 'receipts')}, indent=2))


if __name__ == '__main__':
    main()

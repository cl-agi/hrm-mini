#!/usr/bin/env python3
"""Freeze nested Sudoku subsets from the repository's existing Arrow cache."""
from hrm.paths import analysis_directory
import hashlib
import json
from pathlib import Path
import shutil

from datasets import Dataset, concatenate_datasets, load_from_disk


ROOT = analysis_directory(__file__)
REPO = ROOT.parents[1]
ORIGINAL = REPO / "experiments/sudoku_scaling_20261004/cache/subsets"
SEED = 42
SIZES = (1000, 5000, 10000, 100000, 1000000, 3831994)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_hash(dataset):
    digest = hashlib.sha256()
    for batch in dataset.iter(batch_size=4096):
        for question, answer in zip(batch["question"], batch["answer"]):
            digest.update((question + "\t" + answer + "\n").encode())
    return digest.hexdigest()


def main():
    target = ROOT / "data"
    manifest_path = ROOT / "data_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for split, expected in (("train", SIZES[-1]), ("test", 20000)):
            assert len(load_from_disk(target / split)) == expected
        print(json.dumps({"status": "already_prepared", "data": str(target),
                          "train_size": manifest["splits"]["train"]["subset_size"]}))
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Incomplete data already exists at {target}; inspect before retrying")
    target.mkdir(parents=True, exist_ok=True)
    original_metadata = json.loads((ORIGINAL / "metadata.json").read_text())
    metadata = {key: original_metadata[key] for key in
                ("dataset", "subsample_seed", "source_hub_revision")}
    assert metadata["subsample_seed"] == SEED
    metadata["splits"] = {}
    audit = {}
    for split in ("train", "test"):
        prior = original_metadata["splits"][split]
        paths = [Path(item["filename"]) for item in prior["cache_files"]]
        full = concatenate_datasets([Dataset.from_file(str(path)) for path in paths])
        assert len(full) == prior["full_size"]
        shuffled = full.select_columns(["question", "answer"]).shuffle(
            seed=SEED, keep_in_memory=True)
        selected = shuffled if split == "train" else shuffled.select(range(20000))
        existing = load_from_disk(ORIGINAL / prior["directory"])
        comparison = selected.select(range(len(existing)))
        expected_hash = row_hash(existing)
        assert row_hash(comparison) == expected_hash, f"Changed {split} prefix"
        if split == "train":
            assert len(selected) == SIZES[-1]
            assert len(SIZES) == 6 and all(size <= len(selected) for size in SIZES)
            selected.save_to_disk(str(target / split), max_shard_size="400MB")
        else:
            shutil.copytree(ORIGINAL / prior["directory"], target / split)
        saved = load_from_disk(target / split)
        assert len(saved) == len(selected)
        assert row_hash(saved.select(range(len(existing)))) == expected_hash
        metadata["splits"][split] = {
            "directory": split, "full_size": len(full), "subset_size": len(saved),
            "full_fingerprint": full._fingerprint,
            "shuffled_fingerprint": shuffled._fingerprint,
            "subset_fingerprint": saved._fingerprint,
        }
        audit[split] = metadata["splits"][split] | {
            "source_csv_sha256": prior["source_csv_sha256"],
            "source_csv_bytes": prior["source_csv_bytes"],
            "source_arrow_files": [{"path": str(path), "bytes": path.stat().st_size,
                                    "sha256": file_hash(path)} for path in paths],
            "prefix_matches_existing_rows": len(existing),
            "prefix_row_sha256": expected_hash,
            "prepared_files": [{"path": str(path.relative_to(ROOT)),
                                "bytes": path.stat().st_size, "sha256": file_hash(path)}
                               for path in sorted((target / split).iterdir()) if path.is_file()],
        }
        if split == "train":
            audit[split]["nested_prefixes"] = {
                str(size): {"size": size,
                            "row_sha256": row_hash(saved.select(range(size)))
                            if size <= 10000 else None}
                for size in SIZES
            }
        print(f"Prepared {split}: {len(saved):,} rows; unchanged prefix {len(existing):,}", flush=True)
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = metadata | {
        "splits": audit, "train_sizes": list(SIZES), "test_size": 20000,
        "selection": "All subsets are prefixes of the complete train split shuffled with seed 42",
        "row_hash_format": "UTF-8 question + tab + answer + newline, in dataset order",
        "source_metadata": str(ORIGINAL / "metadata.json"),
        "source_metadata_sha256": file_hash(ORIGINAL / "metadata.json"),
        "evaluation": "Unchanged default shuffled test subset, no augmentation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

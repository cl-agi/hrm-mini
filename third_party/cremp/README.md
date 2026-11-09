# hrm-cremp

HRM molecular-conformation implementation, independently installable with `pip install -e third_party/cremp` from the HRM repository root.

The HRM-specific implementation is covered by the root Apache-2.0 license. Bundled Ringer utilities retain their [MIT license](LICENSE-RINGER). `SOURCE.json` identifies the combined source snapshot by file hashes.

See `SOURCE.json` for the supplied code hashes and source adaptations.

The main repository README documents generation, projection and reference scoring using public HF defaults. `geometry.json` contains data only; the code restores training provenance only for its recognized release hash, and still rejects a mismatch with the checkpoint's geometry cache. The sidechain reconstruction constants are included as package data in wheels.

## New training runs

When training data are omitted, the HF adapter writes an adjacent training-geometry file bound to the converted NPZ hash. Training retains this metadata in checkpoints and `train_geometry.json`; generation writes `generated_angles.geometry.json`, which projection uses automatically. Explicit `--train-geometry` takes priority. Custom local training caches should supply an adjacent `<cache-stem>.geometry.json` with train-only bond statistics and the cache SHA-256. Training validates the geometry metadata against the cache.

An omitted training seed is frozen from Unix time in the run directory. Resume uses the saved checkpoint seed; an explicit seed retains the requested value. Existing configurations retain their saved settings.

# neurosat

Vendored numerical implementation. SOURCE.json records original and adapted file hashes. PyTorch implementation with the original scatter compatibility shim.

## Pretrained weights

After installing this package from the main checkout:

```bash
python -m pip install ./third_party/neurosat
python -m hrm_neurosat.pretrained --device cpu
```

This verifies and loads the public RandCSPBench 3-SAT unsupervised checkpoint linked from its [NeuroSAT documentation](https://github.com/ArtLabBocconi/RandCSPBench/blob/20702c0be4ebbd0f039be67aaa899086e4d6a678/algorithms/NeuroSAT/README.md). The Google Drive file ID is `1qnq0-vNoAbJpiAw6wgxcdllLabUMaNub`; SHA256 is `6d90914da4d44c64c790afd3bfac7a19c6b1d9fd3e6860b334150714f656af99`. Downloads go to `HRM_ASSET_CACHE` (or the main package's HF asset-cache default). Run outputs and weights are written to the configured work directory.

`--checkpoint FILE` supplies an explicit local checkpoint; missing files fail. The loader validates the assignment/LCG/learned-initialization architecture and loads every state-dict key strictly. `--iterations` sets the inference budget (default 2048). The command checks model and input loading. Python callers can use `from hrm_neurosat.pretrained import load_model`.

All model tensors in this public checkpoint were verified equal to the archived
checkpoint used for E43; epoch 125/global step 661500. Container hashes differ
because the archived release had modified checkpoint metadata. The supplied
numerical implementation is unchanged. A CPU three-iteration inference check
is separate from full E43 graph-coloring evaluation, which now has a public-data entry documented in [E43](../../exp/sat/README.md).

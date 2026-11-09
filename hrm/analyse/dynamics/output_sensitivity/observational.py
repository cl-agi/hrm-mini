from __future__ import annotations
import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from hrm.paths import checkpoint_config, dataset_path
SPLIT_MANIFEST_SCHEMA_VERSION = 'hrm-mechanism-splits-v1'
DATASET_SPLIT = 'test_hard'
SEED_DIRECTORY_PATTERN = re.compile('^seed_(\\d+)$')

class ObservationalError(RuntimeError):
    """Raised when strict observational protocol requirements are violated."""

@dataclass(frozen=True)
class RunSpec:
    seed_dir: Path
    seed: int
    epoch: int
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    checkpoint_mtime_ns: int
    config_path: Path
    config: dict[str, Any]
    config_sha256: str
    config_file_sha256: str
    cycles: int

@dataclass(frozen=True)
class FixedDataset:
    loader: DataLoader
    metadata: dict[str, Any]
    dataset_size: int
    board_ids: tuple[int, ...]

def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as handle:
            while (chunk := handle.read(1024 * 1024)):
                digest.update(chunk)
    except OSError as exc:
        raise ObservationalError(f'Could not hash {path}: {exc}') from exc
    return digest.hexdigest()

def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(value).encode('utf-8'))

def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ObservationalError(f'Could not read model config {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise ObservationalError(f'Model config must be a mapping: {path}')
    if not isinstance(value.get('arch'), dict):
        raise ObservationalError(f'Model config has no mapping-valued arch: {path}')
    if not isinstance(value.get('data'), dict):
        raise ObservationalError(f'Model config has no mapping-valued data: {path}')
    return (dict(value), _sha256_bytes(raw))

def load_run_spec(seed_dir: Path, epoch: int) -> RunSpec:
    """Strictly resolve ``seed_N/model_config.json`` and ``epoch_K.pt``."""
    seed_dir = seed_dir.expanduser().resolve()
    if not seed_dir.is_dir():
        raise ObservationalError(f'Seed directory does not exist: {seed_dir}')
    match = SEED_DIRECTORY_PATTERN.match(seed_dir.name)
    if match is None:
        raise ObservationalError(f'Seed directory must be named seed_N, got {seed_dir.name!r}')
    if epoch < 0:
        raise ObservationalError('Epoch must be non-negative')
    seed = int(match.group(1))
    checkpoint = seed_dir / f'epoch_{epoch}.pt'
    config_path = checkpoint_config(checkpoint)
    if not checkpoint.is_file():
        raise ObservationalError(f'Checkpoint does not exist: {checkpoint}')
    if not config_path.is_file():
        raise ObservationalError(f'Expected model_config.json directly in seed directory: {config_path}')
    config, config_file_sha256 = _load_config(config_path)
    configured_seeds = config.get('seeds')
    if not isinstance(configured_seeds, list) or not all((isinstance(value, int) for value in configured_seeds)) or seed not in configured_seeds:
        raise ObservationalError(f'Seed {seed} from directory is absent from config seeds={configured_seeds!r}')
    configured_epochs = config.get('epochs')
    if not isinstance(configured_epochs, int) or configured_epochs <= 0 or epoch >= configured_epochs:
        raise ObservationalError(f'epoch_{epoch}.pt is outside config epochs={configured_epochs!r}')
    cycles = config.get('cycles_per_data')
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
        raise ObservationalError(f'cycles_per_data must be positive, got {cycles!r}')
    eval_splits = config.get('eval_splits', [DATASET_SPLIT])
    if not isinstance(eval_splits, list) or DATASET_SPLIT not in eval_splits:
        raise ObservationalError(f'Config does not declare {DATASET_SPLIT!r} evaluation')
    arch_name = config['arch'].get('name')
    supported = {'hrm@HRM', 'hrm_ablations@HRM', 'hrm_h_to_l_credit@HRM', 'hrm_l_driven_h@HRMLDrivenH', 'hrm_l_direction@HRMLDirection', 'hrm_role_conditioned@HRMRoleConditionedTied', 'hrm_readout@HRMReadout', 'hrm_readout@HRMMultiReadout'}
    if arch_name not in supported:
        raise ObservationalError(f'Unsupported observational architecture {arch_name!r}; expected one of {sorted(supported)}')
    for name in ('H_cycles', 'L_cycles'):
        value = config['arch'].get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ObservationalError(f'Architecture {name} must be positive, got {value!r}')
    checkpoint_stat = checkpoint.stat()
    return RunSpec(seed_dir=seed_dir, seed=seed, epoch=epoch, checkpoint=checkpoint, checkpoint_sha256=_sha256_file(checkpoint), checkpoint_bytes=checkpoint_stat.st_size, checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns, config_path=config_path, config=config, config_sha256=_sha256_mapping(config), config_file_sha256=config_file_sha256, cycles=cycles)

def load_board_indices(path: Path, *, max_boards: int, group: str | None=None) -> tuple[tuple[int, ...], str, str]:
    """Load a fixed JSON list or one group from the hashed split manifest."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ObservationalError(f'Board-index file does not exist: {path}')
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservationalError(f'Could not read board-index JSON {path}: {exc}') from exc
    raw_sha256 = _sha256_bytes(raw)
    if isinstance(value, dict) and value.get('schema_version') == SPLIT_MANIFEST_SCHEMA_VERSION:
        if group is None:
            raise ObservationalError('A mechanism split manifest requires --board-group')
        groups = value.get('groups')
        if not isinstance(groups, dict) or not isinstance(groups.get(group), dict):
            raise ObservationalError(f'Split manifest has no board group {group!r}')
        protocol = value.get('protocol')
        if not isinstance(protocol, dict) or protocol.get('dataset_split') != DATASET_SPLIT or protocol.get('dataset_size') != 20000 or (protocol.get('split_seed') != 20260802) or (protocol.get('primary_counts') != {'fit': 1024, 'val': 256, 'test': 2048}) or (protocol.get('causal_count') != 512) or (protocol.get('jacobian_count') != 128):
            raise ObservationalError('Split manifest does not describe the fixed 20,000-board test_hard protocol')
        digest_path = path.with_name(f'{path.name}.sha256')
        if not digest_path.is_file():
            raise ObservationalError(f'Split manifest SHA256 sidecar is missing: {digest_path}')
        try:
            digest_line = digest_path.read_text(encoding='ascii').rstrip('\r\n')
        except OSError as exc:
            raise ObservationalError(f'Could not read split manifest sidecar {digest_path}: {exc}') from exc
        digest, separator, filename = digest_line.partition('  ')
        if separator != '  ' or digest != raw_sha256 or filename != path.name:
            raise ObservationalError(f'Split manifest SHA256 sidecar mismatch: {digest_path}')
        selected_group = groups[group]
        expected_group_counts = {'fit': 1024, 'val': 256, 'test': 2048, 'causal': 512, 'jacobian': 128}
        if selected_group.get('count') != expected_group_counts[group]:
            raise ObservationalError(f'Split manifest group {group!r} has an invalid count')
        value = selected_group.get('indices')
    elif group is not None:
        raise ObservationalError('--board-group is only valid with an HRM mechanism split manifest')
    elif isinstance(value, dict):
        split = value.get('split', DATASET_SPLIT)
        if split != DATASET_SPLIT:
            raise ObservationalError(f'Board-index split must be {DATASET_SPLIT!r}, got {split!r}')
        value = value.get('indices')
    if not isinstance(value, list) or not value or (not all((isinstance(index, int) and (not isinstance(index, bool)) for index in value))):
        raise ObservationalError("Board-index JSON must be a non-empty integer list or {'split':'test_hard','indices':[...]}")
    indices = tuple(value)
    if any((index < 0 for index in indices)):
        raise ObservationalError('Board indices must be non-negative')
    if len(indices) != len(set(indices)):
        raise ObservationalError('Board indices must be unique')
    if max_boards <= 0:
        raise ObservationalError('max_boards must be positive')
    if len(indices) > max_boards:
        raise ObservationalError(f'Requested {len(indices)} boards exceeds --max-boards={max_boards}')
    content_sha = _sha256_bytes(_canonical_json({'split': DATASET_SPLIT, 'indices': list(indices)}).encode('utf-8'))
    return (indices, raw_sha256, content_sha)

def _load_object(identifier: str, package_prefix: str) -> Any:
    if '@' not in identifier:
        raise ObservationalError(f'Expected module@object identifier, got {identifier!r}')
    module_name, object_name = identifier.split('@', maxsplit=1)
    module = importlib.import_module(__import__('hrm.registry',fromlist=['resolve_module']).resolve_module(f'{package_prefix}{module_name}'))
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ObservationalError(f'Object {object_name!r} not found in {module.__name__}') from exc

def _resolve_dataset_path(data_config: dict[str, Any]) -> None:
    dataset_name = data_config.get('dataset_name')
    if not isinstance(dataset_name, str):
        return
    data_config['dataset_name'] = str(dataset_path(dataset_name))

def build_fixed_dataset(spec: RunSpec, board_ids: tuple[int, ...], *, batch_size: int, expected_dataset_size: int, device: torch.device) -> FixedDataset:
    """Create a deterministic, unaugmented Subset loader in board-ID order."""
    if batch_size <= 0:
        raise ObservationalError('Batch size must be positive')
    if expected_dataset_size <= 0:
        raise ObservationalError('Expected dataset size must be positive')
    data_config = dict(spec.config['data'])
    data_name = data_config.pop('name', None)
    if not isinstance(data_name, str):
        raise ObservationalError('Data config must contain a string name')
    _resolve_dataset_path(data_config)
    data_config.pop('seed', None)
    data_config.update({'augment': False, 'repeat': 1, 'partial_fill_probability': 0.0, 'partial_fill_remaining_blanks': None, 'partial_fill_on_eval': False})
    create_dataloader = _load_object(f'{data_name}@create_dataloader', package_prefix='dataset.')
    full_loader, metadata = create_dataloader(DATASET_SPLIT, batch_size, rank=0, world_size=1, seed=spec.seed, **data_config)
    if len(full_loader.dataset) != expected_dataset_size:
        raise ObservationalError(f'{DATASET_SPLIT} contains {len(full_loader.dataset)} boards; expected {expected_dataset_size}')
    if any((index >= expected_dataset_size for index in board_ids)):
        raise ObservationalError('A requested board index is outside the test_hard dataset')
    if not isinstance(metadata, dict):
        raise ObservationalError('Dataset metadata must be a mapping')
    expected_metadata = {'vocab_size': 10, 'seq_len': 82, 'is_causal': False}
    mismatches = {key: (metadata.get(key), value) for key, value in expected_metadata.items() if metadata.get(key) != value}
    if mismatches:
        raise ObservationalError(f'Unexpected Sudoku metadata: {mismatches}')
    subset = Subset(full_loader.dataset, list(board_ids))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=full_loader.collate_fn, num_workers=0, pin_memory=device.type == 'cuda')
    return FixedDataset(loader=loader, metadata=dict(metadata), dataset_size=expected_dataset_size, board_ids=board_ids)

def _clean_state_dict(value: Any) -> dict[str, Tensor]:
    state: Any = value
    if isinstance(value, dict) and value:
        if not all((isinstance(item, Tensor) for item in value.values())):
            for key in ('state_dict', 'model_state_dict', 'model'):
                candidate = value.get(key)
                if isinstance(candidate, dict) and candidate:
                    state = candidate
                    break
    if not isinstance(state, dict) or not state or (not all((isinstance(item, Tensor) for item in state.values()))):
        raise ObservationalError('Checkpoint does not contain a non-empty tensor state dict')
    cleaned: dict[str, Tensor] = {}
    for raw_key, tensor in state.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ('module.', '_orig_mod.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key in cleaned:
            raise ObservationalError(f'Checkpoint key collision after prefix cleanup: {key!r}')
        if not torch.isfinite(tensor).all().item():
            raise ObservationalError(f'Checkpoint tensor {raw_key!r} contains non-finite values')
        cleaned[key] = tensor
    return cleaned

def load_model(spec: RunSpec, metadata: Mapping[str, Any], device: torch.device) -> nn.Module:
    """Strictly instantiate and load an uncompiled eager model."""
    arch = dict(spec.config['arch'])
    identifier = arch.pop('name', None)
    if not isinstance(identifier, str):
        raise ObservationalError('Architecture config must contain a string name')
    model_class = _load_object(identifier, package_prefix='arch.')
    with torch.device(device):
        model: nn.Module = model_class(arch | dict(metadata))
    try:
        checkpoint_value = torch.load(spec.checkpoint, map_location=device, weights_only=True)
    except Exception as exc:
        raise ObservationalError(f'Could not load checkpoint {spec.checkpoint}: {exc}') from exc
    state_dict = _clean_state_dict(checkpoint_value)
    try:
        incompatibility = model.load_state_dict(state_dict, strict=True, assign=True)
    except Exception as exc:
        raise ObservationalError(f'Strict state-dict load failed for {spec.checkpoint}: {exc}') from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ObservationalError(f'State-dict mismatch: missing={incompatibility.missing_keys}, unexpected={incompatibility.unexpected_keys}')
    model.eval()
    return model

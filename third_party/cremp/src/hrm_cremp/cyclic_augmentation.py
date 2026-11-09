"""Joint, orientation-preserving cyclic ring relabeling, only for training."""
import copy
import hashlib
import random
import struct
import torch
from hrm_cremp.angle_data import AngleDataset


def rotate_item(item, residue_shift):
    length = int(item['lengths'])
    assert length in (12, 15, 18)
    shift = 3 * (int(residue_shift) % (length // 3))
    result = dict(item)
    for key in ['angles', 'atom_features', 'feat_mask', 'attn_mask', 'atom_ids']:
        result[key] = item[key].clone()
        result[key][:length] = torch.roll(item[key][:length], -shift, dims=0)
    # New sequence coordinates stay arange(18). No reversal or chirality change.
    result['cyclic_residue_shift'] = torch.tensor(shift // 3)
    return result


class CyclicTrainingDataset(AngleDataset):
    def __init__(self, path, seed):
        super().__init__(path)
        self.augmentation_seed = seed
        self.augmentation_rng = random.Random(seed)
        self.augmentation_draws = 0
        self.augmentation_digest = bytes(32)
        self.shift_counts = {str(n): [0] * n for n in [4, 5, 6]}

    def __getitem__(self, index, ignore_zero_center=False):
        item = super().__getitem__(index, ignore_zero_center=ignore_zero_center)
        n = int(item['lengths']) // 3
        shift = self.augmentation_rng.randrange(n)
        self.augmentation_draws += 1
        self.shift_counts[str(n)][shift] += 1
        self.augmentation_digest = hashlib.sha256(self.augmentation_digest + struct.pack('<qq', int(index), shift)).digest()
        return rotate_item(item, shift)

    def augmentation_statistics(self):
        return dict(seed=self.augmentation_seed, row_draws=self.augmentation_draws,
                    draw_index_shift_sha256=self.augmentation_digest.hex(),
                    counts_by_ring_residues={k: list(v) for k,v in self.shift_counts.items()})

    def augmentation_state_dict(self):
        return dict(rng=self.augmentation_rng.getstate(), draws=self.augmentation_draws,
                    digest=self.augmentation_digest, counts=copy.deepcopy(self.shift_counts))

    def load_augmentation_state_dict(self, state):
        self.augmentation_rng.setstate(state['rng'])
        self.augmentation_draws = state['draws']
        self.augmentation_digest = state['digest']
        self.shift_counts = copy.deepcopy(state['counts'])

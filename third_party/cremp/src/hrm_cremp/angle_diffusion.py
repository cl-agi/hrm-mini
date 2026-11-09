"""Official wrapped Gaussian/cosine20 process, loss and ancestral DDPM sampler."""
import torch
from hrm_cremp.ringer_support import FEATURES
from ringer.data.noised import NoisedDataset
from ringer.utils.losses import radian_smooth_l1_loss
from ringer.utils.sampling import sample_batch


def noised_dataset(dataset):
    return NoisedDataset(dataset,timesteps=20,beta_schedule='cosine',
                         angular_variance=1.,nonangular_variance=1.,mask_noise=False)


def noise_loss(prediction,batch):
    # Official conditional.json: use_feat_mask=false. Do not silently change this
    # to graph-support masking; absent sidechain channels still predict noise.
    mask=batch['attn_mask'].bool()
    terms=[radian_smooth_l1_loss(prediction[...,i][mask],batch['known_noise'][...,i][mask],
                               beta=torch.pi/10,circle_penalty=0.) for i in range(12)]
    return torch.stack(terms).mean()


@torch.no_grad()
def generate(model,dataset,molecules,k,seed,device):
    """Only condition() is read. Reference angles/counts never select initial states or K."""
    if k<1:
        raise ValueError('K must be fixed positive integer')
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    conditions=[dataset.condition(m) for m in molecules for _ in range(k)]
    lengths=[int(c['lengths']) for c in conditions]
    features=torch.stack([c['atom_features'] for c in conditions]).to(device)
    noised=noised_dataset(dataset)
    samples=sample_batch(model,noised,lengths,atom_features=features,
                         final_timepoint_only=True,disable_pbar=True)
    return samples

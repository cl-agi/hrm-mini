"""A: retain RINGER diffusion while replacing only its noise-prediction network."""
import torch
from torch import nn
from hrm_cremp.ringer_support import FEATURES
from hrm_cremp.arch.hrm_flow_bias import HRMFlowBias
from hrm_cremp.arch.rt_flow_bias import RecurrentTransformer
from hrm_cremp.arch.layers import Transformer
from ringer.models.components.embeddings import GaussianFourierProjection


def configuration(family='hrm', width=384, intermediate=512, heads=12,
                  atom_feature_size=178, dtype='float32'):
    if width % heads or (width // heads) % 2:
        raise ValueError('Attention head size must be an even integer')
    return dict(family=family, width=width, intermediate=intermediate, heads=heads,
                atom_feature_size=atom_feature_size, dtype=dtype, pad=18,
                channels=12, H_cycles=2, L_cycles=6, rt_cycles=7,
                full_bptt=True, fresh_carry_per_denoiser_call=True)


class RecurrentDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_config = dict(config)
        c = config; width = c['width']; family = c['family']
        params = dict(vocab_size=12, seq_len=18, num_layers=2 if family=='hrm' else 4,
                      hidden_size=width, intermediate_size=c['intermediate'],
                      head_dim=width//c['heads'], is_causal=False, norm_eps=1e-6,
                      rope_theta=10000., H_cycles=2, L_cycles=6,
                      cycles=1 if family=='plain' else 7, bptt=True,
                      forward_dtype=c['dtype'], flow_lambda=0., preserve_first_zero=False)
        self.backbone = (HRMFlowBias if family=='hrm' else RecurrentTransformer)(params)
        # No categorical mixture / teacher interpolation is used in continuous diffusion.
        del self.backbone.embed
        angle_dim = 2*width//3
        self.angles_to_hidden = nn.Linear(12, angle_dim)
        self.graph_to_hidden = nn.Linear(c['atom_feature_size'], width-angle_dim)
        self.time_embed = GaussianFourierProjection(width)
        self.n_inputs = 12
        self.ft_is_angular = [True]*12
        self.ft_names = list(FEATURES)

    def forward(self, inputs, timestep, attention_mask, position_ids=None,
                atom_ids=None, atom_features=None):
        if atom_features is None or attention_mask is None:
            raise ValueError('Graph features and actual attention mask are required')
        valid = attention_mask.bool()
        if not valid.any(-1).all():
            raise ValueError('Empty molecules are not valid inputs')
        # All repeated blocks receive the same valid-key mask, including early H/L cycles.
        for module in self.backbone.modules():
            if isinstance(module, Transformer):
                module.valid_mask = valid
        x = torch.cat((self.angles_to_hidden(inputs), self.graph_to_hidden(atom_features)), -1)
        x = x + self.time_embed(timestep.reshape(-1)).unsqueeze(1)
        x = x.to(getattr(torch, self.model_config['dtype']))
        carry = {k: v.expand(len(x), x.shape[1], -1).clone()
                 for k, v in self.backbone.initial_carry.items()}
        # The unused mixture argument is an empty tensor, never a target/noise interpolation.
        _, prediction = self.backbone.forward_embedded(carry, x, x.new_empty((0,)))
        return prediction.float()


def make_model(config):
    if config['family'] != 'ringer':
        return RecurrentDenoiser(config)
    from ringer.models.bert_for_diffusion import BertForDiffusionBase
    from transformers.models.bert.configuration_bert import BertConfig
    from transformers.models.bert import modeling_bert
    from ringer.models.components.attention import BertSelfAttentionWithCyclicEncoding
    # Transformers 4.49 added a class registry after RINGER's original monkeypatch
    # was written. Keep the official cyclic implementation, not SDPA/original BERT.
    modeling_bert.BERT_SELF_ATTENTION_CLASSES['eager']=BertSelfAttentionWithCyclicEncoding
    c = config; angle_dim = 2*c['width']//3
    bert = BertConfig(hidden_size=angle_dim, intermediate_size=c['intermediate'],
                      num_attention_heads=c['heads'], num_hidden_layers=12,
                      max_position_embeddings=18, position_embedding_type='cyclic_relative_key',
                      hidden_dropout_prob=.1, attention_probs_dropout_prob=.1)
    bert._attn_implementation='eager'
    model = BertForDiffusionBase(bert, ft_is_angular=[True]*12, ft_names=FEATURES,
                               time_encoding='gaussian_fourier', decoder='mlp',
                               atom_feature_size=c['atom_feature_size'],
                               atom_feature_embed_size=c['width']-angle_dim)
    model.model_config = dict(config)
    assert all(isinstance(layer.attention.self,BertSelfAttentionWithCyclicEncoding)
               for layer in model.encoder.layer)
    return model


def budget(model):
    c = model.model_config; family = c['family']
    blocks = {'hrm':28, 'rt':28, 'plain':4, 'ringer':12}[family]
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                forward_blocks_per_denoiser_call=blocks,
                gradient_blocks_per_training_call=blocks,
                unique_blocks=12 if family=='ringer' else 4,
                diffusion_steps=20, denoiser_nfe_per_generated_candidate=20,
                forward_blocks_per_generated_candidate=20*blocks,
                fresh_carry_per_call=True, full_bptt=True,
                architecture_comparison='HRM/RT matched; official RINGER separate 12-layer reference',
                modification='Denoiser replacement only; original wrapped diffusion retained',
                position_encoding='official cyclic relative key' if family=='ringer' else 'existing HRM/RT RoPE')

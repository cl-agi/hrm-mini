"""Repeated supervision at fixed DDPM input; this still uses diffusion."""
import hrm.analyse.cremp.evaluation.bootstrap as bootstrap
import torch
from hrm_cremp.angle_model import RecurrentDenoiser, budget as original_budget
from hrm_cremp.arch.layers import Transformer


class OuterDenoiser(RecurrentDenoiser):
    def __init__(self, config):
        if config['family'] not in ('hrm', 'rt'):
            raise ValueError('The retained/reset comparison supports HRM and RT only')
        super().__init__(config)
        self.outer_rounds = int(config.get('outer_rounds', 1))
        self.carry_mode = config.get('carry_mode', 'retain')
        if self.outer_rounds not in (1, 2) or self.carry_mode not in ('retain', 'reset'):
            raise ValueError('Expected R1/R2 with retain/reset carry')

    def transition(self, inputs, timestep, attention_mask, atom_features, carry=None):
        """One 28-block update. Returned carry is detached, prediction is not.

        The input embeddings are recomputed on every call, including after an
        optimizer step. No target/noise label is accepted by this interface.
        """
        if atom_features is None or attention_mask is None:
            raise ValueError('Graph features and actual attention mask are required')
        valid = attention_mask.bool()
        if not valid.any(-1).all():
            raise ValueError('Empty molecules are not valid inputs')
        for module in self.backbone.modules():
            if isinstance(module, Transformer):
                module.valid_mask = valid
        x = torch.cat((self.angles_to_hidden(inputs), self.graph_to_hidden(atom_features)), -1)
        x = x + self.time_embed(timestep.reshape(-1)).unsqueeze(1)
        x = x.to(getattr(torch, self.model_config['dtype']))
        if carry is None or self.carry_mode == 'reset':
            carry = {k: v.expand(len(x), x.shape[1], -1).clone()
                     for k, v in self.backbone.initial_carry.items()}
        else:
            carry = {k: v.detach() for k, v in carry.items()}
            for key in self.backbone.initial_carry:
                if carry[key].shape != x.shape:
                    raise ValueError('Carry does not match this batch shape')
        carry, prediction = self.backbone.forward_embedded(carry, x, x.new_empty((0,)))
        return carry, prediction.float()

    def forward(self, inputs, timestep, attention_mask, position_ids=None,
                atom_ids=None, atom_features=None):
        # Every DDPM sampler call starts a new local episode. Never carry a
        # latent from one diffusion state/timestep to the next.
        carry = None
        for _ in range(self.outer_rounds):
            carry, prediction = self.transition(inputs, timestep, attention_mask, atom_features, carry)
        return prediction


def budget(model):
    result = original_budget(model)
    result.update(outer_rounds=model.outer_rounds, carry_mode=model.carry_mode,
        fresh_carry_per_call=True, carry_detached_between_outer_updates=True,
        forward_blocks_per_training_update=28, gradient_blocks_per_training_update=28,
        optimizer_steps_per_fresh_noised_batch=model.outer_rounds,
        forward_blocks_per_denoiser_call=28,
        forward_blocks_per_ddpm_call=28 * model.outer_rounds,
        ddpm_wrapper_calls_per_generated_candidate=20,
        denoiser_nfe_per_generated_candidate=20 * model.outer_rounds,
        backbone_transition_nfe_per_generated_candidate=20 * model.outer_rounds,
        forward_blocks_per_generated_candidate=560 * model.outer_rounds,
        modification='Diffusion retained; fixed-case deep supervision with retained/reset latent carry',
        reset_control='Reset repeats the same noised case with the same optimizer-step count; at inference its deterministic repeated calls are redundant but charged')
    return result

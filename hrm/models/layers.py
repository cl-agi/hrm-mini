from typing import Sequence, Optional
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
import pydantic

CosSin = tuple[Tensor, Tensor]
Carry = dict[str, Tensor]

def trunc_normal_init_(tensor: Tensor, std: float = 1.0):
    """Fast approximate truncated normal initialization. Fairly accurate."""
    return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)

def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin):
    # q, k: [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, batch_output_dims: Sequence[int] = (), init_std: float | None = None, output_scale: float = 1.0, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.output_scale = output_scale

        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((*batch_output_dims, out_features, in_features), **kwargs), std=init_std if init_std is not None else 1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((math.prod(batch_output_dims) * out_features, ), **kwargs))

    def forward(self, input: Tensor) -> Tensor:
        out = F.linear(input, self.weight.view(-1, self.in_features).to(input.dtype), self.bias.to(input.dtype) if self.bias is not None else None)
        return out if self.output_scale == 1.0 else out * self.output_scale


class CastedScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, cast_to: torch.dtype, base_width: int | None = None):
        super().__init__()
        self.cast_to = cast_to

        # Scale to the same std as most parameters
        self.scale = (base_width if base_width is not None else embedding_dim) ** 0.5
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=1.0 / self.scale)
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.scale * self.weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    """RoPE over token index, or over (row, col) of a square grid when `rope_2d` is set.

    The 2D variant splits each head's channels in half, rotating the first half by the token's row
    and the second half by its column, so the attention logit depends on the 2D offset between two
    cells rather than on their distance along the flattened sequence. Sudoku (9x9 flattened to 81
    tokens) and Maze (30x30 to 900) are grids, and 1D RoPE places e.g. cells (0,8) and (1,0) one
    position apart while they sit in different rows. Any leading non-grid tokens (the BOS pad) get
    row = col = -1, a position no cell occupies.

    Channel layout is chosen to survive `rotate_half`, which pairs channel i with channel i + dim/2:
    `emb = cat(f, f)` with `f = cat(f_row, f_col)` gives [row, col, row, col], so each rotated pair
    shares one axis and one frequency, exactly as the 1D case pairs a channel with its own copy.
    """
    def __init__(self, dim, max_position_embeddings, base, rope_2d: bool = False, device=None):
        super().__init__()
        if rope_2d:
            side = math.isqrt(max_position_embeddings)
            num_prefix = max_position_embeddings - side * side
            assert num_prefix >= 0
            # Half the channels per axis, hence half as many frequencies per axis as the 1D case.
            axis_dim = dim // 2
            inv_freq = 1.0 / (base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim))

            pos = torch.arange(max_position_embeddings, dtype=torch.float32, device=device) - num_prefix
            row = torch.where(pos < 0, pos, torch.div(pos, side, rounding_mode="floor"))
            col = torch.where(pos < 0, pos, pos % side)
            freqs = torch.cat((torch.outer(row, inv_freq), torch.outer(col, inv_freq)), dim=-1)
        else:
            # RoPE
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
            freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, seq_len=None):
        if seq_len is None:
            return self.cos_cached, self.sin_cached
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class AxialRotaryEmbedding(nn.Module):
    """Fixed row/column RoPE for an unpadded square grid preceded by BOS.

    ``rotate_half`` pairs channel i with i + dim/2, so the first half
    contains row frequencies followed by column frequencies; the second
    half repeats that arrangement. Each axis receives dim/4 complex pairs.
    BOS and cell (0, 0) both have zero phase, with no learned position vector.
    All caches are built up front and excluded from checkpoints.
    """
    def __init__(self, dim: int, sizes: Sequence[int], base: float, device=None):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("2D RoPE requires head_dim divisible by 4")
        axis_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim))
        for size in sizes:
            coordinates = torch.arange(size, dtype=torch.float32, device=device)
            zero = torch.zeros(1, dtype=torch.float32, device=device)
            rows = torch.cat((zero, coordinates.repeat_interleave(size)))
            columns = torch.cat((zero, coordinates.repeat(size)))
            freqs = torch.cat((torch.outer(rows, inv_freq), torch.outer(columns, inv_freq)), dim=-1)
            emb = torch.cat((freqs, freqs), dim=-1)
            seq_len = size * size + 1
            self.register_buffer(f"cos_{seq_len}", emb.cos(), persistent=False)
            self.register_buffer(f"sin_{seq_len}", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> CosSin:
        # seq_len is a static shape under torch.compile(dynamic=False).
        return getattr(self, f"cos_{seq_len}"), getattr(self, f"sin_{seq_len}")


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, **kwargs):
        super().__init__()
        self.up_proj = CastedLinear(hidden_size, intermediate_size, bias=False, **kwargs)
        self.down_proj = CastedLinear(intermediate_size, hidden_size, bias=False, **kwargs)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, **kwargs):
        super().__init__()
        self.gate_up_proj = CastedLinear(hidden_size, intermediate_size, bias=False, batch_output_dims=(2, ), **kwargs)
        self.down_proj = CastedLinear(intermediate_size, hidden_size, bias=False, **kwargs)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, is_causal, enable_pe=True, qk_norm=False, norm_eps=1e-6, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.enable_pe = enable_pe
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps

        self.qkv_proj = CastedLinear(hidden_size, self.num_heads * self.head_dim, bias=False, batch_output_dims=(3, ), **kwargs)
        self.o_proj = CastedLinear(head_dim * num_heads, hidden_size, bias=False, **kwargs)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin] = None, attn_mask: Optional[Tensor] = None, valid_mask: Optional[Tensor] = None) -> Tensor:
        # hidden_states, qkv: [..., seq_len, hidden_size]
        qkv = self.qkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        qkv = rearrange(qkv, "... (h hd) -> ... h hd", h=self.num_heads)
        query, key, value = qkv.chunk(3, dim=-1)
        if self.qk_norm:
            query = F.rms_norm(query, (self.head_dim,), eps=self.norm_eps)
            key = F.rms_norm(key, (self.head_dim,), eps=self.norm_eps)
        if self.enable_pe and cos_sin is not None:
            # Rotary embedding
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)
        # PyTorch SDPA attention
        # query, key, value: [... x seq_len x num_heads x head_dim]
        if valid_mask is not None:
            attn_mask = valid_mask[:, None, None, :]
        elif attn_mask is None:
            attn_mask = getattr(self, "graph_mask", None)
        attn_output = F.scaled_dot_product_attention(
            query.transpose(-2, -3), 
            key.transpose(-2, -3), 
            value.transpose(-2, -3), 
            attn_mask=attn_mask, 
            is_causal=self.is_causal).transpose(-2, -3)
        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = rearrange(attn_output, "... h hd -> ... (h hd)")
        return self.o_proj(attn_output)

class TransformerConfig(pydantic.BaseModel):
    seq_len: int

    num_layers: int

    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool = False
    enable_pe: bool = True

    norm_eps: float
    rope_theta: float
    rope_steps: Optional[int] = None
    use_rope: bool = True
    rope_2d: bool = False
    rope_type: str = "1d"
    allowed_sizes: tuple[int, ...] | None = None
    mup_base_width: int | None = None
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256
    norm_type: str = "post"
    norm_mode: str = "post"
    residual_alpha: float = 0.75
    residual_beta: float = 0.25
    qk_norm: bool = False

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.is_mlp_mixer = config.is_mlp_mixer
        self.norm_type = config.norm_type
        self.norm_mode = config.norm_mode
        self.residual_alpha = config.residual_alpha
        self.residual_beta = config.residual_beta
        if self.norm_mode == "bounded_scaled_pre":
            if not 0 <= self.residual_alpha < 1:
                raise ValueError("residual_alpha must lie in [0, 1)")
            if self.residual_beta <= 0:
                raise ValueError("residual_beta must be positive")
        if self.is_mlp_mixer:
            self.mlp_t = SwiGLU(config.seq_len, config.mlp_mixer_intermediate_size)
            self.mlp = SwiGLU(config.hidden_size, config.mlp_mixer_intermediate_size)
        else:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.hidden_size // config.head_dim,
                is_causal=config.is_causal,
                enable_pe=config.enable_pe, qk_norm=config.qk_norm, norm_eps=config.norm_eps
            )
            self.mlp = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size
            )

        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.norm_type == "pre":
            if self.is_mlp_mixer:
                x = x.transpose(-1, -2)
                x = x + self.mlp_t(self.norm(x))
                x = x.transpose(-1, -2)
            else:
                x = x + self.attn(self.norm(x), **kwargs)
            return x + self.mlp(self.norm(x))
        if self.norm_mode == "bounded_scaled_pre":
            alpha, beta = self.residual_alpha, self.residual_beta
            if self.is_mlp_mixer:
                x = x.transpose(-1, -2)
                x = alpha * x + beta * self.mlp_t(self.norm(x))
                x = x.transpose(-1, -2)
            else:
                x = alpha * x + beta * self.attn(self.norm(x), **kwargs)
            return alpha * x + beta * self.mlp(self.norm(x))

        # Preserve the original post-norm path bit for bit.
        if self.is_mlp_mixer:
            x = x.transpose(-1, -2)
            x = self.norm(x + self.mlp_t(x))
            x = x.transpose(-1, -2)
        else:
            x = self.norm(x + self.attn(x, **kwargs))
        return self.norm(x + self.mlp(x))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.allowed_seq_lens = None
        if config.allowed_sizes is not None:
            if not config.allowed_sizes or any(n <= 0 for n in config.allowed_sizes):
                raise ValueError("allowed_sizes must contain positive grid sizes")
            if len(set(config.allowed_sizes)) != len(config.allowed_sizes):
                raise ValueError("duplicate grid sizes")
            self.allowed_seq_lens = tuple(n*n+1 for n in config.allowed_sizes)
            if config.seq_len not in self.allowed_seq_lens:
                raise ValueError("seq_len does not match an allowed grid")
            if config.is_mlp_mixer and self.allowed_seq_lens != (config.seq_len,):
                raise ValueError("Mixer requires a fixed sequence length")
        if not config.use_rope:
            self.rotary_emb = None
        elif config.rope_type == "2d":
            self.rotary_emb = AxialRotaryEmbedding(config.head_dim, config.allowed_sizes, config.rope_theta)
        else:
            length = max(self.allowed_seq_lens) if self.allowed_seq_lens else config.seq_len
            self.rotary_emb = RotaryEmbedding(config.head_dim, length, config.rope_theta, rope_2d=config.rope_2d)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm_f = lambda x: x
        if config.norm_type == "pre":
            self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)
        self.valid_mask = None

    def forward(self, h: Tensor, attn_mask: Optional[Tensor] = None, use_rope: bool = True) -> Tensor:
        length = h.shape[-2]
        if self.allowed_seq_lens is not None and length not in self.allowed_seq_lens:
            raise ValueError(f"Unsupported sequence length {length}")
        cos_sin = self.rotary_emb(length) if use_rope and self.rotary_emb is not None else None
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask, cos_sin=cos_sin, valid_mask=self.valid_mask)
        return self.norm_f(h)


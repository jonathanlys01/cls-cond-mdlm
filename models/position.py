"""
WIP
"""

import math

import flash_attn.layers.rotary as flash_rotary
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


#################################################################################
#                                    NoPE                                       #
#################################################################################


class NoPE(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                                   ALiBi                                       #
#################################################################################

# https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742


def _get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + _get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def get_slopes(n, return_tensor=True):
    slopes = _get_slopes(n)
    if return_tensor:
        return torch.tensor(slopes)
    return slopes


class ALiBiPE(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        epsilon_idx: int,
        alpha: float = None,
        autocast_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.epsilon_idx = epsilon_idx

        self.alpha = alpha
        self.autocast_dtype = autocast_dtype

    def _get_alpha(self, L):
        """
        Average column wise sum of the bias matrix.
        Returned if alpha is not provided.
        """
        return (L * L - 1) / 3

    def forward(self, qkv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        qkv: (B, H, 3, L, D)
        indices: (B, L)
        """

        B, H, _, L, _ = qkv.shape

        if self.alpha is None:
            print("Using default alpha")
            self.alpha = self._get_alpha(L)

        bias = indices.unsqueeze(1) + indices.unsqueeze(2)  # (B, L, L)

        bias = bias.unsqueeze(1).expand(B, H, L, L)

        bias = -self.alpha * bias.float()
        bias = bias.to(dtype=qkv.dtype)

        q, k, v = qkv.unbind(dim=-3)  # (B, H, L, D)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=bias, is_causal=False)

        return out


#################################################################################
#                                 Absolute                                      #
#################################################################################


class APE(torch.nn.Module):
    """
    Absolute Positional Embeddings (APE) module.
    -> The indices are shifted to ignore epsilon tokens.
    -> Need the token indices to be passed to the forward method.
    """

    def __init__(self, dim, default_seq_len, base_frequency=10_000, data_dependent=False, epsilon_idx=None):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.base_frequency = base_frequency
        self.cached_pe = None
        if data_dependent:
            assert epsilon_idx is not None, "epsilon_idx must be provided for data-dependent APE"
            self.epsilon_idx = epsilon_idx

        else:
            self.epsilon_idx = None

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base_frequency) / dim)).to(device)

        self.register_buffer("div_term", div_term)

        self._get_ape(default_seq_len, device)

        if data_dependent:
            print(f"Using APE with base frequency: {base_frequency} and epsilon_idx: {epsilon_idx}")
        else:
            print(f"Using APE with base frequency: {base_frequency} (not data-dependent)")

    def _get_ape(self, seq_len, device):
        if self.cached_pe is None or self.cached_pe.shape[0] < seq_len:
            print("Recomputing APE")
            position = torch.arange(seq_len, device=device).unsqueeze(1)
            pe = torch.zeros(seq_len, self.dim, device=device)
            pe[:, 0::2] = torch.sin(position * self.div_term)
            pe[:, 1::2] = torch.cos(position * self.div_term)
            self.cached_pe = pe
        return self.cached_pe[:seq_len].to(device)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, seq_len, dim = x.shape
        device = x.device

        # Get positional embeddings
        _pe = self._get_ape(seq_len, device)  # Shape: (seq_len, dim)

        if self.epsilon_idx is not None:
            is_eps = indices == self.epsilon_idx  # (B, seq_len)
            c_sum = torch.cumsum(is_eps, dim=1)  # (B, seq_len)

            offset = torch.arange(seq_len, device=device).expand(B, -1) - c_sum  # (B, seq_len)

            pe = _pe[offset, :]
            pe[is_eps] = 0  # Zero out embeddings for epsilon tokens

        else:
            # Directly broadcast positional embeddings
            pe = _pe.unsqueeze(0).expand(B, -1, -1)
        return x + pe


#################################################################################
#                                 Rotary                                      #
#################################################################################


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


class RoPE(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _get(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)

            # freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            freqs = torch.outer(t, self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached

    def apply_rotary_pos_emb(self, qkv):
        cos, sin = self.forward(qkv)
        return apply_rotary_pos_emb(qkv, cos, sin)

    def forward(self, x):
        cos, sin = self._get(x)
        return self.apply_rotary_pos_emb(x, cos, sin)


######################################   Tests   ########################################


def benchmark_attn():
    B = 7
    L = 256
    H = 8
    dim = 512

    device = torch.device("cuda")
    acast_dtype = torch.bfloat16

    alibi = ALiBiPE(num_heads=H, epsilon_idx=2, alpha=0.1, autocast_dtype=acast_dtype, in_bias=False)

    qkv = torch.randn(B, H, 3, L, dim // H, device=device, requires_grad=True)  # B, H, 3, L, D

    q, k, v = qkv.unbind(dim=-3)  # B, H, L, D

    indices = torch.ones(B, L, device=device)
    indices.index_fill_(1, torch.randint(0, L, (int(0.15 * L),), device=device), 2)
    where = indices == 2  # noqa: PLR2004

    bias = where.unsqueeze(1) + where.unsqueeze(2)
    bias = -0.1 * bias.float()

    print(bias.shape)
    print(q.shape, k.shape, v.shape)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False)

    out_alibi = alibi(qkv, where)

    print(torch.norm(out_sdpa - out_alibi))


if __name__ == "__main__":
    benchmark_attn()

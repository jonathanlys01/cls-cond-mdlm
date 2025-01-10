"""
WIP
"""

import math

import torch


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


class ALiBi(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.cached_bias = self._dist(torch.zeros((dim, dim)))
        self.slopes = torch.tensor(get_slopes(n_heads))

    def _dist(self, x: torch.Tensor):
        dim = x.shape[-1]
        base = torch.arange(dim)
        distance_matrix = torch.abs(base.unsqueeze(0) - base.unsqueeze(1))
        return distance_matrix

    def forward(self, x: torch.Tensor):
        if self.dim != x.shape[-1]:  # recompute bias if dimensionality changes
            self.cached_bias = self._dist(x)
        bias = self.cached_bias
        return x + bias

    def with_slopes(self, slopes):
        self.cached_bias = self.cached_bias * slopes
        return self.cached_bias


# Adapted from flash-attn
class RoPE(torch.nn.Module):
    def __init__(self, head_dim, rotated_heads, base=10_000):
        super().__init__()
        self.head_dim = head_dim
        self.rotated_heads = rotated_heads

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

        self.cached_cos = None
        self.cached_sin = None
        self.seq_len_cached = None

    def _precompute_rotary(self, rotated_heads=None):
        if rotated_heads is None:
            return None, None

        # create (L, n_heads, head_dim)
        t = torch.arange(self.seq_len_cached).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)

        cos = freqs.cos()
        sin = freqs.sin()

        return cos, sin

    def forward(self, x: torch.Tensor, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


if __name__ == "__main__":
    rope = RoPE(32, [i % 2 for i in range(8)])
    # create (B, L, n_heads, head_dim)
    x = torch.randn(5, 10, 8, 32)

    cos, sin = rope(x)

    print(cos.shape, sin.shape)

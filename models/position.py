"""
WIP
"""

import math
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import _score_mod_signature, flex_attention
from tqdm import tqdm


# Original implementation from ALiBi paper

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


#### Absolute position embeddings (see Attention is All You Need)


class APE(torch.nn.Module):
    """
    Absolute Positional Embeddings (APE) module.
    -> The indices are shifted to ignore epsilon tokens.
    -> Need the token indices to be passed to the forward method.
    """

    def __init__(self, dim, base_frequency=10_000, epsilon_idx=None):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.base_frequency = base_frequency
        self.cached_pe = None
        self.epsilon_idx = epsilon_idx

    def _get_ape(self, seq_len, device):
        if self.cached_pe is None or self.cached_pe.shape[0] < seq_len:
            position = torch.arange(seq_len, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.dim, 2, device=device) * (-math.log(self.base_frequency) / self.dim)
            )
            pe = torch.zeros(seq_len, self.dim, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.cached_pe = pe
        return self.cached_pe[:seq_len]

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        device = x.device

        _pe = self._get_ape(seq_len, device)

        # Data-dependent positional encodings (ignore epsilon tokens)
        if self.epsilon_idx is not None:
            valid_positions = indices != self.epsilon_idx  # mask
            valid_indices = torch.nonzero(valid_positions).squeeze(1)
            pe = torch.zeros_like(_pe)
            pe[valid_positions] = _pe[: valid_indices.shape[0]]
        else:
            pe = _pe

        return x + pe


def test_ape():
    dim = 2
    ape = APE(dim=dim, epsilon_idx=2)
    seq_len = 5
    device = torch.device("cpu")

    # Sample indices and input
    indices = torch.ones(seq_len, device=device)

    indices[2] = 2  # Force an epsilon token at index 2
    x = torch.zeros(1, seq_len, dim, device=device)
    result = ape(x, indices)

    print(result)

    indices[2] = 1  # Force an epsilon token at index 2
    result = ape(x, indices)
    print(result)


#### Data-dependent ALiBi PE, using flex attention

# Vanilla implementation, from
# https://github.com/pytorch-labs/attention-gym/blob/main/attn_gym/mods/alibi.py


def generate_alibi_bias(H: int) -> _score_mod_signature:
    """
    H is the number of heads in the attention module.
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (kv_idx - q_idx) * scale
        return score + bias

    return alibi_mod


def test_alibi():
    H = 8
    dim = 128
    device = torch.device("cuda")
    # alibi_mod = generate_alibi_bias(H)
    N = 10_000

    # Compile flex_attention for speed
    start = time.time()
    compiled_flex_attention = torch.compile(flex_attention, fullgraph=True, mode="max-autotune")
    q = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
    k = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
    v = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
    out = compiled_flex_attention(q, k, v, enable_gqa=True)
    out.sum().backward()

    print(f"Compilation + warmup took {time.time() - start:.2f}s")

    for _ in tqdm(range(N), desc="Flex (no op)"):
        q = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
        k = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
        v = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)

        out = compiled_flex_attention(q, k, v, enable_gqa=True)

        out.sum().backward()

    print(out.shape)

    for _ in tqdm(range(N), desc="Flash/Efficient"):
        q = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
        k = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
        v = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v)

        out.sum().backward()

    print(out.shape)

    for _ in tqdm(range(N), desc="Math"):
        q = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
        k = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)
        v = torch.randn(1, 5, H, dim // H, device=device, requires_grad=True)

        with sdpa_kernel([SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(q, k, v)

        out.sum().backward()

    print(out.shape)


# Tests

if __name__ == "__main__":
    test_alibi()

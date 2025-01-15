"""
WIP
"""

import math
import time

import flash_attn
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import _score_mod_signature, flex_attention
from tqdm import tqdm


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


# Flex attention with ALiBi bias, definition of score_mod
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


class ALiBiPE(torch.nn.Module):
    def __init__(self, num_heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

        self.score_mod = generate_alibi_bias(num_heads)

        self.compiled_fn = torch.compile(flex_attention, fullgraph=True, mode="max-autotune", dynamic=False)

    def forward(self, qkv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, L, THREE, H, D = qkv.shape
        assert THREE == 3, "qkv must have 3 dimensions"  # noqa: PLR2004

        q, k, v = qkv.unbind(dim=-3)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.compiled_fn(q, k, v, score_mod=self.score_mod)

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
        return self.cached_pe[:seq_len]  # (seq_len, dim)

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
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


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


#################################################################################


def test_ape():
    import io

    from PIL import Image

    def _simple_viz(pe, f):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 10))
        sns.set_style("whitegrid")
        sns.heatmap(pe, cmap="coolwarm", cbar=True, xticklabels=16, yticklabels=16)

        plt.title(f"APE (f={f:.1f})")
        plt.xlabel("Dimension")
        plt.ylabel("Position")
        buff = io.BytesIO()
        plt.savefig(buff, format="png")
        plt.close()
        return Image.open(buff)

    dim = 8  # head dim
    seq_len = 8

    ape = APE(dim=dim, epsilon_idx=2, data_dependent=False, base_frequency=dim, default_seq_len=seq_len)
    device = torch.device("cuda")

    # Sample indices and input
    indices = torch.ones(1, seq_len, device=device)
    indices.index_fill_(1, torch.randint(0, seq_len, (int(0.15 * seq_len),), device=device), 2)
    x = torch.zeros(1, seq_len, dim, device=device)
    result = ape(x, indices)

    _simple_viz(result[0].cpu().detach().numpy(), dim).save("ape_with_eps.png")

    ape.epsilon_idx = 0  # No epsilon tokens
    result = ape(x, indices)
    _simple_viz(result[0].cpu().detach().numpy(), dim).save("ape_no_eps.png")

    return

    def generate_gif(base_frequencies, filename="ape_sweep.gif"):
        images = []
        for base_freq in tqdm(base_frequencies, desc="Generating GIF"):
            ape = APE(dim=dim, epsilon_idx=2, data_dependent=True, base_frequency=base_freq)
            result = ape(x, indices)
            img = _simple_viz(result[0].cpu().detach().numpy(), base_freq)
            images.append(img)

        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )

    log_freqs = torch.linspace(math.log(64), math.log(10_000), steps=50)

    base_frequencies = torch.exp(log_freqs).tolist()
    generate_gif(base_frequencies)


def benchmark_attn():
    B = 8
    L = 128
    H = 12
    dim = 768

    device = torch.device("cuda")
    # alibi_mod = generate_alibi_bias(H)
    N = 10_000

    DTYPE = torch.float32

    # Compile flex_attention for speed
    start = time.time()

    compiled_flex_attention = torch.compile(flex_attention, fullgraph=True, mode="max-autotune", dynamic=False)
    q = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
    k = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
    v = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
    out = compiled_flex_attention(q, k, v)
    out.sum().backward()

    print(f"Compilation + warmup took {time.time() - start:.2f}s")

    for _ in tqdm(range(N), desc="Flex (no op)"):
        q = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
        k = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
        v = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = compiled_flex_attention(q, k, v)

        out.sum().backward()
        torch.cuda.empty_cache()

    print(out.shape)

    for _ in tqdm(range(N), desc="Flash/Efficient"):
        q = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
        k = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
        v = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v)

        out.sum().backward()

        torch.cuda.empty_cache()

    print(out.shape)

    for _ in tqdm(range(N), desc="Math"):
        q = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
        k = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)
        v = torch.randn(B, L, H, dim // H, device=device, requires_grad=True, dtype=DTYPE)

        with sdpa_kernel([SDPBackend.MATH]):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = F.scaled_dot_product_attention(q, k, v)

        out.sum().backward()
        torch.cuda.empty_cache()

    print(out.shape)


if __name__ == "__main__":
    # benchmark_attn()
    test_ape()

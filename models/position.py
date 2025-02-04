"""
WIP
"""

import math

import flash_attn
import flash_attn.layers.rotary as flash_rotary
import torch
import torch.nn.functional as F
from einops import rearrange
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
    def __init__(
        self,
        num_heads: int,
        epsilon_idx: int,
        in_bias: bool = True, # default to all bias
        alpha: float = None,
        autocast_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.epsilon_idx = epsilon_idx
        self.alpha = alpha
        self.autocast_dtype = autocast_dtype

        self.compiled_fn = torch.compile(
            flex_attention, dynamic=False
        )  # fullgraph=True, mode="max-autotune", dynamic=False)
        
        self.forward = self.forward_all if in_bias else self.forward_outer

    def _get_alpha(self, L):
        """
        Average column wise sum of the bias matrix.
        Returned if alpha is not provided.
        """
        return (L * L - 1) / 3

    def forward_all(self, qkv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, H, THREE, L, D = qkv.shape
        assert THREE == 3, "qkv must have 3 dimensions"  # noqa: PLR2004
        assert self.num_heads == H, "num_heads must match H"

        q, k, v = qkv.unbind(dim=-3)

        out_bias_base = indices == self.epsilon_idx  # (B, L)

        if self.alpha is None:
            self.alpha = self._get_alpha(L)

        def alibi_mod(score, b, h, q_idx, kv_idx):
            scale = torch.exp2(-((h + 1) * 8.0 / H))

            in_bias = torch.abs(kv_idx - q_idx)

            # Add bias for epsilon tokens (if any)
            out_bias = (out_bias_base[b, q_idx] + out_bias_base[b, kv_idx]).bool()  # (B, L)

            bias = scale * (in_bias + out_bias * self.alpha)

            return score - bias

        with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
            out = self.compiled_fn(q, k, v, alibi_mod)

        return out
    
    def forward_outer(self, qkv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, H, THREE, L, D = qkv.shape
        assert THREE == 3, "qkv must have 3 dimensions"  # noqa: PLR2004
        assert self.num_heads == H, "num_heads must match H"
        
        q, k, v = qkv.unbind(dim=-3)

        out_bias_base = indices == self.epsilon_idx  # (B, L)

        if self.alpha is None:
            self.alpha = self._get_alpha(L)

        def alibi_mod_no_inner(score, b, h, q_idx, kv_idx):
            scale = torch.exp2(-((h + 1) * 8.0 / H))

            # Add bias for epsilon tokens (if any)
            out_bias = (out_bias_base[b, q_idx] + out_bias_base[b, kv_idx]).bool()  # (B, L)

            bias = scale * out_bias * self.alpha

            return score - bias

        with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
            out = self.compiled_fn(q, k, v, alibi_mod_no_inner)

        return out
        
        

    def _get_alibias_cached(self, B, L, device):
        # cached bc inefficient to compute every time
        if not hasattr(self, "alibias"):
            self.alibias = torch.zeros(B, L, L, device=device)
            for i in range(L):
                for j in range(L):
                    self.alibias[:, i, j] = abs(i - j)
        return self.alibias

    def naive(self, qkv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, H, THREE, L, D = qkv.shape
        assert THREE == 3, "qkv must have 3 dimensions"  # noqa: PLR2004
        assert self.num_heads == H, "num_heads must match H"

        if self.alpha is None:
            self.alpha = self._get_alpha(L)

        in_bias_base = indices == self.epsilon_idx  # (B, L)

        q, k, v = qkv.unbind(dim=-3)  # (B, H, L, D)

        dots = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, L, L)

        in_bias = (in_bias_base.unsqueeze(1) + in_bias_base.unsqueeze(2)).bool()  # (B, L, L)

        out_bias = self._get_alibias_cached(B, L, device=q.device)

        bias = out_bias + in_bias * self.alpha  # (B, L, L)

        bias_scale = torch.exp2(-torch.arange(1, H + 1, device=q.device) * 8.0 / H)  # (H,)

        bias = bias.unsqueeze(1).expand(-1, H, -1, -1) * bias_scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        dots = dots - bias

        attn = F.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, L, D)

        return out, attn

    def naive_reshaped(self, qkv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, L, THREE, H, D = qkv.shape
        assert THREE == 3, "qkv must have 3 dimensions"  # noqa: PLR2004

        qkv = rearrange(qkv, "b l t h d -> b h t l d", h=H)

        # out, _ = self.naive(qkv, indices)
        out = self.forward(qkv, indices)

        out = rearrange(out, "b h l d -> b l (h d)")

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


def test_ape(sweep=False):
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

    if not sweep:
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
    L = 256
    H = 8
    dim = 512

    N = 10_000

    device = torch.device("cuda")
    acast_dtype = torch.bfloat16

    alibi = ALiBiPE(num_heads=H, epsilon_idx=2, alpha=0.1, autocast_dtype=acast_dtype)

    # compile warmup (not benchmarked)

    qkv = torch.randn(B, H, 3, L, dim // H, device=device, requires_grad=True)
    indices = torch.ones(B, L, device=device)
    indices.index_fill_(1, torch.randint(0, L, (int(0.15 * L),), device=device), 2)

    out = alibi(qkv, indices)
    out.sum().backward()

    for _ in tqdm(range(N), desc="ALiBi"):
        qkv = torch.randn(B, H, 3, L, dim // H, device=device, requires_grad=True)

        indices = torch.ones(B, L, device=device)
        indices.index_fill_(1, torch.randint(0, L, (int(0.15 * L),), device=device), 2)

        out = alibi(qkv, indices)

        out.sum().backward()

    for _ in tqdm(range(N), desc="Flash/efficient"):
        qkv = torch.randn(B, H, 3, L, dim // H, device=device, requires_grad=True)

        q, k, v = qkv.unbind(dim=-3)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out.sum().backward()

    for _ in tqdm(range(N), desc="Flash attention"):
        qkv = torch.randn(B, L, 3, H, dim // H, device=device, requires_grad=True, dtype=acast_dtype)

        out = flash_attn.flash_attn_qkvpacked_func(
            qkv,
            0.0,
            causal=False,
        )

        out.sum().backward()


def test_alibi():
    import matplotlib.pyplot as plt

    B = 4
    L = 128
    H = 12
    dim = 64 * H

    device = torch.device("cuda")
    acast_dtype = torch.bfloat16

    alibi = ALiBiPE(num_heads=H, epsilon_idx=2, autocast_dtype=acast_dtype)
    indices = torch.ones(B, L, device=device)

    q = torch.randn(B, H, L, dim // H, device=device) * 2
    k = torch.randn(B, H, L, dim // H, device=device) * 2
    v = torch.randn(B, H, L, dim // H, device=device) * 2

    qkv = torch.stack([q, k, v], dim=2)  # (B, H, 3, L, dim // H)

    # indices.index_fill_(1, torch.randint(0, L, (int(0.30 * L),), device=device), 2)

    out_baseline, attn = alibi.naive(qkv, indices)

    out_flex = alibi.forward(qkv, indices)

    print(torch.allclose(out_baseline, out_flex))

    print(torch.norm(out_baseline - out_flex))

    attn = torch.clip(attn, 0, 1e-5)

    plt.imshow(attn[0, 0].cpu().detach().numpy(), aspect="auto", cmap="jet")
    plt.colorbar()
    plt.title("DD AliBi (rescaled)")
    plt.savefig("attn.png")
    plt.close()

    fig, axs = plt.subplots(H // 4, 4, figsize=(15, 15))
    axs = axs.flatten()
    h_dim = dim // H

    out = rearrange(out_flex, "b h l d -> b l (h d)")
    out = out[0].cpu().detach().numpy()

    for i in range(H):
        im = axs[i].imshow(out[:, i * h_dim : (i + 1) * h_dim], aspect="auto")
        axs[i].set_title(f"Head {i}")
        plt.colorbar(im, ax=axs[i])

    plt.savefig("alibi_output_subplots.png")

    plt.close()


if __name__ == "__main__":
    # benchmark_attn()
    # test_ape()
    test_alibi()

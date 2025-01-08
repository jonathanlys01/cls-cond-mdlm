"""
WIP
"""

import math

import torch


# https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742
def get_slopes(n):
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
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


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
        if dim != x.shape[-1]:  # recompute bias if dimensionality changes
            self.cached_bias = self._dist(x)
        bias = self.cached_bias
        return x + bias

    def with_slopes(self, slopes):
        self.cached_bias = self.cached_bias * slopes
        return self.cached_bias


# Example usage
dim = 10
model = ALiBi(dim)
input_tensor = torch.zeros((dim, dim)).int()
output = model(input_tensor)

print("Output with bias added:")
print(output)

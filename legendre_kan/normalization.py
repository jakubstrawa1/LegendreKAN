import torch, torch.nn as nn
import math

class GaussianCDFNorm(nn.Module):
    def __init__(self, unbiased: bool = True, affine: bool = False,
                 eps: float = 1e-5):
        super().__init__()
        self.unbiased = unbiased
        self.eps = eps

        #affine means stat-only, no learnable params for the sake of research
        if affine:
            self.weight = nn.Parameter(torch.ones(1))  # broadcast over features
            self.bias   = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias",   None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu  = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=self.unbiased, keepdim=True)
        z   = (x - mu) / (var + self.eps).sqrt()

        u = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

        if self.weight is not None:
            u = u * self.weight + self.bias
        return u


class EmpiricalCDFNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sorted_x, _ = torch.sort(x, dim=0)
        idx = torch.searchsorted(sorted_x, x)
        N   = x.shape[0]
        return (idx.to(x.dtype) + 0.5) / N
import torch, torch.nn as nn
import math
from typing import Literal
from config import CDFMethod, Reduction

class CDFNorm(nn.Module):
    def __init__(
        self,
        method:    CDFMethod | str = CDFMethod.GAUSSIAN,
        reduction: Reduction | str = Reduction.BATCH,
        *,
        unbiased: bool = True,
        eps: float = 1e-5,
        affine: bool = False,
    ):
        super().__init__()

        self.method    = str(method)
        self.reduction = str(reduction)
        self.unbiased  = unbiased
        self.eps       = eps

        if affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias   = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias",   None)

    def _gaussian_cdf(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        mu  = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, unbiased=self.unbiased, keepdim=True)
        z   = (x - mu) / (var + self.eps).sqrt()
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    def _empirical_cdf(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        sorted_x, _ = torch.sort(x, dim=dim)
        idx = torch.searchsorted(sorted_x, x, right=False, dim=dim)
        N   = x.size(dim)
        return (idx.to(x.dtype) + 0.5) / N


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stat_dim = 0 if self.reduction == "batch" else 1

        if self.method == "gaussian":
            u = self._gaussian_cdf(x, stat_dim)
        else:
            u = self._empirical_cdf(x, stat_dim)

        if self.weight is not None:
            u = u * self.weight + self.bias
        return u
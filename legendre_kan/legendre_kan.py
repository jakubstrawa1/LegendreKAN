from typing import Literal
import torch, torch.nn as nn
from normalization import GaussianCDFNorm, EmpiricalCDFNorm
from layer import LegendreKANLayer

class LegendreKAN(nn.Module):
    """
    Legendre Kernel Adaptive Network (KAN)

    Args
    ----
    norm : {"gaussian", "empirical"}
        Inter-layer normalisation strategy.
    degree : int
        Degree of the Legendre polynomial basis.
    shifted : bool
        Whether to use shifted Legendre polynomials (domain [0,1] instead of [-1,1]).
    """
    def __init__(
        self,
        norm: Literal["gaussian", "empirical"] = "gaussian",
        degree: int = 4,
        shifted: bool = False
    ):
        super().__init__()
        self.shifted = shifted

        def _make_norm():
            if norm == "gaussian":
                return GaussianCDFNorm(unbiased=False)
            elif norm == "empirical":
                return EmpiricalCDFNorm()
            else:
                raise ValueError(f"norm must be 'gaussian' or 'empirical', got {norm!r}")

        self.norm0 = _make_norm()
        self.kan1  = LegendreKANLayer(28 * 28, 32, degree, shifted=self.shifted)
        self.norm1 = _make_norm()

        self.kan2  = LegendreKANLayer(32, 16, degree, shifted=self.shifted)
        self.norm2 = _make_norm()

        self.kan3  = LegendreKANLayer(16, 10, degree, shifted=self.shifted)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm0(x)
        x = self.kan1(x)
        x = self.norm1(x)
        x = self.kan2(x)
        x = self.norm2(x)
        x = self.kan3(x)
        return x

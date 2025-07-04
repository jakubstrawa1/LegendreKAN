# kan_model.py
from __future__ import annotations

import torch
import torch.nn as nn

from config        import  CDFMethod, Reduction, NormConfig
from layer         import LegendreKANLayer

class LegendreKAN(nn.Module):
    """
    Legendre Kernel Adaptive Network (KAN)

    Parameters
    ----------
    norm_method : CDFMethod | str
        "gaussian"  or  "empirical"
    norm_reduction : Reduction | str
        "batch"     or  "sample"
    degree : int
        Degree of the Legendre polynomial basis in each KAN layer.
    shifted : bool
        If True, *shifted* Legendre polynomials (domain [0, 1]).
        If False, standard Legendre polynomials on [-1, 1].
    affine : bool
        Give every inter-layer normaliser learnable scale/shift.
    """

    def __init__(
        self,
        *,
        norm_method: CDFMethod | str = CDFMethod.GAUSSIAN,
        norm_reduction: Reduction | str = Reduction.SAMPLE,
        degree: int = 4,
        shifted: bool = False,
        affine: bool = False,
    ):
        super().__init__()
        self.shifted = shifted

        def _make_norm() -> nn.Module:
            cfg = NormConfig(
                method    = norm_method,
                reduction = norm_reduction,
                unbiased  = False,
                affine    = affine,
            )
            return cfg.build()

        self.normalizer = _make_norm()

        self.kan1  = LegendreKANLayer(
            28 * 28, 32, degree,
            rescale_to_11 = not shifted,
            norm = self.normalizer,
        )
        self.kan2  = LegendreKANLayer(
            32, 16, degree,
            rescale_to_11 = not shifted,
            norm = self.normalizer,
        )

        self.kan3  = LegendreKANLayer(
            16, 10, degree,
            rescale_to_11 = not shifted,
            norm = self.normalizer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        return x

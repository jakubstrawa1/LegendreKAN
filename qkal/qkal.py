"""
Implementation of a Quantile-Normalised KAN (QKAL)
using a Legendre–polynomial basis expansion and learnable β, γ
parameters in LayerNorm.  The default topology is sized for 28 × 28
greyscale images (e.g. MNIST) and 10 output classes.

Author: Jakub Strawa, Jarek Duda - July 2025
"""

import torch
import torch.nn as nn
from qkal.layer import QKAL_Layer


class QKAL(nn.Module):
    """
    Quantile-Normalised KAN with Legendre basis and LayerNorm only
    (no SiLU/non-linearities aside from the polynomial lift).
    """

    def __init__(
        self,
        degree: int,
        elementwise_affine: bool = True,
        hidden_dim: int = 256,
        in_dim: int = 28 * 28,
        n_classes: int = 10,
    ) -> None:
        super().__init__()

        # ── Flatten-input LayerNorm ────────────────────────────────────
        self.ln0 = nn.LayerNorm(in_dim, elementwise_affine=elementwise_affine)

        # ── QKAL blocks ────────────────────────────────────────────────
        self.l1 = QKAL_Layer(in_dim, hidden_dim, degree)
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine)

        self.l2 = QKAL_Layer(hidden_dim, hidden_dim, degree)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine)

        # ── Output block ───────────────────────────────────────────────
        self.l3 = QKAL_Layer(hidden_dim, n_classes, degree)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, 1, H, W) or already-flattened (B, in_dim)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.ln0(x)
        x = self.l1(x)
        x = self.ln1(x)
        x = self.l2(x)
        x = self.ln2(x)
        x = self.l3(x)
        return x


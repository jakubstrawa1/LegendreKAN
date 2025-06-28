import torch, torch.nn as nn
import math
from legendre import _make_legendre_polynomials

class LegendreKANLayer(nn.Module):
    MAX_DEGREE = 4

    def __init__(self, input_dim: int, output_dim: int, degree: int = 4, shifted=False):
        super().__init__()

        if not (0 <= degree <= self.MAX_DEGREE):
            raise ValueError(f"`degree` must be in [0, {self.MAX_DEGREE}]")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        leg_table = _make_legendre_polynomials(self.MAX_DEGREE, dtype=torch.get_default_dtype())
        self.register_buffer("legendre_table", leg_table, persistent=False)  # 5×5

        self.legendre_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(
            self.legendre_coeffs,
            mean=0.0,
            std=1 / (input_dim * (degree + 1)),
        )

    def _legendre_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute [P0(x) … P_degree(x)] for every feature value in `x`.

        x : (batch, input_dim) already scaled to [0, 1]
        returns : (batch, input_dim, degree+1)
        """
        powers = x.unsqueeze(-1) ** torch.arange(0, self.MAX_DEGREE + 1,
                                                 device=x.device,
                                                 dtype=x.dtype)            # (B, I, 5)

        basis_full = torch.matmul(powers, self.legendre_table.t())          # (B, I, 5)
        return basis_full[..., : self.degree + 1]                           # (B, I, D+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, input_dim) already scaled to [0, 1]
        returns : (batch, output_dim)
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension = {self.input_dim}, got {x.shape[-1]}"
            )

        basis = self._legendre_basis(x)
        return torch.einsum("bid,iod->bo", basis, self.legendre_coeffs)

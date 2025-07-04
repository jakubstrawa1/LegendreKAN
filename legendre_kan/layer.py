import torch
import torch.nn as nn
import math
from typing import Optional, Literal
from config import NormConfig

class LegendreKANLayer(nn.Module):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        degree:     int,
        *,
        rescale_to_11: bool = True,
        norm:     Optional[nn.Module]  = None,     # ready-made
        norm_cfg: Optional[NormConfig] = None,     # or build from cfg
    ):
        super().__init__()

        self.inputdim = input_dim
        self.outdim   = output_dim
        self.degree   = degree
        self.rescale_to_11 = rescale_to_11

        if norm is not None:
            self.norm = norm
        else:
            cfg = norm_cfg or NormConfig()
            self.norm = cfg.build()

        self.legendre_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(self.legendre_coeffs,
                        mean=0.0,
                        std=1 / (input_dim * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #flatten the input so each row is the input
        B = x.reshape(-1, self.inputdim)

        #normalize to [0,1]
        x_uniform = self.norm(B)

        #rescale if needed
        x_leg = 2 * x_uniform - 1 if self.rescale_to_11 else x_uniform

        leg_list = [torch.ones_like(x_leg)]
        if self.degree >= 1:
            leg_list.append(x_leg)

        for n in range(1, self.degree):
            k  = n + 1
            Pk = ((2*n + 1)*x_leg*leg_list[-1] - n*leg_list[-2]) / k
            leg_list.append(Pk)

        legendre = torch.stack(leg_list, dim=2)

        y = torch.einsum('bid,iod->bo', legendre, self.legendre_coeffs)

        return y.view(-1, self.outdim)

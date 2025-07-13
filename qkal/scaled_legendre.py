import torch

from functools import lru_cache

@lru_cache(maxsize=128)
def compute_scaled_legendre_polynomials(x: torch.Tensor, order: int) -> torch.Tensor:
    """
    Function for generating scaled Legendre polynomials. requiring [0,1] input.
    """
    t = 2.0 * x - 1.0
    P0 = torch.ones_like(t)
    if order == 0:
        return P0[..., None]
    P1 = t
    P = [P0, P1]
    for n in range(1, order):
        P_next = ((2.0 * n + 1.0) * t * P[-1] - n * P[-2]) / (n + 1.0)
        P.append(P_next)
    P = torch.stack(P, dim=-1)
    degrees = torch.arange(order + 1, device=x.device, dtype=x.dtype)
    scale = torch.sqrt(2.0 * degrees + 1.0)
    return P * scale
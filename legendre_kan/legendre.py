import torch, torch.nn as nn
import math


SQRT3, SQRT5, SQRT7 = map(math.sqrt, (3.0, 5.0, 7.0))

#Main table for
table = torch.tensor(
    [
        [1.0,               0.0,               0.0,               0.0,               0.0],
        [-SQRT3,            2 * SQRT3,         0.0,               0.0,               0.0],
        [SQRT5,            -6 * SQRT5,         6 * SQRT5,         0.0,               0.0],
        [-SQRT7,           12 * SQRT7,        -30 * SQRT7,        20 * SQRT7,        0.0],
        [3.0,             -60.0,             270.0,             -420.0,             210.0],
    ],
    dtype=torch.float32,
)

def _make_legendre_polynomials(max_degree: int = 4, *, dtype=torch.float32):
    return table[: max_degree + 1, : max_degree + 1]

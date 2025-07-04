from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from normalization import CDFNorm


class CDFMethod(str, Enum):
    GAUSSIAN  = "gaussian"
    EMPIRICAL = "empirical"


class Reduction(str, Enum):
    BATCH  = "batch"
    SAMPLE = "sample"


@dataclass
class NormConfig:
    method:    CDFMethod  = CDFMethod.GAUSSIAN
    reduction: Reduction  = Reduction.SAMPLE
    unbiased:  bool       = False
    eps:       float      = 1e-5
    affine:    bool       = False

    def build(self) -> nn.Module:
        return CDFNorm(
            method    = self.method,
            reduction = self.reduction,
            unbiased  = self.unbiased,
            eps       = self.eps,
            affine    = self.affine,
        )

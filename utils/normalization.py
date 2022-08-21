from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.linalg import norm


class Normalization:
    def __init__(
        self, apply_norm: int, custom_min_max: Optional[Tuple[int, int]] = None
    ):
        self.apply_norm = apply_norm
        self.custom_min_max = custom_min_max
        return

    def apply_normalization(
        self, tensor: Tensor, eps: float, p: float, dim: int
    ) -> Tensor:
        if self.apply_norm == -1:
            return tensor
        elif self.apply_norm == 0:
            return self.min_max_normalization(tensor, eps, dim)
        elif self.apply_norm == 1:
            return self.standardization(tensor, eps, dim)
        elif self.apply_norm == 2:
            return self.p_normalization(tensor, p, dim)
        else:
            raise ValueError("Invalid value of normalization received!")

    def min_max_normalization(self, tensor: Tensor, eps: float, dim: int) -> Tensor:
        if self.custom_min_max is not None:
            min_ = self.custom_min_max[0]
            max_ = self.custom_min_max[1]
        else:
            min_ = torch.min(tensor, dim=dim)
            max_ = torch.max(tensor, dim=dim)

        tensor = 2.0 * ((tensor - min_) / (max_ - min_ + eps)) - 1
        return tensor

    @staticmethod
    def standardization(tensor: Tensor, eps: float, dim: int) -> Tensor:
        mean_ = torch.mean(tensor, dim=dim)
        std_ = torch.std(tensor, dim=dim)

        tensor = (tensor - mean_) / (std_ + eps)
        return tensor

    @staticmethod
    def p_normalization(tensor: Tensor, p: float, dim: int) -> Tensor:
        tensor = norm(tensor, dim=dim, ord=p)
        return tensor

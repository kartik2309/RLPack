from typing import Optional, Tuple

from rlpack import pytorch


class Normalization:
    def __init__(
        self, apply_norm: int, custom_min_max: Optional[Tuple[int, int]] = None
    ):
        self.apply_norm = apply_norm
        self.custom_min_max = custom_min_max
        return

    def apply_normalization(
        self, tensor: pytorch.Tensor, eps: float, p: float, dim: int
    ) -> pytorch.Tensor:
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

    def min_max_normalization(
        self, tensor: pytorch.Tensor, eps: float, dim: int
    ) -> pytorch.Tensor:
        if self.custom_min_max is not None:
            min_ = self.custom_min_max[0]
            max_ = self.custom_min_max[1]
        else:
            min_ = pytorch.min(tensor, dim=dim)
            max_ = pytorch.max(tensor, dim=dim)

        tensor = 2.0 * ((tensor - min_) / (max_ - min_ + eps)) - 1
        return tensor

    @staticmethod
    def standardization(tensor: pytorch.Tensor, eps: float, dim: int) -> pytorch.Tensor:
        mean_ = pytorch.mean(tensor, dim=dim)
        std_ = pytorch.std(tensor, dim=dim)

        tensor = (tensor - mean_) / (std_ + eps)
        return tensor

    @staticmethod
    def p_normalization(tensor: pytorch.Tensor, p: float, dim: int) -> pytorch.Tensor:
        tensor = pytorch.linalg.norm(tensor, dim=dim, ord=p)
        return tensor

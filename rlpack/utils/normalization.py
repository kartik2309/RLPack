from typing import Optional, Tuple

from rlpack import pytorch


class Normalization:
    """
    Normalization class providing methods for normalization techniques.
    """

    def __init__(
        self, apply_norm: int, custom_min_max: Optional[Tuple[int, int]] = None
    ):
        """
        :param apply_norm: int: apply_norm code for normalization. (Refer rlpack/utils/setup.py
            for more information).
        :param custom_min_max: Optional[Tuple[int, int]]: Tuple of custom min and max value
            for min-max normalization. Default: None
        """
        self.apply_norm = apply_norm
        self.custom_min_max = custom_min_max
        return

    def apply_normalization(
        self, tensor: pytorch.Tensor, eps: float = 5e-12, p: int = 2, dim: int = 0
    ) -> pytorch.Tensor:
        """
        All encompassing function to perform normalization depending on the instance's apply_norm code.

        :param tensor: pytorch.Tensor: The tensor to apply normalization on.
        :param eps: float: The epsilon value for normalization (small value for numerical stability). Default: 5e-12
        :param p: int: The p-value for p-normalization. Default: 2
        :param dim: int: The dimension along which normalization is to be applied. Default: 0
        :return: pytorch.Tensor: The normalized tensor.
        """
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
        """
        Method to apply min-max normalization.

        :param tensor: pytorch.Tensor: The input tensor to be min-max normalized.
        :param eps: float: The epsilon value for normalization (small value for numerical stability).
        :param dim: int: The dimension along which normalization is to be applied.
        :return (pytorch.Tensor): The normalized tensor.
        """
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
        """
        Method to standardize the input tensor.
        :param tensor: pytorch.Tensor: he input tensor to be standardized.
        :param eps: float: The epsilon value for normalization (small value for numerical stability).
        :param dim: int: The dimension along which normalization is to be applied.
        :return: pytorch.Tensor: The standardized tensor.
        """
        mean_ = pytorch.mean(tensor, dim=dim)
        std_ = pytorch.std(tensor, dim=dim)
        tensor = (tensor - mean_) / (std_ + eps)
        return tensor

    @staticmethod
    def p_normalization(tensor: pytorch.Tensor, p: int, dim: int) -> pytorch.Tensor:
        """
        The p-normalization method.
        :param tensor: pytorch.Tensor: he input tensor to be standardized.:
        :param p: int: The p-value for p-normalization.
        :param dim: int: The dimension along which normalization is to be applied.
        :return: pytorch.Tensor: The p-normalized tensor.
        """
        tensor = pytorch.linalg.norm(tensor, dim=dim, ord=p)
        return tensor

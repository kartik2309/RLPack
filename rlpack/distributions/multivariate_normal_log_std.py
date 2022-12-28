"""!
@package rlpack.distributions
@brief This package implements the additional or modified distributions for RLPack.


Currently following classes have been implemented:
    - `MultivariateNormalLogStd`: This is the modified version of MultiVariateNormal provided by PyTorch, wherein
        diagonal of input matrix for scale is raised to the power `e` (Napier's constant). Implemented as
        rlpack.distributions.multivariate_normal_log_std.MultivariateNormalLogStd.
    - `NormalLogStd`: This is the modified version of Normal provided by PyTorch, wherein input scale is raised to
        the power `e` (Napier's constant). Implemented as rlpack.distributions.normal_log_std.NormalLogStd.
"""


from abc import ABC
from typing import Any

from rlpack import pytorch, pytorch_distributions


class MultivariateNormalLogStd(pytorch_distributions.MultivariateNormal, ABC):
    """
    This class implements a variant of MultivariateNormal distribution where input scale matrix is assumed to
    be of form log(scale); hence is raise to the power `e` (Napier's constant).
    """

    def __init__(
        self,
        loc: pytorch.Tensor,
        covariance_matrix: pytorch.Tensor = None,
        precision_matrix: pytorch.Tensor = None,
        scale_tril: pytorch.Tensor = None,
        validate_args: Any = None,
    ):
        """
        Initialization method for NormalLogStd. This method raise scale to the power and initializes the super class.
        Either one of covariance_matrix or precision_matrix or scale_tril must be passed. If all passed `scale_tril`
        is considered. Power is raised on diagonal elements of the matrix and diagonal embedding is used to create
        the new matrix.
        @param loc: pytorch.Tensor: The `loc` parameter for MultivariateNormal distribution; the mean. Also called `mu`.
        @param covariance_matrix: pytorch.Tensor: The positive-definite covariance matrix.
        @param precision_matrix: pytorch.Tensor: The positive-definite precision matrix.
        @param scale_tril: pytorch.Tensor: The lower-triangular factor of covariance, with positive-valued diagonal.
        @param validate_args: Any: Internal argument for PyTorch. Default: None

        """
        if scale_tril is not None:
            scale_tril = self._raise_to_exp(scale_tril)
        elif precision_matrix is not None:
            precision_matrix = self._raise_to_exp(precision_matrix)
        elif covariance_matrix is not None:
            covariance_matrix = self._raise_to_exp(covariance_matrix)
        else:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril must be specified."
            )

        super(MultivariateNormalLogStd, self).__init__(
            loc, covariance_matrix, precision_matrix, scale_tril, validate_args
        )

    @staticmethod
    def _raise_to_exp(scale_tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        Function to raise power on given scale tensor. Power is raised on diagonal elements of the matrix and
        diagonal embedding is used to create the new matrix.
        @param scale_tensor: The input scale tensor on which power is to be raised.
        @return: pytorch.Tensor: The power raised tensor.
        """
        scale_tensor = pytorch.exp(scale_tensor) * pytorch.eye(
            *scale_tensor.size(),
            device=scale_tensor.device,
            dtype=scale_tensor.dtype,
            requires_grad=False,
        )
        return scale_tensor

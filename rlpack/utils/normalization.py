"""!
@package rlpack.utils
@brief This package implements the basic utilities to be used across rlpack.


Currently following classes have been implemented:
    - `Normalization`: Normalization tool implemented as rlpack.utils.normalization.Normalization with
        support for regular normalization methods.
    - `SanityCheck`: Sanity check for arguments when using Simulator from rlpack.simulator.Simulator. Class is
        implemented as rlpack.utils.sanity_check.SanityCheck.
    - `Setup`: Sets up the simulator to run the agent with environment. Implemented as rlpack.utils.setup.Setup.
    - `InternalCodeSetup`: For internal use to check/validate arguments and to retrieve codes for internal use.
        Implemented as rlpack.utils.internal_code_setup.InternalCodeSetup.

Following packages are part of utils:
    - `base`: A package for base class, implemented as utils.base

Following TypeVars have been defined:
    - `LRScheduler`: The Typing variable for LR Schedulers.
    - `LossFunction`: The Typing variable for Loss Functions.
    - `Activation`: The Typing variable for Activations.
"""


from typing import Optional, Tuple

from rlpack import pytorch


class Normalization:
    """
    Normalization class providing methods for normalization techniques.
    """

    def __init__(
        self,
        apply_norm: int,
        custom_min_max: Optional[Tuple[int, int]] = None,
        eps: float = 5e-12,
        p: int = 2,
        dim: int = 0,
    ):
        """
        Initialize Normalization class.
        @param apply_norm: int: apply_norm code for normalization. (Refer rlpack.utils.setup.Setup
            for more information).
        @param custom_min_max: Optional[Tuple[int, int]]: Tuple of custom min and max value
            for min-max normalization. Default: None.
        @param eps: float: The epsilon value for normalization (small value for numerical stability). Default: 5e-12.
        @param p: int: The p-value for p-normalization. Default: 2.
        @param dim: int: The dimension along which normalization is to be applied. Default: 0.
        """
        ## The input `apply_norm` argument; indicating the normalisation to be used. @I{# noqa: E266}
        self.apply_norm = apply_norm
        ## The input `custom_min_max` argument.  @I{# noqa: E266}
        ## Indicating the custom min-max values for min-max normalisation to be used. @I{# noqa: E266}
        self.custom_min_max = custom_min_max
        ## The input `eps` argument; indicating epsilon to be used for normalisation. @I{# noqa: E266}
        self.eps = eps
        ## The input `p` argument; indicating p-value for p-normalisation. @I{# noqa: E266}
        self.p = p
        ## The input `dim` argument; indicating dimension along which we wish to normalise. @I{# noqa: E266}
        self.dim = dim
        return

    def apply_normalization(self, tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        All encompassing function to perform normalization depending on the instance's apply_norm code.
        @param tensor: pytorch.Tensor: The tensor to apply normalization on.
        @return pytorch.Tensor: The normalized tensor.
        """
        if self.apply_norm == -1:
            return tensor
        elif self.apply_norm == 0:
            return self.min_max_normalization(tensor)
        elif self.apply_norm == 1:
            return self.standardization(tensor)
        elif self.apply_norm == 2:
            return self.p_normalization(tensor)
        else:
            raise ValueError("Invalid value of normalization received!")

    def min_max_normalization(self, tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        Method to apply min-max normalization.
        @param tensor: pytorch.Tensor: The input tensor to be min-max normalized.
        @return (pytorch.Tensor): The normalized tensor.
        """
        if self.custom_min_max is not None:
            min_ = self.custom_min_max[0]
            max_ = self.custom_min_max[1]
        else:
            min_ = pytorch.min(tensor, dim=self.dim)
            max_ = pytorch.max(tensor, dim=self.dim)
        tensor = (tensor - min_) / (max_ - min_ + self.eps)
        return tensor

    def standardization(self, tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        Method to standardize the input tensor.
        @param tensor: pytorch.Tensor: he input tensor to be standardized.
        @return pytorch.Tensor: The standardized tensor.
        """
        mean_ = pytorch.mean(tensor, dim=self.dim)
        std_ = pytorch.std(tensor, dim=self.dim)
        tensor = (tensor - mean_) / (std_ + self.eps)
        return tensor

    def p_normalization(self, tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        The p-normalization method
        @param tensor: pytorch.Tensor: The input tensor to be standardized.
        @return pytorch.Tensor: The p-normalized tensor.
        """
        tensor = pytorch.linalg.norm(tensor, dim=self.dim, ord=self.p)
        return tensor

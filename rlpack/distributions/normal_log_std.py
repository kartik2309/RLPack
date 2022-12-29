"""!
@package rlpack.distributions
@brief This package implements the additional or modified distributions for RLPack.


Currently following classes have been implemented:
    - `GaussianMixture`: This class implements the gaussian mixture model with optional weights for mixture. Implemented
        as rlpack.distributions.gaussian_mixture.GaussianMixture
    - `GaussianMixtureLogStd`: This class implements the gaussian mixture model with optional weights for mixture
        where input scale is raised to the power `e` (Napier's constant). Implemented as
        rlpack.distributions.gaussian_mixture_log_std.GaussianMixtureLogStd
    - `MultivariateNormalLogStd`: This is the modified version of MultiVariateNormal provided by PyTorch, wherein
        diagonal of input matrix for scale is raised to the power `e` (Napier's constant). Implemented as
        rlpack.distributions.multivariate_normal_log_std.MultivariateNormalLogStd.
    - `NormalLogStd`: This is the modified version of Normal provided by PyTorch, wherein input scale is raised to
        the power `e` (Napier's constant). Implemented as rlpack.distributions.normal_log_std.NormalLogStd.
"""


from abc import ABC
from typing import Any, Union

from rlpack import pytorch, pytorch_distributions


class NormalLogStd(pytorch_distributions.Normal, ABC):
    """
    This class implements a variant of Normal distribution where input standard deviation (scale) is assumed to
    be of form log(scale); hence is raise to the power `e` (Napier's constant). This class inherits from PyTorch's
    Normal class.
    """

    def __init__(
        self,
        loc: Union[float, pytorch.Tensor],
        scale: Union[float, pytorch.Tensor],
        validate_args: Any = None,
    ):
        """
        Initialization method for NormalLogStd. This method raise scale to the power and initializes the super class.
        @param loc: Union[float, pytorch.Tensor]: The `loc` parameter for Normal distribution; the mean. Also called
            `mu`.
        @param scale: Union[float, pytorch.Tensor]: The `scale` parameter for Normal distribution;
            the standard deviation. Also called `sigma`.
        @param validate_args: Any: Internal argument for PyTorch. Default: None
        """
        scale = pytorch.exp(scale)
        super(NormalLogStd, self).__init__(loc, scale, validate_args)

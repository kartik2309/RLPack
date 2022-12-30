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
from typing import Any

from rlpack import pytorch, pytorch_distributions


class GaussianMixtureLogStd(pytorch_distributions.MixtureSameFamily, ABC):
    """
    This class implements the Gaussian Mixture distribution where input standard deviation (scale) is assumed to
    be of form log(scale); hence is raise to the power `e` (Napier's constant). This class inherits from PyTorch's
    Normal MixtureSameFamily.
    """

    def __init__(
        self,
        loc: pytorch.Tensor,
        scale: pytorch.Tensor,
        mixture_distribution: pytorch_distributions.Categorical = None,
        validate_args: Any = None,
    ):
        """
        Initialization method for GaussianMixture. This will create the gaussian mixture object for the given
            parameters
        @param loc: pytorch.Tensor: The `loc` parameter for Normal distribution; the mean. Also called `mu`.
        @param scale: pytorch.Tensor: The `scale` parameter for Normal distribution;
            the standard deviation. Also called `sigma`.
        @param mixture_distribution: pytorch_distributions.Categorical: The mixture distribution to be used for
            weighing gaussian mixture model. By default, will create a distribution with same weights. Default: None.
        @param validate_args: Any: Internal argument for PyTorch. Default: None
        """
        if mixture_distribution is None:
            mixture_distribution = pytorch_distributions.Categorical(
                pytorch.ones(loc.size(), device=loc.device, dtype=loc.dtype)
            )
        scale = pytorch.exp(scale)
        mixture_component = pytorch_distributions.Normal(
            loc, scale, validate_args=validate_args
        )
        super(GaussianMixtureLogStd, self).__init__(
            mixture_distribution, mixture_component, validate_args
        )

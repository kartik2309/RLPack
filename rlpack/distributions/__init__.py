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


from rlpack.distributions.gaussian_mixture import GaussianMixture
from rlpack.distributions.gaussian_mixture_log_std import GaussianMixtureLogStd
from rlpack.distributions.multivariate_normal_log_std import MultivariateNormalLogStd
from rlpack.distributions.normal_log_std import NormalLogStd

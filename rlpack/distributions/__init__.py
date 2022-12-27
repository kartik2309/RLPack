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


from rlpack.distributions.multivariate_normal_log_std import MultivariateNormalLogStd
from rlpack.distributions.normal_log_std import NormalLogStd

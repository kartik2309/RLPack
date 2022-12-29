"""!
@package rlpack.exploration.utils
@brief This package utilities for rlpack.exploration package.

Currently following classes have been implemented:
    - `Exploration`: Abstract base class for exploration tools. Implemented as
        rlpack.exploration.utils.exploration.Exploration

Following typing hints have been defined:
    - `GaussianAnnealFuncSignature`: Typing hint signature for anneal function (`anneal_func`) argument in
        GaussianNoise class. Implemented as rlpack.exploration.utils.typing_hints.GaussianAnnealFuncSignature
"""


from typing import List, Protocol

from rlpack import pytorch


class GaussianAnnealFuncSignature(Protocol):
    """
    Signature for annealing function for loc, scale and weights in Gaussian Noise Exploration.
    """

    def __call__(
        self,
        loc: pytorch.Tensor,
        scale: pytorch.Tensor,
        weight: pytorch.Tensor,
        *args,
        **kwargs
    ) -> List[pytorch.Tensor]:
        """
        __call__ method indicating signature of annealing function.
        @param loc: pytorch.Tensor: The `loc` parameter for Normal distribution; the mean. Also called `mu`.
        @param scale: pytorch.Tensor: The `scale` parameter for Normal distribution;
            the standard deviation. Also called `sigma`.
        @param weight: pytorch.Tensor: The factor multiply on sampled noise. By default, will create a tensor of
            ones. Default: None.
        @param *args: Any positional arguments you may want to pass.
        @param **kwargs: Any keyword arguments you may want to pass.
        @return: List[pytorch.Tensor]: The list of annealed loc, scale, weights
        """
        pass

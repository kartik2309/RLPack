"""!
@package rlpack.exploration
@brief This package implements the exploration tools for RLPack to explore the environment


Currently following classes have been implemented:
    - `GaussianNoiseExploration`: This class implements the gaussian noise exploration tool with optional weights for
        samples and annealing of distribution parameters. Implemented as
        rlpack.exploration.gaussian_noise_exploration.GaussianNoiseExploration


Following packages are part of exploration:
    - `utils`: A package utilities for exploration package.
"""


from typing import Union

from rlpack import pytorch, pytorch_distributions
from rlpack.exploration.utils.exploration import Exploration
from rlpack.exploration.utils.typing_hints import GaussianAnnealFuncSignature


class GaussianNoiseExploration(Exploration):
    """
    Exploration tool to produce gaussian noise.
    """

    def __init__(
        self,
        loc: pytorch.Tensor,
        scale: pytorch.Tensor,
        anneal_func: Union[GaussianAnnealFuncSignature, None] = None,
        weight: Union[pytorch.Tensor, None] = None,
    ):
        """
        Initialization Method for GaussianNoiseExploration
        @param loc: pytorch.Tensor: The `loc` parameter for Normal distribution; the mean. Also called `mu`.
        @param scale: pytorch.Tensor: The `scale` parameter for Normal distribution;
            the standard deviation. Also called `sigma`.
        @param anneal_func: Union[GaussianAnnealFuncSignature, None]: Annealing function for loc, scale and weights.
            Refer to signature in rlpack.exploration.utils.typing_hints.GaussianAnnealFuncSignature. This is an
            optional parameter and if not passed, no annealing takes place. Default: None
        @param weight: The factor multiply on sampled noise. By default, will create a tensor of ones. Default: None
        """
        self.loc = loc
        self.scale = scale
        self.anneal_func = anneal_func
        if weight is None:
            self.weight = pytorch.ones(loc.size(), device=loc.device, dtype=loc.dtype)
        else:
            self.weight = weight
        self.normal_distribution = pytorch_distributions.Normal(loc, scale)
        super(GaussianNoiseExploration, self).__init__()

    @pytorch.no_grad()
    def sample(self, *args, **kwargs) -> pytorch.Tensor:
        """
        Samples a weighted gaussian noise with given parameters. This is run with No Grad Guard and operations are
        not attached to computational graph
        @param args: Arbitrary positional arguments.
        @param kwargs: Arbitrary keyword arguments.
        @return: pytorch.Tensor: The sampled weighed gaussian noise tensor.
        """
        return self.weight * self.normal_distribution.sample()

    def rsample(self, *args, **kwargs) -> pytorch.Tensor:
        """
        The method has been overriden here to raise a `NotImplementedError`, since this exploration tool does not
        have any learnable parameters and hence `sample` method must be used.
        @param args: Arbitrary positional arguments.
        @param kwargs: Arbitrary keyword arguments.
        @return: pytorch.Tensor: The sampled weighed gaussian noise tensor. Note that this just shows the
            signature of the method and is not valid if called.
        """
        raise NotImplementedError(
            "`rsample` method has not been implemented for GaussianNoiseExploration! "
            "It should be called in a Model!"
        )

    @pytorch.no_grad()
    def reset(self):
        """
        Resets the exploration tool by calling anneal function (argument `anneal_func`). If no anneal function was
        passed, does nothing.
        """
        if self.anneal_func is not None:
            self.loc, self.scale, self.weight = self.anneal_func(
                self.loc,
                self.scale,
                self.weight,
            )
            self.normal_distribution = pytorch_distributions.Normal(
                self.loc, self.scale
            )

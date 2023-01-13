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


from abc import ABC, abstractmethod

from rlpack import pytorch


class Exploration(pytorch.nn.Module, ABC):
    """
    Exploration class is the base class for exploration tools in rlpack. It inherits from pytorch.nn.Module and
    ABC. pytorch.nn.Module is only relevant when exploration tool has learnable parameters.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Abstract initialization method for Exploration.
        @param args: Arbitrary positional arguments.
        @param kwargs: Arbitrary keyword arguments.
        """
        super(Exploration, self).__init__()
        pass

    @abstractmethod
    def sample(self, *args, **kwargs) -> pytorch.Tensor:
        """
        The abstract sampling method for Exploration. This method is to be used when there are no learnable
        parameters in the exploration tool. This is typically used inside the agents.Any subsequent
        implementation must raise error if exploration tool has learnable parameters.
        @param args: Arbitrary positional arguments.
        @param kwargs: Arbitrary keyword arguments.
        @return: pytorch.Tensor: The sampled noise tensor
        """
        pass

    @abstractmethod
    def rsample(self, *args, **kwargs) -> pytorch.Tensor:
        """
        The abstract re-parametrized sampling method for Exploration. This method is to be used when there
        are learnable parameters in exploration tool. This is typically used inside models. Any subsequent
        implementation must raise error if exploration tool has no learnable parameters.
        @param args: Arbitrary positional arguments.
        @param kwargs: Arbitrary keyword arguments.
        @return: pytorch.Tensor: The sampled noise tensor
        """
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """
        Resets the exploration tool.
        @param args: Arbitrary positional arguments.
        @param kwargs: Arbitrary keyword arguments.
        """
        pass

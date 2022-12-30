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


class Exploration(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs) -> pytorch.Tensor:
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        pass

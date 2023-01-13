"""!
@package rlpack.utils.base
@brief This package implements the base classes to be used across rlpack


Currently following base classes have been implemented:
    - `Agent`: Base class for all agents, implemented as rlpack.utils.base.agent.Agent.
    - `TrainerBase`: BAse class for all trainers, implemented as rlpack.utils.base.trainer.Trainer.
    - `Model`: Base class for all models, implemented as rlpack.utils.base.model.Model.

Following packages are part of utils:
    - `registers`: A package for base classes for registers. Implemented in rlpack.utils.base.registers.
"""


from abc import abstractmethod

from rlpack import pytorch


class Model(pytorch.nn.Module):
    """
    Base class for all Models in RLPack. This class inherits from pytorch.nn.Module.
    """
    @abstractmethod
    def __init__(self):
        """
        Abstract init method for Model.
        """
        super().__init__()
        self.has_exploration_tool = False
        pass

    @abstractmethod
    def forward(self, *args) -> pytorch.Tensor:
        """
        Abstract forward method for Model
        @param args: Arbitrary positional arguments.
        @return: pytorch.Tensor: The output of the model after forward pass.
        """
        pass

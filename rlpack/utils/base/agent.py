"""!
@package rlpack.utils.base
@brief This package implements the base classes to be used across rlpack


Currently following base classes have been implemented:
    - `Agent`: Base class for all agents, implemented as rlpack.utils.base.agent.Agent.
    - `Register`: Register of information of all in-built models and agents implemented as
        rlpack.utils.base.register.Register.
"""


from typing import Any, Dict, List, Tuple, Union

import numpy as np

from rlpack import pytorch


class Agent(object):
    """
    The base class for all agents.
    """

    def __init__(self):
        """
        The class initializer. Defines basic variables useful for all agents.
        """
        ## The state normalisation codes; indicating the codes to normalise states. @I{# noqa: E266}
        self._state_norm_codes = (0, 3, 4)
        ## The reward normalisation codes; indicating the codes to normalise rewards. @I{# noqa: E266}
        self._reward_norm_codes = (1, 3)
        ## The TD normalisation codes; indicating the codes to normalise TD Errors. @I{# noqa: E266}
        self._td_norm_codes = (2, 4)
        ## The Advantage normalisation codes; indicating the codes to normalise Advantages. @I{# noqa: E266}
        self._advantage_norm_codes = (2, 4)
        ## The list of losses accumulated after each backward call. @I{# noqa: E266}
        self.loss = list()
        ## The path to save agent states and models. @I{# noqa: E266}
        self.save_path = str()

    def train(self, *args, **kwargs) -> Any:
        """
        Training method for the agent. This class needs to be overriden for every agent that inherits it.
        @param args: Positional arguments for train method.
        @param kwargs: Keyword arguments for train method.
        @return Any: Action to be taken.
        """
        pass

    def policy(self, *args, **kwargs) -> Any:
        """
        Policy method for the agent. This class needs to be overriden for every agent that inherits it
        @param args: Positional arguments for policy method
        @param kwargs: Keyword arguments for policy method.
        @return Any: Action to be taken.
        """
        pass

    def save(self, *args, **kwargs) -> None:
        """
        Save method for the agent. This class needs to be overriden for every agent that inherits it.
        All necessary agent states and attributes must be saved in the implementation such that training can
        be restarted.
        @param args: Positional arguments for save method.
        @param kwargs: Keyword arguments for save method.
        """
        pass

    def load(self, *args, **kwargs) -> None:
        """
        Load method for the agent. This class needs to be overriden for every agent that inherits it.
        All necessary agent states and attributes must be loaded in the implementation such that training can
        be restarted.
        @param args: Positional arguments for load method.
        @param kwargs: Keyword arguments for load method.
        """
        pass

    def __getstate__(self) -> Dict[str, Any]:
        """
        To get the agent's current state (dict of attributes).
        @return Dict[str, Any]: The agent's states in dictionary.
        """
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        To load the agent's current state (dict of attributes).
        @param state: Dict[str, Any]: The agent's states in dictionary.
        """
        self.__dict__ = state

    @staticmethod
    def _cast_to_tensor(
        data: Union[List, Tuple, np.ndarray, pytorch.Tensor]
    ) -> pytorch.Tensor:
        """
        Helper function to cast data to tensor.
        @param data: Union[List, Tuple, np.ndarray, pytorch.Tensor]: The data to convert to tensor.
        @return pytorch.Tensor: The tensor from the input data.
        """
        if isinstance(data, pytorch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = pytorch.from_numpy(data)
        elif isinstance(data, (list, tuple)):
            tensor = pytorch.tensor(data)
        else:
            raise TypeError(
                f"Invalid type received!"
                f" Expected one of {pytorch.Tensor}, {np.ndarray}, {list}, {tuple}, got {type(data)}."
            )
        return tensor

    @staticmethod
    def _adjust_dims_for_tensor(
        tensor: pytorch.Tensor, target_dim: int
    ) -> pytorch.Tensor:
        """
        Helper function to adjust dimensions of tensor. This only works for tensors when they have a single axis.
        along any dimension and doesn't change underlying data or change the storage.
        @param tensor: pytorch.Tensor: The tensor whose dimensions are required to be changed.
        @param target_dim: int: The target number of dimensions.
        @return pytorch.Tensor: The tensor with adjusted dimensions.
        """
        if target_dim is None:
            return tensor
        curr_dim = tensor.dim()
        if target_dim > curr_dim:
            for _ in range(target_dim - curr_dim):
                tensor = pytorch.unsqueeze(tensor, dim=-1)
        else:
            for _ in range(curr_dim - target_dim):
                tensor = pytorch.squeeze(tensor, dim=-1)
        return tensor

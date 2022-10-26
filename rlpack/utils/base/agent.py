from typing import Any, Dict, List, Tuple, Union

import numpy as np

from rlpack import pytorch


class Agent(object):
    """
    The base class for all agents.
    """

    def __init__(self):
        self.state_norm_codes = (0, 3, 4)
        self.reward_norm_codes = (1, 3)
        self.td_norm_codes = (2, 4)
        self.loss = list()
        self.save_path = ""

    def train(self, *args, **kwargs):
        pass

    def policy(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def __getstate__(self) -> Dict[str, Any]:
        """
        @:return (Dict[str, Any]): The agent's states in dictionary.
        """
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        @:param state (Dict[str, Any]): The agent's states in dictionary.
        """
        self.__dict__ = state

    @staticmethod
    def _cast_to_tensor(
        data: Union[List, Tuple, np.ndarray, pytorch.Tensor]
    ) -> pytorch.Tensor:
        """
        Helper function to cast data to tensor.
        @:param data (Union[List, Tuple, np.ndarray, pytorch.Tensor]): The data to convert to tensor.
        @:return (pytorch.Tensor): The tensor from the input data.
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
        Helper function to adjust dimensions of tensor. This only works for tensors when they have a single axis
        along any dimension and doesn't change underlying data or change the storage.

        @:param tensor (pytorch.Tensor): The tensor whose dimensions are required to be changed.
        @:param target_dim (int): The target number of dimensions.
        @:return (pytorch.Tensor): The tensor with adjusted dimensions.
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

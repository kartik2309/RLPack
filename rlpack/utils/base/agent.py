from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch


class Agent(object):
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

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__ = state

    @staticmethod
    def _cast_to_tensor(
        data: Union[List, Tuple, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif isinstance(data, (list, tuple)):
            tensor = torch.tensor(data)
        else:
            raise TypeError(
                f"Invalid type received!"
                f" Expected one of {torch.Tensor}, {np.ndarray}, {list}, {tuple}, got {type(data)}."
            )
        return tensor

    @staticmethod
    def _adjust_dims_for_tensor(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        if target_dim is None:
            return tensor
        curr_dim = tensor.dim()
        if target_dim > curr_dim:
            for _ in range(target_dim - curr_dim):
                tensor = tensor.unsqueeze(tensor, dim=-1)
        else:
            for _ in range(curr_dim - target_dim):
                tensor = torch.squeeze(tensor, dim=-1)

        return tensor

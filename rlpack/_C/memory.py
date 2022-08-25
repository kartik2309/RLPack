from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from numpy import ndarray

from rlpack import C_Memory, pytorch


class Memory(object):
    def __init__(self, buffer_size=None, device="cpu"):
        self.c_memory = C_Memory.C_Memory(buffer_size, device)
        self.buffer_size = buffer_size
        self.device = device
        self.__c_memory_attr = {
            "data": lambda: {
                k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
            },
            "terminal_state_indices": lambda: [
                v for v in self.c_memory.c_memory_data.terminal_state_indices_deref
            ],
            "states_current": lambda: {
                k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
            }["states_current"],
            "states_next": lambda: {
                k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
            }["states_next"],
            "rewards": lambda: {
                k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
            }["rewards"],
            "actions": lambda: {
                k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
            }["actions"],
            "dones": lambda: {
                k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
            }["dones"],
        }

    def insert(
        self,
        state_current: np.ndarray,
        state_next: np.ndarray,
        reward: Union[np.ndarray, float],
        action: Union[np.ndarray, float],
        done: Union[np.ndarray, float],
    ) -> None:
        is_terminal_state = False
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward)
            reward = self._adjust_dims_for_array(
                reward, len(state_current.shape), dtype=np.float32
            )
        if not isinstance(action, np.ndarray):
            action = np.array(action)
            action = self._adjust_dims_for_array(
                action, len(state_current.shape), dtype=np.int64
            )
        if not isinstance(done, np.ndarray):
            if isinstance(done, bool):
                if done:
                    is_terminal_state = True
                done = int(done)
            if done == 1:
                is_terminal_state = True
            done = np.array(done)
            done = self._adjust_dims_for_array(
                done, len(state_current.shape), dtype=np.int32
            )
        state_current = pytorch.from_numpy(state_current)
        state_next = pytorch.from_numpy(state_next)
        reward = pytorch.from_numpy(reward)
        action = pytorch.from_numpy(action)
        done = pytorch.from_numpy(done)
        self.c_memory.insert(
            state_current, state_next, reward, action, done, is_terminal_state
        )

    def reserve(self, buffer_size: int) -> None:
        self.c_memory.reserve(buffer_size)

    def sample(
        self, batch_size: int, force_terminal_state_probability: float = 0.0
    ) -> Tuple[
        pytorch.Tensor, pytorch.Tensor, pytorch.Tensor, pytorch.Tensor, pytorch.Tensor
    ]:
        samples = self.c_memory.sample(batch_size, force_terminal_state_probability)
        return (
            samples["states_current"],
            samples["states_next"],
            samples["rewards"],
            samples["actions"],
            samples["dones"],
        )

    def clear(self) -> None:
        self.c_memory.clear()

    def view(self) -> C_Memory.C_MemoryData:
        return self.c_memory.view()

    def __getitem__(self, index: int) -> List[np.ndarray]:
        return self.c_memory.get_item(index)

    def __delitem__(self, index: int) -> None:
        self.c_memory.delete_item(index)

    def __len__(self) -> int:
        return self.c_memory.size()

    def __getstate__(self) -> Dict[str, Any]:
        return {"memory": self.view(), "__dict__": self.__dict__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state["__dict__"]
        self.c_memory.initialize(state["memory"])

    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __getattr__(self, item: str) -> Union[ndarray, List[Union[float, int]]]:
        if item not in self.__c_memory_attr.keys():
            raise ValueError("Invalid attribute!")
        result = self.__c_memory_attr[item]()
        return result

    def __repr__(self) -> str:
        return f"<Python object for {self.c_memory.__repr__()} at {hex(id(self))}>"

    def __str__(self) -> str:
        updated_attr = {k: v() for k, v in self.__c_memory_attr.items()}
        return f"{updated_attr}"

    @staticmethod
    def _adjust_dims_for_array(
        array: np.ndarray, target_dim: int, dtype: Type[np.ndarray] = np.float
    ) -> np.ndarray:
        if target_dim is None:
            return array
        curr_dim = array.ndim
        if target_dim > curr_dim:
            for _ in range(target_dim - curr_dim):
                array = np.expand_dims(array, axis=-1).astype(dtype)
        else:
            for _ in range(curr_dim - target_dim):
                array = np.squeeze(array, axis=-1).astype(dtype)

        return array

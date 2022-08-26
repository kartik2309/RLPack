import os
from typing import Any, Dict, List, Tuple, Type, Union, Optional

import numpy as np

from rlpack import C_Memory, pytorch


class Memory(object):
    def __init__(self, buffer_size: Optional[int] = 32768, device: Optional[str] = "cpu"):
        self.c_memory = C_Memory.C_Memory(buffer_size, device)
        self.buffer_size = buffer_size
        self.device = device
        self.data = lambda: {
            k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
        }
        self.terminal_state_indices = lambda: {
            k: v
            for k, v in self.c_memory.c_memory_data.terminal_state_indices_deref.items()
        }
        self.states_current = lambda: {
            k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
        }["states_current"]
        self.states_next = lambda: {
            k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
        }["states_next"]
        self.rewards = lambda: {
            k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
        }["rewards"]
        self.actions = lambda: {
            k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
        }["actions"]
        self.dones = lambda: {
            k: v for k, v in self.c_memory.c_memory_data.data_deref.items()
        }["dones"]
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        os.environ["OMP_NUM_THREADS"] = f"{os.cpu_count()}"

    def insert(
        self,
        state_current: np.ndarray,
        state_next: np.ndarray,
        reward: Union[np.ndarray, float],
        action: Union[np.ndarray, float],
        done: Union[np.ndarray, float],
    ) -> None:
        is_terminal_state = False
        if isinstance(state_current, np.ndarray):
            state_current = pytorch.from_numpy(state_current)
        if isinstance(state_next, np.ndarray):
            state_next = pytorch.from_numpy(state_next)
        reward = pytorch.tensor(reward)
        action = pytorch.tensor(action)
        if done:
            is_terminal_state = True
        done = pytorch.tensor(done)
        self.c_memory.insert(
            state_current, state_next, reward, action, done, is_terminal_state
        )

    def sample(
        self,
        batch_size: int,
        force_terminal_state_probability: float = 0.0,
        parallelism_size_threshold: int = 4096,
    ) -> Tuple[
        pytorch.Tensor, pytorch.Tensor, pytorch.Tensor, pytorch.Tensor, pytorch.Tensor
    ]:
        samples = self.c_memory.sample(
            batch_size, force_terminal_state_probability, parallelism_size_threshold
        )
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

    def __getattribute__(self, item: str) -> Any:
        result = super(Memory, self).__getattribute__(item)
        c_memory_attr = [
            "data",
            "terminal_state_indices",
            "states_current",
            "states_next",
            "rewards",
            "actions",
            "dones",
        ]
        if item in c_memory_attr:
            result = result()
        return result

    def __getattr__(self, item: str) -> Any:
        return self.__dict__[item]

    def __repr__(self) -> str:
        return f"<Python object for {self.c_memory.__repr__()} at {hex(id(self))}>"

    def __str__(self) -> str:
        updated_attr = {k: self.__dict__[v]() for k, v in self.__c_memory_attr}
        return f"{updated_attr}"

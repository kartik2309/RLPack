import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from rlpack import C_Memory, pytorch


class Memory(object):
    """
    This class provides the python interface to C_Memory, the C++ class which performs heavier workloads.
    """

    def __init__(
        self, buffer_size: Optional[int] = 32768, device: Optional[str] = "cpu"
    ):
        """
        @:param buffer_size (Optional[int]): The buffer size of the memory. No more than specified buffer
            elements are stored in the memory. Default: 32768
        @:param device str: The device on which models are currently running. Default: "cpu"
        """
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
        """
        This method performs insertion to the memory.
        @:param state_current (np.ndarray): The current state agent is in.
        @:param state_next (np.ndarray): The next state agent will go in for the specified action.
        @:param reward (Union[np.ndarray, float]): The reward obtained in the transition.
        @:param action (Union[np.ndarray, float]): The action taken for the transition.
        @:param done (Union[np.ndarray, float]): Indicates weather episodes ended or not, i.e.
            if state_next is a terminal state or not.
        """
        is_terminal_state = False
        if not isinstance(state_current, np.ndarray) or not isinstance(
            state_next, np.ndarray
        ):
            raise TypeError(
                f"Expected arguments `state_current` and `state_next` to be of type {np.ndarray}"
            )

        state_current = pytorch.from_numpy(state_current)
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
        """
        Load random samples from memory for a given batch.

        @:param batch_size (int): The desired batch size of the samples.
        @:param force_terminal_state_selection_prob (float): The probability for forcefully selecting a terminal state
            in a batch. Default: 0.0
        @:param parallelism_size_threshold: The minimum size of memory beyond which parallelism is used to shuffle
            and retrieve the batch of sample. Default: 4096.
        @:return (Tuple[pytorch.Tensor, pytorch.Tensor, pytorch.Tensor, pytorch.Tensor, pytorch.Tensor]): The tuple
            of tensors as (states_current, states_next, rewards, actions, dones).
        """
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
        """
        This method clear the memory and renders it empty
        """
        self.c_memory.clear()

    def view(self) -> C_Memory.C_MemoryData:
        """
        This method returns the view of Memory, i.e. the data stored in the memory.
        @:return (C_Memory.C_MemoryData): The C_MemoryData object which packages the current memory information.
            This object is pickleable and data can also be accessed via attributes.
        """
        return self.c_memory.view()

    def initialize(self, memory_data: C_Memory.C_MemoryData) -> None:
        """
        This loads the memory from the provided C_MemoryData instance.
        @:param memory_data (C_Memory.C_MemoryData): The C_MemoryData instance to load the memory form
        """
        self.c_memory.initialize(memory_data)

    def __getitem__(self, index: int) -> List[pytorch.Tensor]:
        """
        Indexing method for memory.
        @:param index (int): The index at which we want to obtain the memory data.
        @:return (List[pytorch.Tensor]): The transition as tensors from the memory.
        """
        return self.c_memory.get_item(index)

    def __delitem__(self, index: int) -> None:
        """
        Deletion method for memory
        @:param index (int): Index at which we want to delete an item.

        Note that this operation can be expensive depending on the size of memory; O(n).
        """
        self.c_memory.delete_item(index)

    def __len__(self) -> int:
        """
        Length method for memory
        @:return (int): The size of the memory
        """
        return self.c_memory.size()

    def __getstate__(self) -> Dict[str, Any]:
        """
        Get state method for memory. This makes this Memory class pickleable.
        @:return (Dict[str, Any]): The state of the memory.
        """
        return {"memory": self.view(), "__dict__": self.__dict__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Set state method for the memory
        @:param state (Dict[str, Any]): This method loads the states back to memory instance. This helps unpickle
            the Memory.
        """
        self.__dict__ = state["__dict__"]
        self.c_memory.initialize(state["memory"])

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set attr method for memory
        @:param key (str): The desired attribute name.
        @:param value (Any): The value for corresponding key.
        """
        self.__dict__[key] = value

    def __getattribute__(self, item: str) -> Any:
        """
        Get attribute method for memory
        @:param item: The attribute name we wish to access
        @:return (Any): The attribute value for the passed item

        The following attributes are attributes from C_MemoryData and can be retrieved into python by calling
        the attribute on the memory instance.
            - data: A dictionary with keys states_current, states_next, rewards, actions, dones and their
                corresponding value in list of tensors.
            - terminal_state_indices: A dictionary with single key 'terminal_state_indices' with
                list of tensors. of terminal state indices seen so far.
            - states_current: A dictionary with single key 'states_current' with
                list of tensors of states_current values seen so far.
            - states_next: A dictionary with single key 'states_next' with
                list of tensors of states_next values seen so far.
            - rewards: A dictionary with single key 'rewards' with
                list of tensors of rewards values seen so far.
            - actions: A dictionary with single key 'actions' with
                list of tensors of actions values seen so far.
            - dones: A dictionary with single key 'dones' with
                list of tensors of dones values seen so far.
        """
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
        """
        Get attr method for memory
        @:param item: The attributes that has been set during runtime (through __setattr__)
        @:return (Any): The value for the item pass.
        """
        return self.__dict__[item]

    def __repr__(self) -> str:
        """
        Repr method for memory.
        @:return (str): String with object's memory location
        """
        return f"<Python object for {self.c_memory.__repr__()} at {hex(id(self))}>"

    def __str__(self) -> str:
        """
        The str method for memory.
        :return: The dictionary with encapsulated data of memory.
        """
        updated_attr = {**self.data, **self.terminal_state_indices}
        return f"{updated_attr}"

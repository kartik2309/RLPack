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
        @:param device (str): The device on which models are currently running. Default: "cpu"
        """
        self.c_memory = C_Memory.C_Memory(buffer_size, device)
        self.buffer_size = buffer_size
        self.device = device
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        os.environ["OMP_NUM_THREADS"] = f"{os.cpu_count()}"

    def insert(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        state_next: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        reward: Union[np.ndarray, float],
        action: Union[np.ndarray, float],
        done: Union[bool, int],
        priority: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1.0,
        probability: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1.0,
        weight: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1.0,
    ) -> None:
        """
        This method performs insertion to the memory.
        @:param state_current Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current
            state agent is in.
        @:param state_next Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The next
            state agent will go in for the specified action.
        @:param reward (Union[np.ndarray, float]): The reward obtained in the transition.
        @:param action (Union[np.ndarray, float]): The action taken for the transition.
        @:param done Union[bool, int]: Indicates weather episodes ended or not, i.e.
            if state_next is a terminal state or not.
        @:param priority (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The priority of the
            transition (for priority relay memory). Default: None
        @:param probability (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The probability of the transition
            (for priority relay memory). Default: None
        @:param weight (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The important sampling weight
            of the transition (for priority relay memory). Default: None
        """

        self.c_memory.insert(
            *self.__prepare_inputs_c_memory_(
                state_current,
                state_next,
                reward,
                action,
                done,
                priority,
                probability,
                weight,
            )
        )

    def sample(
        self,
        batch_size: int,
        force_terminal_state_probability: float = 0.0,
        parallelism_size_threshold: int = 4096,
        is_prioritized: bool = False,
    ) -> Tuple[
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
    ]:
        """
        Load random samples from memory for a given batch.
        @:param batch_size (int): The desired batch size of the samples.
        @:param force_terminal_state_selection_prob (float): The probability for forcefully selecting a terminal state
            in a batch. Default: 0.0
        @:param parallelism_size_threshold: The minimum size of memory beyond which parallelism is used to shuffle
            and retrieve the batch of sample. Default: 4096.
        @:param is_prioritized (bool): If memory is prioritized or non-prioritized. Default: False
        @:return (Tuple[
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
            ]): The tuple of tensors as (states_current, states_next, rewards, actions, dones, priorities,
            probabilities, weights, random_indices).
        """
        samples = self.c_memory.sample(
            batch_size,
            force_terminal_state_probability,
            parallelism_size_threshold,
            is_prioritized,
        )
        return (
            samples["states_current"],
            samples["states_next"],
            samples["rewards"],
            samples["actions"],
            samples["dones"],
            samples["priorities"],
            samples["probabilities"],
            samples["weights"],
            samples["random_indices"],
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

    def get_terminal_state_indices(self) -> List[int]:
        """
        This retrieves the terminal state indices accumulated so far.
        :return (List[int]): The list of terminal state indices
        """
        return [
            v
            for v in self.c_memory.view().terminal_state_indices()[
                "terminal_state_indices"
            ]
        ]

    def get_transitions(self) -> Dict[str, pytorch.Tensor]:
        """
        This retrieves all the transitions accumulated so far.
        :return (Dict[str, pytorch.Tensor]): A dictionary with all transition information
        """
        return {k: v for k, v in self.c_memory.view().transition_information().items()}

    def get_states_current(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the current states from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with current state values.
        """
        return [
            v for v in self.c_memory.view().transition_information()["states_current"]
        ]

    def get_states_next(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the next states from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with next state values.
        """
        return [v for v in self.c_memory.view().transition_information()["states_next"]]

    def get_rewards(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the rewards from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with reward values.
        """
        return [v for v in self.c_memory.view().transition_information()["rewards"]]

    def get_actions(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the actions from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with action values.
        """
        return [v for v in self.c_memory.view().transition_information()["actions"]]

    def get_dones(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the dones from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with done values.
        """
        return [v for v in self.c_memory.view().transition_information()()["dones"]]

    def get_priorities(self) -> List[float]:
        """
        This retrieves all the priorities for all the transitions, ordered by index.
        :return (List[float]): A list of priorities ordered by index.
        """
        return [v for v in self.c_memory.view().priorities()["priorities"]]

    def num_terminal_states(self) -> int:
        """
        Returns the number of terminal states
        :return (int): Num of terminal states
        """
        return self.c_memory.num_terminal_states()

    @staticmethod
    def __prepare_inputs_c_memory_(
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        state_next: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        reward: Union[pytorch.Tensor, np.ndarray, float],
        action: Union[pytorch.Tensor, np.ndarray, float],
        done: Union[bool, int],
        priority: Union[pytorch.Tensor, np.ndarray, float],
        probability: Union[pytorch.Tensor, np.ndarray, float],
        weight: Union[pytorch.Tensor, np.ndarray, float],
    ) -> Tuple[
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        bool,
    ]:
        """
        Prepares inputs to be sent to C++ backend.
        @:param state_current Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current
            state agent is in.
        @:param state_next Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The next
            state agent will go in for the specified action.
        @:param reward (Union[np.ndarray, float]): The reward obtained in the transition.
        @:param action (Union[np.ndarray, float]): The action taken for the transition.
        @:param done Union[bool, int]: Indicates weather episodes ended or not, i.e.
            if state_next is a terminal state or not.
        @:param priority (Union[pytorch.Tensor, np.ndarray, float]): The priority of the
            transition (for priority relay memory). Default: None
        @:param probability (Union[pytorch.Tensor, np.ndarray, float]): The probability of the transition
            (for priority relay memory). Default: None
        @:param weight (Union[pytorch.Tensor, np.ndarray, float]): The important sampling weight
            of the transition (for priority relay memory). Default: None
        @:return (Tuple[
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                bool
            ]):  The tuple of in order of (state_current, state_next, reward, action, done, priority,
             probability, weight, is_terminal_state).
             `is_terminal_state` indicates if the state is terminal state or not (corresponds to done).
            All the input values associated with transition tuple are type-casted to PyTorch Tensors.
        """
        if not isinstance(state_current, np.ndarray) or not isinstance(
            state_next, np.ndarray
        ):
            raise TypeError(
                f"Expected arguments `state_current` and `state_next` to be of type {np.ndarray}"
            )

        if isinstance(state_current, np.ndarray):
            state_current = pytorch.from_numpy(state_current)
        elif isinstance(state_current, list):
            state_current = pytorch.tensor(state_current)
        elif isinstance(state_current, pytorch.Tensor):
            pass
        else:
            raise TypeError(
                f"Expected argument `state_current` to be of type"
                f" {pytorch.Tensor}, {np.ndarray} or {list}"
                f" but got {type(state_current)}"
            )

        if isinstance(state_next, np.ndarray):
            state_next = pytorch.from_numpy(state_next)
        elif isinstance(state_next, list):
            state_next = pytorch.tensor(state_next)
        elif isinstance(state_next, pytorch.Tensor):
            pass
        else:
            raise TypeError(
                f"Expected argument `state_next` to be of type"
                f" {pytorch.Tensor}, {np.ndarray} or {list}"
                f" but got {type(state_next)}"
            )

        # Handle reward
        if isinstance(reward, np.ndarray):
            reward = pytorch.from_numpy(reward)
        elif isinstance(reward, (int, float)):
            reward = pytorch.tensor(reward)
        elif isinstance(reward, pytorch.Tensor):
            pass
        else:
            raise TypeError(
                f"Expected argument `reward` to be of type {pytorch.Tensor}, {np.ndarray} or {float} or {int}"
                f"but got type {type(reward)}"
            )

        # Handle action
        if isinstance(action, np.ndarray):
            action = pytorch.from_numpy(action)
        elif isinstance(action, (float, int)):
            action = pytorch.tensor(action)
        elif isinstance(action, pytorch.Tensor):
            pass
        else:
            raise TypeError(
                f"Expected argument `action` to be of type {pytorch.Tensor}, {np.ndarray}, {float} or {int}"
                f"but got type {type(action)}"
            )

        # Handle done variable
        is_terminal_state = False
        if isinstance(done, bool):
            is_terminal_state = done
            done = pytorch.tensor(int(done))
        elif isinstance(done, int):
            done = pytorch.tensor(done)
            if done == 1:
                is_terminal_state = False
        else:
            raise TypeError(
                f"Expected argument `done` to be of type {bool} or {int} "
                f"but got type {type(done)}"
            )

        # Handle priority
        if isinstance(priority, np.ndarray):
            priority = pytorch.from_numpy(priority)
        elif isinstance(priority, float):
            priority = pytorch.tensor(priority)
        elif isinstance(priority, pytorch.Tensor):
            pass
        else:
            raise TypeError(
                f"Expected argument `priority` to be of type {pytorch.Tensor}, {np.ndarray} or {float} "
                f"but got type {type(priority)}"
            )

        # Handle probability
        if isinstance(probability, np.ndarray):
            probability = pytorch.from_numpy(probability)
        elif isinstance(probability, float):
            probability = pytorch.tensor(probability)
        elif isinstance(probability, pytorch.Tensor):
            pass
        else:
            raise TypeError(
                f"Expected argument `probability` to be of type {pytorch.Tensor}, {np.ndarray} or {float} "
                f"but got type {type(probability)}"
            )

        # Handle weight
        if isinstance(weight, np.ndarray):
            weight = pytorch.from_numpy(weight)
        elif isinstance(weight, float):
            weight = pytorch.tensor(weight)
        elif isinstance(weight, pytorch.Tensor):
            pass
        else:
            raise TypeError(
                f"Expected argument `weight` to be of type {pytorch.Tensor}, {np.ndarray} or {float} "
                f"but got type {type(weight)}"
            )

        return (
            state_current,
            state_next,
            reward,
            action,
            done,
            priority,
            probability,
            weight,
            is_terminal_state,
        )

    def __getitem__(self, index: int) -> List[pytorch.Tensor]:
        """
        Indexing method for memory.
        @:param index (int): The index at which we want to obtain the memory data.
        @:return (List[pytorch.Tensor]): The transition as tensors from the memory.
        """
        return self.c_memory.get_item(index)

    def __setitem__(
        self,
        index: int,
        transition: Tuple[
            Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
            Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[bool, int],
            Union[pytorch.Tensor, np.ndarray, float],
            Union[pytorch.Tensor, np.ndarray, float],
            Union[pytorch.Tensor, np.ndarray, float],
        ],
    ) -> None:
        """
        Set item method for the memory.
        @:param index (int): index to insert
        @:param transition (Tuple[
                Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
                Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
                Union[np.ndarray, float],
                Union[np.ndarray, float],
                Union[bool, int],
                Union[pytorch.Tensor, np.ndarray, float],
                Union[pytorch.Tensor, np.ndarray, float],
                Union[pytorch.Tensor, np.ndarray, float]
            ]): The transition tuple in the order (state_current, state_next, reward, action, done,
             priority, probability, weight).
        """
        self.c_memory.set_item(
            index,
            *self.__prepare_inputs_c_memory_(
                transition[0],
                transition[1],
                transition[2],
                transition[3],
                transition[4],
                transition[5],
                transition[6],
                transition[7],
            ),
        )

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
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Set state method for the memory
        @:param state (Dict[str, Any]): This method loads the states back to memory instance. This helps unpickle
            the Memory.
        """
        self.__dict__.update(state)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set attr method for memory
        @:param key (str): The desired attribute name.
        @:param value (Any): The value for corresponding key.
        """
        self.__dict__[key] = value

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
        return f"<Python object for {repr(self.c_memory)} at {hex(id(self))}>"

    def __str__(self) -> str:
        """
        The str method for memory. Useful for printing the memory.
        On calling print(memory), will print the transition information.
        :return: The dictionary with encapsulated data of memory.
        """
        return f"{self.get_transitions()}"

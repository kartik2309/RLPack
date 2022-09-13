from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from rlpack import C_Memory, pytorch


class Memory(object):
    """
    This class provides the python interface to C_Memory, the C++ class which performs heavier workloads.
    """

    def __init__(
        self,
        buffer_size: Optional[int] = 32768,
        parallelism_size_threshold: int = 8092,
        device: Optional[str] = "cpu",
        prioritized: bool = False,
    ):
        """
        @:param buffer_size (Optional[int]): The buffer size of the memory. No more than specified buffer
            elements are stored in the memory. Default: 32768
        @:param parallelism_size_threshold (int): The minimum size of memory beyond which parallelism is used to shuffle
            and retrieve the batch of sample. Default: 4096.
        @:param device (str): The device on which models are currently running. Default: "cpu".
        @:param prioritized (bool): Indicates weather memory is a prioritized relay. Default: False
        """
        self.c_memory = C_Memory.C_Memory(
            buffer_size, device, parallelism_size_threshold
        )
        self.buffer_size = buffer_size
        self.parallelism_size_threshold = parallelism_size_threshold
        self.device = device
        self.prioritized = prioritized
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        os.environ["OMP_NUM_THREADS"] = f"{os.cpu_count()}"

    def insert(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        state_next: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        reward: Union[np.ndarray, float],
        action: Union[np.ndarray, float],
        done: Union[bool, int],
        priority: Optional[Union[pytorch.Tensor, np.ndarray, float]] = None,
        probability: Optional[Union[pytorch.Tensor, np.ndarray, float]] = None,
        weight: Optional[Union[pytorch.Tensor, np.ndarray, float]] = None,
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
        if self.prioritized:
            assert priority is not None, (
                "When using prioritized memory, argument `priority` must be passed of "
                "type Union[pytorch.Tensor, np.ndarray, float]]!"
            )
            assert probability is not None, (
                "When using prioritized memory, argument `probability` must be passed of "
                "type Union[pytorch.Tensor, np.ndarray, float]]!"
            )
            assert weight is not None, (
                "When using prioritized memory, argument `weight` must be passed of "
                "type Union[pytorch.Tensor, np.ndarray, float]]!"
            )
        else:
            # Set to default values of 1.0. When un-prioritized memory is used,
            #   uniform distribution sampling will be used
            priority, probability, weight = 1.0, 1.0, 1.0

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
    ) -> Tuple[
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
        @:return (Tuple[pytorch.Tensor, pytorch.Tensor, pytorch.Tensor, pytorch.Tensor, pytorch.Tensor]): The tuple
            of tensors as (states_current, states_next, rewards, actions, dones, random_indices, weights).
        """
        samples = self.c_memory.sample(
            batch_size, force_terminal_state_probability, self.prioritized
        )
        return (
            samples["states_current"],
            samples["states_next"],
            samples["rewards"],
            samples["actions"],
            samples["dones"],
            samples["random_indices"],
            samples["weights"],
        )

    def update_transition_priorities(
        self,
        indices: Union[pytorch.Tensor, List[int]],
        new_priorities: Union[List[pytorch.Tensor], pytorch.Tensor],
        alpha: float,
        beta: float,
    ) -> None:
        """
        Updates the transition priorities. This will also update corresponding probabilities and weights.
        @:param indices (List[int]): The indices in which changes are to be done. If tensor is passed, must be a 1d
            tensor.
        @:param new_priorities (Union[pytorch.Tensor, List[int]]): A list of Tensors or a single tensor,
            representing values of new priorities for each of the index in `indices`.
        @:param alpha (float): The alpha value to compute probability for each transition.
        @:param beta (float): The beta value to update important sampling weights.
        """
        self.c_memory.update_transition_priorities(indices, new_priorities, alpha, beta)

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
        return {k: v for k, v in self.c_memory.view().transitions().items()}

    def get_states_current(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the current states from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with current state values.
        """
        return [v for v in self.c_memory.view().transitions()["states_current"]]

    def get_states_next(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the next states from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with next state values.
        """
        return [v for v in self.c_memory.view().transitions()["states_next"]]

    def get_rewards(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the rewards from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with reward values.
        """
        return [v for v in self.c_memory.view().transitions()["rewards"]]

    def get_actions(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the actions from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with action values.
        """
        return [v for v in self.c_memory.view().transitions()["actions"]]

    def get_dones(self) -> List[pytorch.Tensor]:
        """
        This retrieves all the dones from transitions accumulated so far.
        :return (List[pytorch.Tensor]): A list of tensors with done values.
        """
        return [v for v in self.c_memory.view().transitions()()["dones"]]

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
        priority: Optional[Union[pytorch.Tensor, np.ndarray, float]] = None,
        probability: Optional[Union[pytorch.Tensor, np.ndarray, float]] = None,
        weight: Optional[Union[pytorch.Tensor, np.ndarray, float]] = None,
    ) -> Tuple[
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        pytorch.Tensor,
        Union[None, pytorch.Tensor],
        Union[None, pytorch.Tensor],
        Union[None, pytorch.Tensor],
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
        @:param priority (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The priority of the
            transition (for priority relay memory). Default: None
        @:param probability (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The probability of the transition
            (for priority relay memory). Default: None
        @:param weight (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The important sampling weight
            of the transition (for priority relay memory). Default: None
        @:return (Tuple[
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                pytorch.Tensor,
                Union[None, pytorch.Tensor],
                Union[None, pytorch.Tensor],
                Union[None, pytorch.Tensor],
            ]):  The tuple of in order of (state_current, state_next, reward, action, done, priority,
             probability, weight),
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
        if isinstance(done, bool):
            done = pytorch.tensor(int(done))
        elif isinstance(done, int):
            done = pytorch.tensor(done)
        else:
            raise TypeError(
                f"Expected argument `done` to be of type {bool} or {int} "
                f"but got type {type(done)}"
            )

        # Handle priority
        if priority is not None:
            if isinstance(priority, np.ndarray):
                priority = pytorch.from_numpy(priority)
            elif isinstance(priority, float):
                priority = pytorch.tensor(priority)
            elif isinstance(priority, pytorch.Tensor):
                pass
            else:
                raise TypeError(
                    f"Expected argument `action` to be of type {np.ndarray} or {float} "
                    f"but got type {type(priority)}"
                )

        # Handle probability
        if probability is not None:
            if isinstance(probability, np.ndarray):
                probability = pytorch.from_numpy(probability)
            elif isinstance(probability, float):
                probability = pytorch.tensor(probability)
            elif isinstance(probability, pytorch.Tensor):
                pass
            else:
                raise TypeError(
                    f"Expected argument `action` to be of type {np.ndarray} or {float} "
                    f"but got type {type(probability)}"
                )

        # Handle weight
        if weight is not None:
            if isinstance(weight, np.ndarray):
                weight = pytorch.from_numpy(weight)
            elif isinstance(weight, float):
                weight = pytorch.tensor(weight)
            elif isinstance(weight, pytorch.Tensor):
                pass
            else:
                raise TypeError(
                    f"Expected argument `action` to be of type {np.ndarray} or {float} "
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
            ]): The transition tuple in the order (state_current, state_next, reward, action, done).
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
        for k, v in state.items():
            setattr(self, k, v)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set attr method for memory
        @:param key (str): The desired attribute name.
        @:param value (Any): The value for corresponding key.
        """
        self.__dict__[key] = value

    def __delattr__(self, item: str) -> None:
        """
        Delete an attribute
        :param item (str): The attribute name we wish to delete.
        """
        del self.__dict__[item]

    def __getattr__(self, item: str) -> Any:
        """
        Get attr method for memory
        @:param item: The attributes that has been set during runtime (through __setattr__)
        @:return (Any): The value for the item pass.
        """
        return self.__dict__[item]

    def __copy__(self) -> Memory:
        """
        Performs shallow copy of memory instance.
        :return (Memory): New Memory instance with reference to self (current instance).
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: Dict[Any, Any] = None) -> Memory:
        """
        Performs deep copy of memory instance.
        :param memo (Dict[Any, Any]): Memo for deepcopy
        :return (Memory): New memory instance with different storage.
        """
        if memo is None:
            memo = dict()
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __repr__(self) -> str:
        """
        Repr method for memory.
        @:return (str): String with object's memory location
        """
        return f"<Python object for {repr(self.c_memory)} at {hex(id(self))}>"

    def __str__(self) -> str:
        """
        The str method for memory.
        :return: The dictionary with encapsulated data of memory.
        """
        return f"{self.get_transitions()}"

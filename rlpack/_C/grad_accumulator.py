"""!
@package rlpack._C
@brief This package implements the classes to interface between C++ and Python.


Currently following classes have been implemented:
    - `ReplayBuffer`: Implemented as rlpack._C.replay_buffer.ReplayBuffer, this class is responsible for using
        Optimized C_ReplayBuffer class implemented in C++ and providing simple Python methods to access it.
    - `GradAccumulator`: Implemented as rlpack._C.grad_accumulator.GradAccumulator, this class is responsible for
        using optimized C_GradAccumulator class implemented in C++ and providing simple python methods to access it.
    - `RolloutBuffer`: Implemented as rlpack._C.rollout_buffer.RolloutBuffer, this class is responsible for using
        C_RolloutBuffer class implemented in C++ and providing simple python methods to access it.
"""


from typing import Iterable, List

from rlpack import C_GradAccumulator


class GradAccumulator:
    """
    This class provides the python interface to C_GradAccumulator, the C++ class which performs heavier workloads.
    This class is used for accumulating gradients and performing reduction operations on it.
    """

    def __init__(self, parameter_keys: List[str], bootstrap_rounds: int):
        """
        @param parameter_keys: List[str]: The parameter keys (names) of the model.
        @param bootstrap_rounds: int: The bootstrap rounds defined the agent.
        """
        ## The instance of C_GradAccumulator; the C++ backend of GradAccumulator class. @I{# noqa: E266}
        self.c_grad_accumulator = C_GradAccumulator.C_GradAccumulator(
            parameter_keys, bootstrap_rounds
        )
        ## The instance of MapOfTensors; the custom object used by C++ backend. @I{# noqa: E266}
        self.map_of_tensors = C_GradAccumulator.MapOfTensors()

    def accumulate(self, named_parameters: Iterable) -> None:
        """
        Accumulates the parameters from the model. C++ backend extracts the gradients from the parameters.
        @param named_parameters: Iterable for parameters (use model.named_parameters()).
        """
        for name, param in named_parameters:
            self.map_of_tensors[name] = param
        self.c_grad_accumulator.accumulate(self.map_of_tensors)

    def mean_reduce(self) -> C_GradAccumulator.MapOfTensors:
        """
        Performs the mean reduction of accumulated gradients.
        @return MapOfTensors: The custom map object from C++ backend with mean of gradient of parameters for each key.
        """
        mean_reduced_params = self.c_grad_accumulator.mean_reduce()
        return mean_reduced_params

    def sum_reduce(self) -> C_GradAccumulator.MapOfTensors:
        """
        Performs the sum reduction of accumulated gradients.
        @return MapOfTensors: The custom map object from C++ backend with sum of gradient of parameters for each key.
        """
        sum_reduced_params = self.c_grad_accumulator.sum_reduce()
        return sum_reduced_params

    def clear(self) -> None:
        """
        Clears the accumulated gradients.
        """
        self.c_grad_accumulator.clear()

    def __getitem__(self, index: int) -> C_GradAccumulator.MapOfTensors:
        """
        Retrieve named parameter gradients at a given index.
        @param index: int: The index at which we wish to obtain the gradient values.
        @return MapOfTensors: The custom map object from C++ backend with gradient of parameters for each key.
        """
        return self.c_grad_accumulator.get_item(index)

    def __setitem__(self, index: int, named_parameters: Iterable) -> None:
        """
        Set named parameter gradients at a given index.
        @param index: int: The index at which we wish to set the gradient values.
        @param named_parameters: Iterable for parameters (use model.named_parameters()).
        """
        self.c_grad_accumulator.set_item(index, named_parameters)

    def __delitem__(self, index: int):
        """
        Set named parameter gradients at a given index.
        @param index: int: The index at which we wish to set the gradient values.
        """
        self.c_grad_accumulator.delete_item(index)

    def __len__(self):
        """
        Get the number of named parameters' accumulated gradients so far.
        @return int: The size of GradAccumulator.
        """
        return self.c_grad_accumulator.size()

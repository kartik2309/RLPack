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


from rlpack import C_RolloutBuffer, pytorch


class RolloutBuffer:
    def __init__(self, buffer_size: int, device: str = "cpu", dtype: str = "float32"):
        """
        Initialization method for RolloutBuffer.
        @param buffer_size: int: The buffer size to be used. This allocates the memory accordingly. For on-policy
            methods this is typically the maximum number of timesteps per episodes
        @param device: str: The device for on which tensors are kept and processed.
        @param dtype: str: The datatype to for tensors.
        """
        ## The instance of C_RolloutBuffer; the C++ backend of C_RolloutBuffer class. @I{# noqa: E266}
        self.c_rollout_buffer = C_RolloutBuffer.C_RolloutBuffer(
            buffer_size, device, dtype
        )
        ## The instance of MapOfTensors; the custom object used by C++ backend. @I{# noqa: E266}
        self.map_of_tensors = C_RolloutBuffer.MapOfTensors()

    def insert(
        self,
        reward: pytorch.Tensor,
        action_log_probability: pytorch.Tensor,
        state_current_value: pytorch.Tensor,
        entropy: pytorch.Tensor,
    ) -> None:
        """
        Insertion method to the rollout buffer. This method moves the tensors to MapOfTensors to make them opaque
        hence keeping the computational graph intact even when tensors are processed with C++ backend.
        @param reward: pytorch.Tensor: The reward obtained at the given timestep.
        @param action_log_probability: pytorch.Tensor: The log probability of sampled action in the current
            distribution for current timestep.
        @param state_current_value: pytorch.Tensor: The current state values for the timestep.
        @param entropy: pytorch.Tensor: The entropy values for the distribution for the given timestep
        """
        self.map_of_tensors["reward"] = reward
        self.map_of_tensors["action_log_probability"] = action_log_probability
        self.map_of_tensors["state_current_value"] = state_current_value
        self.map_of_tensors["entropy"] = entropy
        self.c_rollout_buffer.insert(self.map_of_tensors)

    def compute_returns(self, gamma: float) -> pytorch.Tensor:
        """
        Computes the returns for the rewards accumulated so far given the gamma.
        @param gamma: float: The discounting factor for the agent.
        @return: pytorch.Tensor: The tensor of computed returns.
        """
        return self.c_rollout_buffer.compute_returns(gamma)["returns"]

    def get_stacked_rewards(self) -> pytorch.Tensor:
        """
        Gets the stacked rewards accumulated so far.
        @return: pytorch.Tensor: The tensor of rewards.
        """
        return self.c_rollout_buffer.get_stacked_rewards()["rewards"]

    def get_stacked_action_log_probabilities(self) -> pytorch.Tensor:
        """
        Gets the stacked log probabilities of action in the given distribution accumulated so far.
        @return: pytorch.Tensor: The tensor of log of action probabilities of action.
        """
        return self.c_rollout_buffer.get_stacked_action_log_probabilities()[
            "action_log_probabilities"
        ]

    def get_stacked_state_current_values(self) -> pytorch.Tensor:
        """
        Gets the stacked current states accumulated so far.
        @return: pytorch.Tensor: The tensor of current states.
        """
        return self.c_rollout_buffer.get_stacked_state_current_values()[
            "state_current_values"
        ]

    def get_stacked_entropies(self) -> pytorch.Tensor:
        """
        Gets the stacked entropies accumulated so far.
        @return: pytorch.Tensor: The tensor of entropies.
        """
        return self.c_rollout_buffer.get_stacked_entropies()["entropies"]

    def clear(self) -> None:
        """
        Clears the accumulated quantities so far. This will not de-allocate the memory.
        """
        self.c_rollout_buffer.clear()

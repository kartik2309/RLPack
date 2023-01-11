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


from datetime import timedelta
from typing import Union

from rlpack import C_RolloutBuffer, pytorch, pytorch_distributed


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        device: str = "cpu",
        dtype: str = "float32",
        process_group: Union[pytorch_distributed.ProcessGroup, None] = None,
        work_timeout: timedelta = timedelta(minutes=30),
    ):
        """
        Initialization method for RolloutBuffer.
        @param buffer_size: int: The buffer size to be used. This allocates the memory accordingly. For on-policy
            methods this is typically the maximum number of timesteps per episodes
        @param device: str: The device for on which tensors are kept and processed.
        @param dtype: str: The datatype to for tensors.
        @param process_group: Union[pytorch_distributed.ProcessGroup, None]: The current process group being used
            in distributed setting. If not in a distributed setting, must be None. Default: None.
        @param work_timeout: timedelta: The duration for work wait for all processes to complete the
            gather process when RolloutBuffer.extend_transitions is called in distributed setting.
            Default: 30 minutes.
        """
        process_group_map = C_RolloutBuffer.ProcessGroupMap()
        process_group_map["process_group"] = process_group
        ## The instance of C_RolloutBuffer; the C++ backend of C_RolloutBuffer class. @I{# noqa: E266}
        self.c_rollout_buffer = C_RolloutBuffer.C_RolloutBuffer(
            buffer_size, device, dtype, process_group_map, work_timeout
        )
        ## The instance of TensorMap; the custom object used by C++ backend. @I{# noqa: E266}
        self.map_of_tensors = C_RolloutBuffer.TensorMap()

    def insert_transition(
        self,
        state_current: pytorch.Tensor,
        state_next: pytorch.Tensor,
        reward: pytorch.Tensor,
        done: pytorch.Tensor,
    ) -> None:
        """
        Insertion method for transitions to the rollout buffer. This method moves the tensors to
        TensorMap to make them opaque hence keeping the computational graph intact even when tensors are
        processed with C++ backend.
        @param state_current: pytorch.Tensor: The current state for the timestep.
        @param state_next: pytorch.Tensor: The next state after taking the specific action.
        @param reward: pytorch.Tensor: The reward obtained at the given timestep.
        @param done: pytorch.Tensor: The done flag indicating if episode has terminated/truncated.
        """
        self.map_of_tensors["state_current"] = state_current
        self.map_of_tensors["state_next"] = state_next
        self.map_of_tensors["reward"] = reward
        self.map_of_tensors["done"] = done
        self.c_rollout_buffer.insert_transition(self.map_of_tensors)

    def insert_policy_output(
        self,
        action_log_probability: pytorch.Tensor,
        state_current_value: pytorch.Tensor,
        state_next_value: pytorch.Tensor,
        entropy: pytorch.Tensor,
    ) -> None:
        """
        Insertion method for policy outputs in the rollout buffer. This method moves the tensors to
        TensorMap to make them opaque hence keeping the computational graph intact even when tensors are
        processed with C++ backend.
        @param action_log_probability: pytorch.Tensor: The log probability of sampled action in the current
            distribution for current timestep.
        @param state_current_value: pytorch.Tensor: The current state values for the timestep.
        @param state_next_value: pytorch.Tensor: The next state values for the timestep.
        @param entropy: pytorch.Tensor: The entropy values for the distribution for the given timestep
        """
        self.map_of_tensors["action_log_probability"] = action_log_probability
        self.map_of_tensors["state_current_value"] = state_current_value
        self.map_of_tensors["state_next_value"] = state_next_value
        self.map_of_tensors["entropy"] = entropy
        self.c_rollout_buffer.insert_policy_output(self.map_of_tensors)

    def compute_returns(self, gamma: float) -> pytorch.Tensor:
        """
        Computes the returns for the rewards accumulated so far given the gamma.
        @param gamma: float: The discounting factor for the agent.
        @return: pytorch.Tensor: The tensor of computed returns.
        """
        return self.c_rollout_buffer.compute_returns(gamma)["returns"]

    def compute_discounted_td_residuals(self, gamma: float) -> pytorch.Tensor:
        """
        Computes the TD residual error for state values accumulated so far given the gamma.
        @param gamma: float: The discounting factor for the agent.
        @return: pytorch.Tensor: The tensor of computed TD residual errors.
        """
        return self.c_rollout_buffer.compute_discounted_td_residual(gamma)[
            "td_residuals"
        ]

    def compute_generalized_advantage_estimates(
        self, gamma: float, gae_lambda: float
    ) -> pytorch.Tensor:
        """
        Computes the Generalized Advantage Estimate (GAE) for given gamma and lambda.
        @param gamma: float: The discounting factor for the agent.
        @param gae_lambda: float: The bias-variance trade-off parameter.
        @return: pytorch.Tensor: The tensor of computed generalized advantage estimates.
        """
        return self.c_rollout_buffer.compute_generalized_advantage_estimates(
            gamma, gae_lambda
        )["advantages"]

    def get_stacked_states_current(self) -> pytorch.Tensor:
        """
        Gets the stacked states current accumulated so far.
        @return: pytorch.Tensor: The tensor of states current.
        """
        return self.c_rollout_buffer.get_stacked_states_current()["states_current"]

    def get_stacked_states_next(self) -> pytorch.Tensor:
        """
        Gets the stacked states next accumulated so far.
        @return: pytorch.Tensor: The tensor of next states.
        """
        return self.c_rollout_buffer.get_stacked_states_next()["states_next"]

    def get_states_statistics(self) -> C_RolloutBuffer.TensorMap:
        return self.c_rollout_buffer.get_states_statistics()

    def get_stacked_rewards(self) -> pytorch.Tensor:
        """
        Gets the stacked rewards accumulated so far.
        @return: pytorch.Tensor: The tensor of rewards.
        """
        return self.c_rollout_buffer.get_stacked_rewards()["rewards"]

    def get_stacked_dones(self) -> pytorch.Tensor:
        """
        Gets the stacked dones accumulated so far.
        @return: pytorch.Tensor: The tensor of dones.
        """
        return self.c_rollout_buffer.get_stacked_dones()["dones"]

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
        Gets the stacked current states' values accumulated so far.
        @return: pytorch.Tensor: The tensor of current states values.
        """
        return self.c_rollout_buffer.get_stacked_state_current_values()[
            "state_current_values"
        ]

    def get_stacked_state_next_values(self) -> pytorch.Tensor:
        """
        Gets the stacked next states' values accumulated so far.
        @return: pytorch.Tensor: The tensor of next states values.
        """
        return self.c_rollout_buffer.get_stacked_state_next_values()[
            "state_next_values"
        ]

    def get_stacked_entropies(self) -> pytorch.Tensor:
        """
        Gets the stacked entropies accumulated so far.
        @return: pytorch.Tensor: The tensor of entropies.
        """
        return self.c_rollout_buffer.get_stacked_entropies()["entropies"]

    def clear_transitions(self) -> None:
        """
        Clears the accumulated transitions so far. This will not de-allocate the memory.
        """
        self.c_rollout_buffer.clear_transitions()

    def clear_policy_outputs(self) -> None:
        """
        Clears the accumulated policy outputs so far. This will not de-allocate the memory.
        """
        self.c_rollout_buffer.clear_policy_outputs()

    def transition_at(self, index: int) -> C_RolloutBuffer.TensorMap:
        """
        Returns the transitions at a given index
        @param index: int: The index from which transition is to be obtained
        @return: StlBindings.TensorMap: The custom map object from C++ backend with transitions with keys
        """
        return self.c_rollout_buffer.transition_at(index)

    def policy_output_at(self, index: int) -> C_RolloutBuffer.TensorMap:
        """
        Returns the policy outputs at a given index
        @param index: int: The index from which policy output is to be obtained
        @return: StlBindings.TensorMap: The custom map object from C++ backend with policy outputs with keys
        """
        return self.c_rollout_buffer.policy_output_at(index)

    def size_transitions(self) -> int:
        """
        Returns the size of transition buffer. This indicates the number of transitions accumulated so far.
        @return: int: The size of transition buffer.
        """
        return self.c_rollout_buffer.size_transitions()

    def size_policy_outputs(self) -> int:
        """
        Returns the size of policy output buffer. This indicates the number of policy outputs accumulated so far.
        @return: int: The size of policy output buffer.
        """
        return self.c_rollout_buffer.size_policy_outputs()

    def extend_transitions(self) -> None:
        """
        Method to extend the transitions. This method will perform gather and extend transition with
        transitions from other RolloutBuffer instances from other process. This is done synchronously and
        hence will be a blocking call.


        The transitions are only extended for master process and this method will raise an error if called
        from non-distributed setting.
        """
        self.c_rollout_buffer.extend_transitions()

    def __len__(self) -> int:
        """
        Gets the total length of Rollout Buffer.
        @return: int: The size of the Rollout Buffer
        """
        return (
            self.c_rollout_buffer.size_transitions()
            + self.c_rollout_buffer.size_policy_outputs()
        )

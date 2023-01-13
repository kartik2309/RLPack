"""!
@package rlpack.actor_critic.utils
@brief This package implements the utilities for Actor-Critic methods.


Currently following methods are implemented:
    - `ActorCriticAgent`: Implemented as rlpack.actor_critic.utils.ActorCriticAgent. This is the base class for
        all actor-critic methods and implements some basic methods to be used across different actor critic methods.
"""

import os
from abc import abstractmethod
from datetime import timedelta
from typing import List, Optional, Tuple, Type, Union

import numpy as np

from rlpack import pytorch, pytorch_distributions
from rlpack._C.grad_accumulator import GradAccumulator
from rlpack._C.rollout_buffer import RolloutBuffer
from rlpack.exploration.utils.exploration import Exploration
from rlpack.utils import LossFunction, LRScheduler
from rlpack.utils.base.agent import Agent
from rlpack.utils.base.model import Model
from rlpack.utils.exceptions import AgentError
from rlpack.utils.internal_code_setup import InternalCodeSetup
from rlpack.utils.normalization import Normalization


class ActorCriticAgent(Agent):
    """
    The ActorCriticAgent is the base class for actor-critic methods. This class implements basic methods and
    abstract functions for actor-critic methods.
    """

    def __init__(
        self,
        policy_model: Model,
        optimizer: pytorch.optim.Optimizer,
        lr_scheduler: Union[LRScheduler, None],
        loss_function: LossFunction,
        distribution: Type[pytorch_distributions.Distribution],
        gamma: float,
        entropy_coefficient: float,
        state_value_coefficient: float,
        lr_threshold: float,
        action_space: Union[int, Tuple[int, Union[List[int], None]]],
        backup_frequency: int,
        save_path: str,
        gae_lambda: float = 1.0,
        batch_size: int = 1,
        exploration_steps: Union[int, None] = None,
        grad_accumulation_rounds: int = 1,
        training_frequency: Union[int, None] = None,
        exploration_tool: Union[Exploration, None] = None,
        device: str = "cpu",
        dtype: str = "float32",
        normalization_tool: Union[Normalization, None] = None,
        apply_norm_to: Union[int, List[str]] = -1,
        max_grad_norm: Optional[float] = None,
        grad_norm_p: float = 2.0,
        clip_grad_value: Optional[float] = None,
        timeout: timedelta = timedelta(minutes=30),
    ):
        """!
        @param policy_model: Model: The policy model to be used. Policy model must return a tuple of
            action logits and state values.
        @param optimizer: pytorch.optim.Optimizer: The optimizer to be used for policy model. Optimizer must be
            initialized and wrapped with policy model parameters.
        @param lr_scheduler: Union[LRScheduler, None]: The LR Scheduler to be used to decay the learning rate.
            LR Scheduler must be initialized and wrapped with passed optimizer.
        @param loss_function: LossFunction: A PyTorch loss function.
        @param distribution : Type[pytorch_distributions.Distribution]: The distribution of PyTorch to be used to
            sampled actions in action space. (See `action_space`).
        @param gamma: float: The discounting factor for rewards.
        @param entropy_coefficient: float: The coefficient to be used for entropy in policy loss computation.
        @param state_value_coefficient: float: The coefficient to be used for state value in final loss computation.
        @param lr_threshold: float: The threshold LR which once reached LR scheduler is not called further.
        @param action_space: Union[int, Tuple[int, Union[List[int], None]]]: The action space of the environment.
            - If discrete action set is used, number of actions can be passed.
            - If continuous action space is used, a list must be passed with first element representing
                the output features from model, second element representing the shape of action to be sampled. Second
                element can be an empty list, if you wish to sample the default no. of samples.
        @param backup_frequency: int: The timesteps after which policy model, optimizer states and lr
            scheduler states are backed up.
        @param save_path: str: The path where policy model, optimizer states and lr scheduler states are to be saved.
        @param gae_lambda: float: The Generalized Advantage Estimation coefficient (referred to as lambda), indicating
            the bias-variance trade-off.
        @param batch_size: int: The batch size to be used while training policy model. Default: 1
        @param exploration_steps: Union[int, None]: The size of rollout buffer before performing optimizer
            step. Whole rollout buffer is used to fit the policy model and is cleared. By default, after every episode.
             Default: None.
        @param grad_accumulation_rounds: int: The number of rounds until which gradients are to be accumulated before
            performing calling optimizer step. Gradients are mean reduced for grad_accumulation_rounds > 1. Default: 1.
        @param training_frequency: Union[int, None]: The number of timesteps after which policy model is to be trained.
            By default, training is done at the end of an episode: Default: None.
        @param exploration_tool: Union[Exploration, None]: Exploration tool to be used to explore the environment.
            These tools can be found in `rlpack.exploration`.
        @param device: str: The device on which models are run. Default: "cpu".
        @param dtype: str: The datatype for model parameters. Default: "float32".
        @param normalization_tool: Union[Normalization, None]: The normalization tool to be used. This must be an
            instance of rlpack.utils.normalization.Normalization if passed. By default, is initialized to None and
            no normalization takes place. If passed, make sure a valid `apply_norm_to` is passed.
        @param apply_norm_to: Union[int, List[str]]: The code to select the quantity to which normalization is
            to be applied. Direct list of quantities can also be passed as per accepted keys. Refer
            below in Notes to see the accepted values. Default: -1.
        @param max_grad_norm: Optional[float]: The max norm for gradients for gradient clipping. Default: None
        @param grad_norm_p: float: The p-value for p-normalization of gradients. Default: 2.0
        @param clip_grad_value: Optional[float]: The gradient value for clipping gradients by value. Default: None
        @param timeout: timedelta: The timeout for synchronous calls. Default is 30 minutes.


        **Notes**

        The value accepted for `apply_norm_to` are as follows and must be passed in a list:
            - `"none"`: -1; Don't apply normalization to any quantity.
            - `"states"`: 0; Apply normalization to states.
            - `"state_values"`: 1; Apply normalization to state values.
            - `"rewards"`: 2; Apply normalization to rewards.
            - `"returns"`: 3; Apply normalization to rewards.
            - `"td"`: 4; Apply normalization for TD values.
            - `"advantage"`: 5; Apply normalization to advantage values


        If a valid `max_norm_grad` is passed, then gradient clipping takes place else gradient clipping step is
        skipped. If `max_norm_grad` value was invalid, error will be raised from PyTorch.
        If a valid `clip_grad_value` is passed, then gradients will be clipped by value. If `clip_grad_value` value
        was invalid, error will be raised from PyTorch.
        """
        super(ActorCriticAgent, self).__init__()
        setup = InternalCodeSetup()
        ## The input policy model moved to desired device. @I{# noqa: E266}
        self.policy_model = policy_model.to(
            device=pytorch.device(device=device), dtype=setup.get_torch_dtype(dtype)
        )
        ## The input optimizer wrapped with policy_model parameters. @I{# noqa: E266}
        self.optimizer = optimizer
        ## The input optional LR Scheduler (this can be None). @I{# noqa: E266}
        self.lr_scheduler = lr_scheduler
        ## The input loss function. @I{# noqa: E266}
        self.loss_function = loss_function
        ## The input distribution object. @I{# noqa: E266}
        self.distribution = distribution
        ## The input discounting factor. @I{# noqa: E266}
        self.gamma = gamma
        ## The input entropy coefficient. @I{# noqa: E266}
        self.entropy_coefficient = entropy_coefficient
        ## The input state value coefficient. @I{# noqa: E266}
        self.state_value_coefficient = state_value_coefficient
        ## The input LR Threshold. @I{# noqa: E266}
        self.lr_threshold = float(lr_threshold)
        # Check validity of action space before setting the attribute.
        setup.check_validity_of_action_space(action_space)
        ## The input number of actions. @I{# noqa: E266}
        self.action_space = action_space
        ## The input model backup frequency in terms of timesteps. @I{# noqa: E266}
        self.backup_frequency = backup_frequency
        ## The input save path for backing up agent models. @I{# noqa: E266}
        self.save_path = save_path
        ## The input GAE Lambda (from argument `gae_lambda`). @I{# noqa: E266}
        self.gae_lambda = gae_lambda
        # Check sanity of `batch_size`.
        if training_frequency is None:
            assert (
                batch_size == 1
            ), "`batch_size` must be 1 if `training_frequency` is not passed or None is passed!"
        else:
            assert (
                batch_size <= training_frequency
            ), "`batch_size` must be smaller than or equal to `training_frequency`"
        ## The input batch size (from argument `batch_size`). @I{# noqa: E266}
        self.batch_size = batch_size
        # Check sanity of `grad_accumulation_rounds`
        assert (
            grad_accumulation_rounds > 0
        ), "Argument `grad_accumulation_rounds` must be an integer greater than 0"
        ## The input `rollout_accumulation_size`. @I{# noqa: E266}
        self.exploration_steps = exploration_steps
        ## The input `grad_accumulation_rounds`. @I{# noqa: E266}
        self.grad_accumulation_rounds = grad_accumulation_rounds
        ## The input `training_frequency`. @I{# noqa: E266}
        self.training_frequency = training_frequency
        ## The input `exploration_tool`. @I{# noqa: E266}
        self.exploration_tool = exploration_tool
        ## The input `device` argument; indicating the device name as device type class. @I{# noqa: E266}
        self.device = pytorch.device(device=device)
        ## The input `dtype` argument; indicating the datatype class. @I{# noqa: E266}
        self.dtype = setup.get_torch_dtype(dtype)
        # Get code(s) for `apply_norm_to` and check validity.
        apply_norm_to = setup.get_apply_norm_to_mode_code(apply_norm_to)
        setup.check_validity_of_apply_norm_to_codes(apply_norm_to)
        ## The input `apply_norm_to` argument; indicating the quantity to normalise. @I{# noqa: E266}
        self.apply_norm_to = apply_norm_to
        ## The input `max_grad_norm`; indicating the maximum gradient norm for gradient clippings. @I{# noqa: E266}
        self.max_grad_norm = max_grad_norm
        ## The input `grad_norm_p`; indicating the p-value for p-normalisation for gradient clippings. @I{# noqa: E266}
        self.grad_norm_p = grad_norm_p
        ## The input `clip_grad_value`; indicating the clipping range for gradients. @I{# noqa: E266}
        self.clip_grad_value = clip_grad_value
        ## The input `timeout`; indicating the timeout for synchronous calls. @I{# noqa: E266}
        self.timeout = timedelta(milliseconds=(timeout.seconds * 1e3))
        ## The step counter; counting the total timesteps done so far. @I{# noqa: E266}
        self.step_counter = 0
        ## Flag indicating if action space is continuous or discrete. @I{# noqa: E266}
        self.is_continuous_action_space = True
        if isinstance(self.action_space, int):
            self.is_continuous_action_space = False
        ## Flag indicating the agent has an exploration initialized. @I{# noqa: E266}
        self._agent_has_exploration_tool = (
            True if self.exploration_tool is not None else False
        )
        ## Flag indicating the model has an exploration initialized. @I{# noqa: E266}
        self._policy_model_has_exploration_tool = self.policy_model.has_exploration_tool
        ## Flag indicating whether to perform post-forward exploration (for exploring policy outputs). @I{# noqa: E266}
        self._policy_outputs_exploration = (
            True if self.exploration_steps is not None else False
        )
        # Parameter keys of the model.
        keys = list(dict(self.policy_model.named_parameters()).keys())
        ## The list of gradients from each backward call. @I{# noqa: E266}
        ## This is only used when boostrap_rounds > 1 and is cleared after each boostrap round. @I{# noqa: E266}
        ## The rlpack._C.grad_accumulator.GradAccumulator object for grad accumulation. @I{# noqa: E266}
        self._grad_accumulator = GradAccumulator(keys, grad_accumulation_rounds)
        ## The normalisation tool to be used for agent. @I{# noqa: E266}
        ## An instance of rlpack.utils.normalization.Normalization. @I{# noqa: E266}
        self._normalization = normalization_tool
        ## The process rank for multi-agents. For uni-agents, will be None. @I{# noqa: E266}
        self._process_rank = None
        ## The world size for multi-agents. For uni-agents, will be None. @I{# noqa: E266}
        self._world_size = None
        ## The master process id for multi-agents. Is set to 0 as a standard. @I{# noqa: E266}
        self._master_process_rank = 0
        ## The process group currently being used. For uni-agents, will be None. @I{# noqa: E266}
        self._process_group = None
        ## The flag indicating weather to perform forward pass. @I{# noqa: E266}
        self._take_forward_step = True
        ## The flag indicating weather to perform backward pass. @I{# noqa: E266}
        self._take_backward_step = True
        ## The flag indicating weather to perform gradient processing (normalization and clipping). @I{# noqa: E266}
        self._perform_grad_processing = True
        ## The flag indicating weather to take optimizer step. It is dangerous to modify this externally.  @I{# noqa: E266}
        self._take_optimizer_step = True
        ## The flag indicating weather to take LR scheduler step. It is dangerous to modify this externally. @I{# noqa: E266}
        self._take_lr_scheduler_step = True
        # Call to set
        self._set_attribute_custom_values()
        ## The rollout buffer to be used for agent to store necessary outputs. @I{# noqa: E266}
        ## An instance of rlpack._C.rollout_buffer.RolloutBuffer @I{# noqa: E266}
        self._rollout_buffer = RolloutBuffer(
            buffer_size=self._get_rollout_buffer_size(
                training_frequency=training_frequency,
                exploration_steps=exploration_steps,
            ),
            device=device,
            dtype=dtype,
            process_group=self._process_group,
            work_timeout=timeout,
        )

    def train(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        state_next: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        reward: Union[int, float],
        done: Union[bool, int],
        **kwargs,
    ) -> np.ndarray:
        """
        The train method to train the agent and underlying policy model.
        @param state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current state.
        @param state_next: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The next state returned.
        @param reward: Union[int, float]: The reward returned from previous action
        @param done: Union[bool, int]: Flag indicating if episode has terminated or not
        @param kwargs: Other keyword arguments.
        @return np.ndarray: The action to be taken
        """
        state_current = self._cast_to_tensor(state_current).to(dtype=self.dtype)
        state_next = self._cast_to_tensor(state_next).to(dtype=self.dtype)
        self._rollout_buffer.insert_transition(
            state_current=state_current,
            state_next=state_next,
            reward=self._cast_to_tensor(reward).to(dtype=self.dtype),
            done=self._cast_to_tensor(int(done)),
        )
        # Run exploration until `exploration_steps` if not None.
        if self.exploration_steps is not None:
            if (self.step_counter + 1) % self.exploration_steps == 0:
                self._finish_transitions_exploration()
        # Run training as per `training_frequency`
        else:
            if self.training_frequency is not None:
                if (self.step_counter + 1) % self.training_frequency == 0:
                    self._train_policy_model()
            else:
                if self._is_done(done):
                    self._train_policy_model()
        # Backup model every `backup_frequency` steps.
        self._call_to_save()
        # Increment `step_counter` and use policy model to get next action.
        self.step_counter += 1
        # Run in PyTorch No-Grad.
        with pytorch.no_grad():
            # Apply normalization to state current if required
            if (
                self._state_norm_code in self.apply_norm_to
                and self._normalization is not None
            ):
                state_current = self._normalization.apply_normalization_pre_silent(
                    state_current, "states"
                )
            # Perform forward pass with policy model.
            action_values, _ = self.policy_model(state_current)
            # Create action distribution from action values.
            distribution = self._create_action_distribution(action_values)
            # Create action from distribution by sampling
            action = self._get_action_from_distribution(distribution)
            # Add noise to the given action.
            action = self._add_noise_to_actions(action)
            # Move action tensor to CPU and convert to NumPy.
            action = action.cpu().numpy()
        return action

    @pytorch.no_grad()
    def policy(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        **kwargs,
    ) -> np.ndarray:
        """
        The policy method to evaluate the agent. This runs in pure inference mode.
        @param state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current state returned
            from gym environment
        @param kwargs: Other keyword arguments
        @return np.ndarray: The action to be taken
        """
        if self.policy_model.training:
            self.policy_model.eval()
        state_current = self._cast_to_tensor(state_current).to(
            device=self.device, dtype=self.dtype
        )
        if (
            self._state_norm_code in self.apply_norm_to
            and self._normalization is not None
        ):
            state_current = self._normalization.apply_normalization_pre_silent(
                state_current, "states"
            )
        action_values, _ = self.policy_model(state_current)
        distribution = self._create_action_distribution(action_values)
        action = self._get_action_from_distribution(distribution)
        action = action.cpu().numpy()
        return action

    def save(self, custom_name_suffix: Optional[str] = None) -> None:
        """
        This method saves the target_model, policy_model, optimizer, lr_scheduler and agent_states in the supplied
            `save_path` argument in the DQN Agent class' constructor (also called __init__).
        agent_states includes current memory and epsilon values in a dictionary.
        @param custom_name_suffix: Optional[str]: If supplied, additional suffix is added to names of target_model,
            policy_model, optimizer and lr_scheduler. Useful to save best model by a custom suffix supplied
            during a train run. Default: None
        """
        if custom_name_suffix is None:
            custom_name_suffix = ""
        if not isinstance(custom_name_suffix, str):
            raise TypeError(
                f"Argument `custom_name_suffix` must be of type "
                f"{str} or {type(None)}, but got of type {type(custom_name_suffix)}"
            )
        if custom_name_suffix != "":
            custom_name_suffix = f"_{custom_name_suffix}"
        save_path = self.save_path
        checkpoint = {
            "policy_model_state_dict": self.policy_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self._normalization is not None:
            checkpoint[
                "normalization_state_dict"
            ] = self._normalization.get_state_dict()
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, f"actor_critic{custom_name_suffix}.pt")
        pytorch.save(checkpoint, save_path)
        return

    def load(self, custom_name_suffix: Optional[str] = None) -> None:
        """
        This method loads the target_model, policy_model, optimizer, lr_scheduler and agent_states from
            the supplied `save_path` argument in the DQN Agent class' constructor (also called __init__).
        @param custom_name_suffix: Optional[str]: If supplied, additional suffix is added to names of target_model,
            policy_model, optimizer and lr_scheduler. Useful to load the best model by a custom suffix supplied
            for evaluation. Default: None
        """
        if custom_name_suffix is None:
            custom_name_suffix = ""
        if not isinstance(custom_name_suffix, str):
            raise TypeError(
                f"Argument `custom_name_suffix` must be of type "
                f"{str} or {type(None)}, but got of type {type(custom_name_suffix)}"
            )
        if custom_name_suffix != "":
            custom_name_suffix = f"_{custom_name_suffix}"
        save_path = self.save_path
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, f"actor_critic{custom_name_suffix}.pt")
        if not os.path.isfile(save_path):
            raise FileNotFoundError(
                "Given path does not contain the valid agent. "
                "If directory is passed, the file named `actor_critic.pth` or `actor_critic_<custom_suffix>.pth "
                "must be present, else must pass the valid file path!"
            )
        checkpoint = pytorch.load(save_path, map_location="cpu")
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            self.lr_scheduler is not None
            and "lr_scheduler_state_dict" in checkpoint.keys()
        ):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if (
            self._normalization is not None
            and "normalization_state_dict" in checkpoint.keys()
        ):
            self._normalization.load_state_dict(checkpoint["normalization_state_dict"])
        return

    @abstractmethod
    def _set_attribute_custom_values(self) -> None:
        """
        Method to set attributes for ActorCriticAgent with custom values. This method is called in __init__.
        """
        pass

    @abstractmethod
    def _call_to_save(self) -> None:
        """
        Method calling the save method when required. This method is to be overriden.
        """
        pass

    @abstractmethod
    def _call_to_extend_transitions(self) -> None:
        """
        Method calling the method RolloutBuffer.extend_transitions. This method is to be overriden by
        appropriate class.
        """
        pass

    @abstractmethod
    def _share_gradients(self) -> None:
        """
        Asynchronously averages the gradients across the world_size (number of processes) using non-blocking
        reduce method to share gradients to master process. This method is to be overriden by appropriate class.
        """
        pass

    @abstractmethod
    def _share_parameters(self) -> None:
        """
        Method to share parameters from a single model. This must typically implement scatter collective communication
        operation. This method is to be overriden by appropriate class.
        """
        pass

    def _train_policy_model(self):
        if not self.policy_model.training:
            self.policy_model.train()
        # Extend rollout buffer if required (typically by gathering from multiple actors).
        self._call_to_extend_transitions()
        # Perform forward pass in policy model if required.
        if self._take_forward_step:
            self._forward_pass()
        if self._policy_outputs_exploration and self._take_forward_step:
            self._finish_policy_outputs_exploration()
        # Perform backward pass if required.
        if self._take_backward_step:
            control_flag = self._backward_pass()
            # Reset exploration tool.
            self._reset_exploration_tool()
            if control_flag:
                return
        # Run optimizer.
        self._run_optimizer()

    def _forward_pass(self) -> None:
        """
        Method to perform forward pass. This method will iterate through batches of transitions and accumulate
        policy outputs in rollout buffer.
        """
        # Obtain the transitions iterator from rollout buffer.
        transition_iteration = self._rollout_buffer.get_transitions_iterator(
            self.batch_size
        )
        for transition in transition_iteration:
            state_next = transition["state_next"]
            state_current = transition["state_current"]
            if self._state_norm_code in self.apply_norm_to:
                state_next = self._normalization.apply_normalization_pre_silent(
                    state_next, "states"
                )
                state_current = self._normalization.apply_normalization_pre_silent(
                    state_current, "states"
                )
            # Compute state_next_value.
            _, state_next_value = self.policy_model(state_next)
            # Compute action values and state_current_values.
            action_value, state_current_value = self.policy_model(state_current)
            # Create distribution from action values.
            distribution = self._create_action_distribution(action_value)
            # Create action from distribution by sampling.
            action = self._get_action_from_distribution(
                distribution, reparametrize=True
            )
            # Add noise to actions.
            action = self._add_noise_to_actions(action)
            # Accumulate policy outputs.
            self._rollout_buffer.insert_policy_output(
                action_log_probability=distribution.log_prob(action),
                state_current_value=state_current_value,
                state_next_value=state_next_value,
                entropy=distribution.entropy(),
            )
        # Delete the transitions iterator. This will delete the object which is kept alive by C++.
        del transition_iteration

    def _backward_pass(self) -> bool:
        """
        Method to perform backward pass. This method will compute gradients and accumulate gradients if
        required.
        @return bool: Indicates if gradients were accumulated when True; False if accumulated gradients
            were reduced and loaded into policy model.
        """
        if not self._take_forward_step:
            raise AgentError(
                "Cannot take backward pass without having done forward pass!"
            )
        loss = self._compute_loss()
        # Prepare for optimizer step by setting zero grads.
        self.optimizer.zero_grad()
        # Backward call.
        loss.backward()
        # Clear the buffer values.
        self._clear_rollout_buffer()
        # Append loss to list.
        self.loss.append(loss.item())
        # Accumulate gradients if required.
        if self.grad_accumulation_rounds > 1:
            # When `grad_accumulation_rounds` is greater than 1; accumulate gradients if no. of rounds
            # specified by `grad_accumulation_rounds` have not been completed and return.
            # If no. of rounds have been completed, perform mean reduction and proceed with optimizer step.
            if len(self._grad_accumulator) < self.grad_accumulation_rounds:
                self._grad_accumulator.accumulate(self.policy_model.named_parameters())
                return True
            else:
                # Perform mean reduction.
                self._grad_mean_reduction()
                # Clear Accumulated Gradient buffer.
                self._grad_accumulator.clear()
        return False

    @pytorch.no_grad()
    def _finish_transitions_exploration(self):
        """
        Finishes exploration in transitions and sets the necessary statistics if requested for normalization.
        Sets the attribute `exploration_steps` to None
        """
        # If states are to be normalized. compute the state statistics of explored states.
        if (
            self._state_norm_code in self.apply_norm_to
            and self._normalization is not None
        ):
            states_statistics = self._rollout_buffer.get_states_statistics()
            # Update the statistics for states in dictionary
            self._normalization.update_statistics("states", dict(states_statistics))
        if self._normalization is not None:
            self._normalization.all_reduce_statistics(self._process_group, self.timeout)
        self.exploration_steps = None

    @pytorch.no_grad()
    def _finish_policy_outputs_exploration(self):
        """
        Finishes exploration in policy outputs and sets the necessary statistics if requested for
        normalization. Sets the attribute `exploration_steps` to None. Sets the attribute `_policy_outputs_exploration`
        to False.
        """
        # If advantages are to be normalized. compute the advantage statistics in explored space.
        if (
            self._advantage_norm_code in self.apply_norm_to
            and self._normalization is not None
        ):
            advantages_statistics = self._rollout_buffer.get_advantage_statistics(
                self.gamma, self.gae_lambda
            )
            # Update the statistics for advantages in dictionary
            self._normalization.update_statistics(
                "advantages", dict(advantages_statistics)
            )
        # If state values are to be normalized. compute the state value statistics in explored space.
        if (
            self._state_value_norm_code in self.apply_norm_to
            and self._normalization is not None
        ):
            state_values_statistics = self._rollout_buffer.get_state_values_statistics()
            # Update the statistics for state values in dictionary
            self._normalization.update_statistics(
                "state_values", dict(state_values_statistics)
            )
        # If action log probabilities are to be normalized, compute the state value statistics in explored space.
        if (
            self._action_log_probability_norm_code in self.apply_norm_to
            and self._normalization is not None
        ):
            action_log_probability_statistics = (
                self._rollout_buffer.get_action_log_probabilities_statistics()
            )
            # Update the statistics for action log probabilities in dictionary
            self._normalization.update_statistics(
                "action_log_probabilities", dict(action_log_probability_statistics)
            )
        # If entropies are to be normalized, compute the state value statistics in explored space.
        if (
            self._entropy_norm_code in self.apply_norm_to
            and self._normalization is not None
        ):
            entropy_statistics = self._rollout_buffer.get_entropy_statistics()
            # Update the statistics for entropies in dictionary
            self._normalization.update_statistics("entropies", dict(entropy_statistics))
        if self._normalization is not None:
            self._normalization.all_reduce_statistics(self._process_group, self.timeout)
        self._policy_outputs_exploration = False

    def _reset_exploration_tool(self) -> None:
        """
        Resets exploration tool for both policy and model if valid.
        """
        if self._agent_has_exploration_tool:
            self.exploration_tool.reset()
        if self._policy_model_has_exploration_tool:
            self.policy_model.exploration_tool.reset()
            self.policy_model.clear_noise()

    def _add_noise_to_actions(self, action: pytorch.Tensor) -> pytorch.Tensor:
        """
        Adds noise to the given action according to the exploration tool being used.
        @param action: pytorch.Tensor: The action tensors
        @return: pytorch.Tensor: The noisy action tensor.
        """
        noise = None
        if self._policy_model_has_exploration_tool:
            noise = self.policy_model.get_noise()
        elif self._agent_has_exploration_tool:
            noise = self.exploration_tool.sample()
        if noise is not None:
            action = action + noise
        return action

    def _compute_loss(self) -> pytorch.Tensor:
        """
        Method to compute total loss (from actor and critic).
        @return pytorch.Tensor: The loss tensor.
        """
        # Get action log probabilities.
        action_log_probabilities = (
            self._rollout_buffer.get_stacked_action_log_probabilities()
        )
        # Get entropy values
        entropy = self._rollout_buffer.get_stacked_entropies()
        # Get state values and normalize if required.
        state_current_values = self._rollout_buffer.get_stacked_state_current_values()
        if (
            self._state_value_norm_code in self.apply_norm_to
            and self._normalization is not None
        ):
            state_current_values = self._normalization.apply_normalization_pre_silent(
                state_current_values, "state_values", fallback=True
            )
        # Compute Advantage Values and normalize if required.
        advantage = self._rollout_buffer.compute_generalized_advantage_estimates(
            self.gamma, self.gae_lambda
        ).detach()
        if (
            self._advantage_norm_code in self.apply_norm_to
            and self._normalization is not None
        ):
            advantage = self._normalization.apply_normalization_pre_silent(
                advantage, "advantages", fallback=True
            )
        # Compute Actor Losses.
        actor_losses = (-action_log_probabilities * advantage) + (
            self.entropy_coefficient * entropy
        )
        # Compute Mean for Actor Losses.
        actor_loss = actor_losses.mean()
        # Compute Critic Loss.
        critic_loss = self.state_value_coefficient * self.loss_function(
            state_current_values, advantage
        )
        # Compute final loss
        loss = actor_loss + critic_loss
        return loss

    def _run_optimizer(self) -> None:
        """
        Protected void method to train to process gradients (gradient normalization and clipping), run optimizer and
        LR Scheduler.
        """
        # Perform distributed gradient reduction if required.
        self._share_gradients()
        # Perform gradient processing if required.
        if self._perform_grad_processing:
            # Clip gradients if requested.
            if self.max_grad_norm is not None:
                pytorch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    max_norm=self.max_grad_norm,
                    norm_type=self.grad_norm_p,
                )
            # Clip gradients by value if requested.
            if self.clip_grad_value is not None:
                pytorch.nn.utils.clip_grad_value_(
                    self.policy_model.parameters(), clip_value=self.clip_grad_value
                )
        # Take optimizer step if required.
        if self._take_optimizer_step:
            if not self._take_backward_step:
                raise AgentError("Cannot take optimizer step without backward pass!")
            self.optimizer.step()
        # Perform distributed parameter scattering if required.
        self._share_parameters()
        # Take an LR Scheduler step if required.
        if (
            self.lr_scheduler is not None
            and min([*self.lr_scheduler.get_last_lr()]) > self.lr_threshold
            and self._take_lr_scheduler_step
        ):
            self.lr_scheduler.step()

    @pytorch.no_grad()
    def _grad_mean_reduction(self) -> None:
        """
        Performs mean reduction for accumulated gradients and assigns the policy model's parameter the mean reduced
        gradients.
        """
        reduced_parameters = self._grad_accumulator.mean_reduce()
        # Assign average parameters to model.
        for key, param in self.policy_model.named_parameters():
            if param.requires_grad:
                param.grad = reduced_parameters[key]

    def _clear_rollout_buffer(self) -> None:
        """
        Protected void method to clear the rollout buffer (transitions and policy outputs)
        """
        self._rollout_buffer.clear_transitions()
        self._rollout_buffer.clear_policy_outputs()

    def _get_action_sample_shape(self) -> pytorch.Size:
        """
        Gets the action sample shape to be sampled from distribution.
        @return pytorch.Size: Sample shape of to-be sampled tensor from continuous distribution
        """
        if self.is_continuous_action_space:
            assert len(self.action_space) == 2
            if self.action_space[-1] is None:
                return pytorch.Size([])
            return pytorch.Size(self.action_space[-1])
        return pytorch.Size([])

    def _create_action_distribution(
        self,
        action_values: Union[List[pytorch.Tensor], pytorch.Tensor],
    ) -> pytorch_distributions.Distribution:
        """
        Protected static method to create distributions from action logits
        @param action_values: Union[List[pytorch.Tensor], pytorch.Tensor]: The action values from policy model
        @return Distribution: A Distribution object initialized with given action logits
        """
        if not self.is_continuous_action_space:
            distribution = self.distribution(action_values)
        else:
            if isinstance(action_values, pytorch.Tensor):
                action_values_ = action_values.flatten()
                distribution = self.distribution(*action_values_)
            elif isinstance(action_values, list):
                assert len(action_values) == 2, (
                    f"`action_values, if a list, must be a list of two tensors, "
                    f"but got list of length {len(action_values)}.\n"
                    "HINT: Check your policy model's output."
                )
                distribution = self.distribution(*action_values)
            else:
                raise TypeError(
                    f"Expected `action_values` to be either a tensor or a list of Tensors, "
                    f"received {type(action_values)}"
                )
        return distribution

    def _get_action_from_distribution(
        self,
        distribution: pytorch_distributions.Distribution,
        reparametrize: bool = False,
    ) -> pytorch.Tensor:
        """
        Creates action from the given distribution by sampling.
        @param distribution: pytorch_distributions.Distribution: The distribution object to be used.
        @param reparametrize: bool: Whether to use reparametrization trick while sampling. When set to True,
            operations are attached to computation graph. This is only valid for continuous action spaces.
            Default: False
        @return pytorch.Tensor: The sampled action tensor.
        """
        if not self.is_continuous_action_space:
            action = distribution.sample(self._get_action_sample_shape())
        else:
            sample_shape = self._get_action_sample_shape()
            if not reparametrize:
                action = distribution.sample(sample_shape=sample_shape)
            else:
                action = distribution.rsample(sample_shape=sample_shape)
        return action

    @staticmethod
    def _is_done(done: Union[bool, int]) -> bool:
        """
        A basic utility function to know if episode has terminated if there are different types of `done`
        @param done: Union[bool, int]: The done flag from OpenAI gym.
        @return: bool: The done flag in boolean
        """
        is_done = False
        if isinstance(done, bool):
            if done:
                is_done = True
        elif isinstance(done, int):
            if done == 1:
                is_done = True
        else:
            raise TypeError(
                f"Expected `done` argument to be of type {bool} or {int} but received {type(done)}!"
            )
        return is_done

    @staticmethod
    def _get_rollout_buffer_size(
        training_frequency: Union[int, None], exploration_steps: Union[int, None]
    ) -> int:
        """
        A utility function to obtain the size of rollout buffer. Note that space is reserved as per buffer size and
        exceeding buffer size can often make the program prone to error. If both arguments are None, this will be
        initialized to 1024.

        @param training_frequency: Union[int, None]: The `training_frequency` passed in ActorCriticAgent.__init__.
        @param exploration_steps: Union[int, None] The `exploration_steps` passed in ActorCriticAgent.__init__.
        @return: int: The buffer size set.
        """
        # Initializing with constant default of 1024.
        rollout_buffer_size = 1024
        if training_frequency is not None:
            if exploration_steps is not None:
                rollout_buffer_size = exploration_steps + training_frequency
        else:
            if exploration_steps is not None:
                if exploration_steps > rollout_buffer_size:
                    rollout_buffer_size = exploration_steps
        return rollout_buffer_size

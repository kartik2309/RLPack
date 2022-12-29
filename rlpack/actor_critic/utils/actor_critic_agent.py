"""!
@package rlpack.actor_critic.utils
@brief This package implements the utilities for Actor-Critic methods.


Currently following methods are implemented:
    - `ActorCriticAgent`: Implemented as rlpack.actor_critic.utils.ActorCriticAgent. This is the base class for
        all actor-critic methods and implements some basic methods to be used across different actor critic methods.
"""


import os
from abc import abstractmethod
from typing import List, Optional, Tuple, Type, Union

import numpy as np

from rlpack import pytorch, pytorch_distributions
from rlpack._C.grad_accumulator import GradAccumulator
from rlpack._C.rollout_buffer import RolloutBuffer
from rlpack.exploration.utils.exploration import Exploration
from rlpack.utils import LossFunction, LRScheduler
from rlpack.utils.base.agent import Agent
from rlpack.utils.internal_code_setup import InternalCodeSetup
from rlpack.utils.normalization import Normalization


class ActorCriticAgent(Agent):
    """
    The ActorCriticAgent is the base class for actor-critic methods. This class implements basic methods and
    abstract functions for actor-critic methods.
    """

    def __init__(
        self,
        policy_model: pytorch.nn.Module,
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
        rollout_accumulation_size: Union[int, None] = None,
        grad_accumulation_rounds: int = 1,
        exploration_tool: Union[Exploration, None] = None,
        device: str = "cpu",
        dtype: str = "float32",
        apply_norm: Union[int, str] = -1,
        apply_norm_to: Union[int, List[str]] = -1,
        eps_for_norm: float = 5e-12,
        p_for_norm: int = 2,
        dim_for_norm: int = 0,
        max_grad_norm: Optional[float] = None,
        grad_norm_p: float = 2.0,
        clip_grad_value: Optional[float] = None,
    ):
        """!
        @param policy_model: *pytorch.nn.Module*: The policy model to be used. Policy model must return a tuple of
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
            element can be an empty list or None, if you wish to sample the default no. of samples.
        @param backup_frequency: int: The timesteps after which policy model, optimizer states and lr
            scheduler states are backed up.
        @param save_path: str: The path where policy model, optimizer states and lr scheduler states are to be saved.
        @param rollout_accumulation_size: Union[int, None]: The size of rollout buffer before performing optimizer
            step. Whole rollout buffer is used to fit the policy model and is cleared. By default, after every episode.
             Default: None.
        @param grad_accumulation_rounds: int: The number of rounds until which gradients are to be accumulated before
            performing calling optimizer step. Gradients are mean reduced for grad_accumulation_rounds > 1. Default: 1.
        @param exploration_tool: Union[Exploration, None]: Exploration tool to be used to explore the environment.
            These tools can be found in `rlpack.exploration`.
        @param device: str: The device on which models are run. Default: "cpu".
        @param dtype: str: The datatype for model parameters. Default: "float32"
        @param apply_norm: Union[int, str]: The code to select the normalization procedure to be applied on
            selected quantities; selected by `apply_norm_to`: see below)). Direct string can also be
            passed as per accepted keys. Refer below in Notes to see the accepted values. Default: -1
        @param apply_norm_to: Union[int, List[str]]: The code to select the quantity to which normalization is
            to be applied. Direct list of quantities can also be passed as per accepted keys. Refer
            below in Notes to see the accepted values. Default: -1.
        @param eps_for_norm: float: Epsilon value for normalization; for numeric stability. For min-max normalization
            and standardized normalization. Default: 5e-12.
        @param p_for_norm: int: The p value for p-normalization. Default: 2; L2 Norm.
        @param dim_for_norm: int: The dimension across which normalization is to be performed. Default: 0.
        @param max_grad_norm: Optional[float]: The max norm for gradients for gradient clipping. Default: None
        @param grad_norm_p: float: The p-value for p-normalization of gradients. Default: 2.0
        @param clip_grad_value: Optional[float]: The gradient value for clipping gradients by value. Default: None

        **Notes**


        The values accepted for `apply_norm` are: -
            - No Normalization: -1; `"none"`
            - Min-Max Normalization: 0; `"min_max"`
            - Standardization: 1; `"standardize"`
            - P-Normalization: 2; `"p_norm"`


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
        # Check sanity of `grad_accumulation_rounds`
        assert (
            grad_accumulation_rounds > 0
        ), "Argument `grad_accumulation_rounds` must be an integer between 0 and 1"
        ## The input `rollout_accumulation_size`. @I{# noqa: E266}
        self.rollout_accumulation_size = rollout_accumulation_size
        ## The input `grad_accumulation_rounds`. @I{# noqa: E266}
        self.grad_accumulation_rounds = grad_accumulation_rounds
        ## The input `exploration_tool`. @I{# noqa: E266}
        self.exploration_tool = exploration_tool
        ## The input `device` argument; indicating the device name as device type class. @I{# noqa: E266}
        self.device = pytorch.device(device=device)
        ## The input `dtype` argument; indicating the datatype class. @I{# noqa: E266}
        self.dtype = setup.get_torch_dtype(dtype)
        # Get code(s) for `apply_norm` and check validity.
        apply_norm = setup.get_apply_norm_mode_code(apply_norm)
        setup.check_validity_of_apply_norm_code(apply_norm)
        ## The input `apply_norm` argument; indicating the normalisation to be used. @I{# noqa: E266}
        self.apply_norm = apply_norm
        # Get code(s) for `apply_norm_to` and check validity.
        apply_norm_to = setup.get_apply_norm_to_mode_code(apply_norm_to)
        setup.check_validity_of_apply_norm_to_codes(apply_norm_to)
        ## The input `apply_norm_to` argument; indicating the quantity to normalise. @I{# noqa: E266}
        self.apply_norm_to = apply_norm_to
        ## The input `eps_for_norm` argument; indicating epsilon to be used for normalisation. @I{# noqa: E266}
        self.eps_for_norm = eps_for_norm
        ## The input `p_for_norm` argument; indicating p-value for p-normalisation. @I{# noqa: E266}
        self.p_for_norm = p_for_norm
        ## The input `dim_for_norm` argument; indicating dimension along which we wish to normalise. @I{# noqa: E266}
        self.dim_for_norm = dim_for_norm
        ## The input `max_grad_norm`; indicating the maximum gradient norm for gradient clippings. @I{# noqa: E266}
        self.max_grad_norm = max_grad_norm
        ## The input `grad_norm_p`; indicating the p-value for p-normalisation for gradient clippings. @I{# noqa: E266}
        self.grad_norm_p = grad_norm_p
        ## The input `clip_grad_value`; indicating the clipping range for gradients. @I{# noqa: E266}
        self.clip_grad_value = clip_grad_value
        ## The step counter; counting the total timesteps done so far. @I{# noqa: E266}
        self.step_counter = 0
        ## Flag indicating if action space is continuous or discrete. @I{# noqa: E266}
        self.is_continuous_action_space = True
        if isinstance(self.action_space, int):
            self.is_continuous_action_space = False
        # Parameter keys of the model.
        keys = list(dict(self.policy_model.named_parameters()).keys())
        ## The list of gradients from each backward call. @I{# noqa: E266}
        ## This is only used when boostrap_rounds > 1 and is cleared after each boostrap round. @I{# noqa: E266}
        ## The rlpack._C.grad_accumulator.GradAccumulator object for grad accumulation. @I{# noqa: E266}
        self._grad_accumulator = GradAccumulator(keys, grad_accumulation_rounds)
        ## The normalisation tool to be used for agent. @I{# noqa: E266}
        ## An instance of rlpack.utils.normalization.Normalization. @I{# noqa: E266}
        self._normalization = Normalization(
            apply_norm=apply_norm, eps=eps_for_norm, p=p_for_norm, dim=dim_for_norm
        )
        ## The rollout buffer to be used for agent to store necessary outputs. @I{# noqa: E266}
        ## An instance of rlpack._C.rollout_buffer.RolloutBuffer @I{# noqa: E266}
        rollout_buffer_size = rollout_accumulation_size if rollout_accumulation_size is not None else 1024
        self._rollout_buffer = RolloutBuffer(rollout_buffer_size, device, dtype)
        ## The PyTorch Normal distribution object initialized with standard mean and std. @I{# noqa: E266}
        self.gaussian_noise = self._create_noise_distribution()

    def train(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        reward: Union[int, float],
        done: Union[bool, int],
        **kwargs,
    ) -> np.ndarray:
        """
        The train method to train the agent and underlying policy model.
        @param state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current state returned
        @param reward: Union[int, float]: The reward returned from previous action
        @param done: Union[bool, int]: Flag indicating if episode has terminated or not
        @param kwargs: Other keyword arguments.
        @return np.ndarray: The action to be taken
        """
        # Cast `state_current` to tensor.
        if not self.policy_model.training:
            self.policy_model.train()
        state_current = self._cast_to_tensor(state_current).to(
            device=self.device, dtype=self.dtype
        )
        action_values, state_current_value = self.policy_model(state_current)
        distribution = self._create_action_distribution(action_values)
        if not self.is_continuous_action_space:
            action = distribution.sample()
        else:
            sample_shape = self._get_action_sample_shape_for_continuous()
            action = distribution.rsample(sample_shape=sample_shape)
        # Accumulate quantities.
        self._rollout_buffer.insert(
            reward=pytorch.tensor((reward,), device=self.device, dtype=self.dtype),
            action_log_probability=distribution.log_prob(action),
            state_current_value=state_current_value,
            entropy=distribution.entropy(),
        )
        # Call train policy method.
        self._call_to_run_optimizer(done)
        # Backup model every `backup_frequency` steps.
        self._call_to_save()
        # Increment `step_counter` and use policy model to get next action.
        self.step_counter += 1
        with pytorch.no_grad():
            action = self._call_to_add_noise_to_actions(action)
            action = action.cpu().numpy()
        self._call_to_reset_exploration_tool(done)
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
        action_values, _ = self.policy_model(state_current)
        distribution = self._create_action_distribution(action_values)
        if not self.is_continuous_action_space:
            action = distribution.sample()
        else:
            sample_shape = self._get_action_sample_shape_for_continuous()
            action = distribution.sample(sample_shape=sample_shape)
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
        return

    @abstractmethod
    def _call_to_save(self) -> None:
        """
        Method calling the save method when required. This method is to be overriden.
        """
        pass

    def _call_to_run_optimizer(self, done: Union[bool, int]) -> None:
        """
        Protected method to train the policy model. If done flag is True, will compute the loss and run the optimizer.
        This method is meant to periodically check if episode hsa been terminated or and train policy models if
        episode has terminated.
        @param done: Union[bool, int]: Flag indicating if episode has terminated or not
        """
        run_optimizer = False
        if isinstance(done, bool):
            if done:
                run_optimizer = True
        elif isinstance(done, int):
            if done == 1:
                run_optimizer = True
        else:
            raise TypeError(
                f"Expected `done` argument to be of type {bool} or {int} but received {type(done)}!"
            )
        if run_optimizer:
            if self.rollout_accumulation_size is not None:
                if len(self._rollout_buffer) < self.rollout_accumulation_size:
                    return
            loss = self._compute_loss()
            self._run_optimizer(loss)
            
    def _call_to_reset_exploration_tool(self, done: Union[bool, int]) -> None:
        if self.exploration_tool is None:
            return
        reset_exploration_tool = False
        if isinstance(done, bool):
            if done:
                reset_exploration_tool = True
        elif isinstance(done, int):
            if done == 1:
                reset_exploration_tool = True
        else:
            raise TypeError(
                f"Expected `done` argument to be of type {bool} or {int} but received {type(done)}!"
            )
        if reset_exploration_tool:
            self.exploration_tool.reset()
            
    def _call_to_add_noise_to_actions(self, action: pytorch.Tensor) -> pytorch.Tensor:
        if self.exploration_tool is None:
            return action
        action = action + self.exploration_tool.sample()
        return action

    def _compute_loss(self) -> pytorch.Tensor:
        """
        Method to compute total loss (from actor and critic).
        @return pytorch.Tensor: The loss tensor.
        """
        if not self.policy_model.training:
            self.policy_model.train()
        self.policy_model.train()
        # Stack the action log probabilities.
        action_log_probabilities = (
            self._rollout_buffer.get_stacked_action_log_probabilities()
        )
        # Get entropy values
        entropy = self._rollout_buffer.get_stacked_entropies()
        # Stack the State values.
        state_current_values = self._rollout_buffer.get_stacked_state_current_values()
        # Compute returns.
        returns = self._rollout_buffer.compute_returns(self.gamma)
        # Apply normalization if required to state values.
        if self._state_value_norm_code in self.apply_norm_to:
            state_current_values = self._normalization.apply_normalization(
                state_current_values
            )
        # Apply normalization if required to returns.
        if self._returns_norm_code in self.apply_norm_to:
            returns = self._normalization.apply_normalization(returns)
        # Compute Advantage Values
        advantage = self._compute_advantage(returns, state_current_values).detach()
        # Compute Policy Losses
        policy_losses = (-action_log_probabilities * advantage) + (
            self.entropy_coefficient * entropy
        )
        # Compute Value Losses
        value_loss = self.state_value_coefficient * self.loss_function(
            state_current_values, advantage
        )
        # Compute Mean for policy losses
        policy_loss = policy_losses.mean()
        # Compute final loss
        loss = policy_loss + value_loss
        return loss

    @abstractmethod
    def _run_optimizer(self, loss) -> None:
        """
        Protected void method to train the model or accumulate the gradients for training. This method is to be
        overriden.
        """
        pass

    @pytorch.no_grad()
    def _grad_mean_reduction(self) -> None:
        """
        Performs mean reduction and assigns the policy model's parameter the mean reduced gradients.
        """
        reduced_parameters = self._grad_accumulator.mean_reduce()
        # Assign average parameters to model.
        for key, param in self.policy_model.named_parameters():
            if param.requires_grad:
                param.grad = reduced_parameters[key] / self.grad_accumulation_rounds

    def _compute_advantage(
        self, returns: pytorch.Tensor, state_current_values: pytorch.Tensor
    ) -> pytorch.Tensor:
        """
        Computes the advantage from returns and state values
        @param returns: pytorch.Tensor: The discounted returns; computed from _compute_returns method
        @param state_current_values: pytorch.Tensor: The corresponding state values
        @return pytorch.Tensor: The advantage for the given returns and state values
        """
        advantage = returns - state_current_values
        if self._advantage_norm_code in self.apply_norm_to:
            advantage = self._normalization.apply_normalization(advantage)
        return advantage

    def _clear(self) -> None:
        """
        Protected void method to clear the lists of rewards, action_log_probs and state_values.
        """
        #
        self._rollout_buffer.clear()

    def _get_action_sample_shape_for_continuous(self) -> pytorch.Size:
        """
        Gets the action sample shape to be sampled from continuous distribution.
        @return pytorch.Size: Sample shape of to-be sampled tensor from continuous distribution
        """
        if self.is_continuous_action_space:
            assert len(self.action_space) == 2
            if self.action_space[-1] is None:
                return pytorch.Size([])
            return pytorch.Size(self.action_space[-1])
        raise ValueError(
            "`_get_action_sample_shape_for_continuous` must only be called for continuous action spaces"
        )

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
                action_values = [
                    action_value.squeeze(0)
                    if action_value.size(0) == 1
                    else action_value
                    for action_value in action_values
                ]
                distribution = self.distribution(*action_values)
            else:
                raise TypeError(
                    f"Expected `action_values` to be either a tensor or a list of Tensors, "
                    f"received {type(action_values)}"
                )
        return distribution

    def _create_noise_distribution(self) -> pytorch_distributions.Normal:
        """
        Create the standard Gaussian Distribution object for adding noise to sampled actions.
        """
        _mean = pytorch.nn.Parameter(
            pytorch.tensor(0.0, device=self.device, dtype=self.dtype),
            requires_grad=False,
        )
        _std = pytorch.nn.Parameter(
            pytorch.tensor(1.0, device=self.device, dtype=self.dtype),
            requires_grad=False,
        )
        # The PyTorch Normal distribution object initialized with standard mean and std.
        gaussian_noise = pytorch_distributions.Normal(_mean, _std)
        return gaussian_noise

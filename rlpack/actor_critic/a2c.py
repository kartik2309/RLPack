"""!
@package rlpack.actor_critic
@brief This package implements the Actor-Critic methods.


Currently following methods are implemented:
    - `A2C`: Implemented in rlpack.actor_critic.a2c.A2C. More details can be found
     [here](@ref agents/actor_critic/a2c.md)
     - `A3C`: Implemented in rlpack.actor_critic.a3c.A3C. More details can be found
     [here](@ref agents/actor_critic/a3c.md)
"""


import math
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from rlpack import pytorch
from rlpack._C.grad_accumulator import GradAccumulator
from rlpack.utils import Distribution, LossFunction, LRScheduler
from rlpack.utils.base.agent import Agent
from rlpack.utils.internal_code_setup import InternalCodeSetup
from rlpack.utils.normalization import Normalization


class A2C(Agent):
    """
    The A2C class implements the synchronous Actor-Critic method.
    """

    def __init__(
        self,
        policy_model: pytorch.nn.Module,
        optimizer: pytorch.optim.Optimizer,
        lr_scheduler: Union[LRScheduler, None],
        loss_function: LossFunction,
        distribution: Distribution,
        gamma: float,
        entropy_coefficient: float,
        state_value_coefficient: float,
        lr_threshold: float,
        action_space: Union[int, Tuple[int, Union[List[int], None]]],
        backup_frequency: int,
        save_path: str,
        bootstrap_rounds: int = 1,
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
        variance: Optional[
            Tuple[
                Union[float, np.ndarray, pytorch.Tensor],
                Callable[[float, bool, int], float],
            ]
        ] = None,
    ):
        """!
        @param policy_model: *pytorch.nn.Module*: The policy model to be used. Policy model must return a tuple of
            action logits and state values.
        @param optimizer: pytorch.optim.Optimizer: The optimizer to be used for policy model. Optimizer must be
            initialized and wrapped with policy model parameters.
        @param lr_scheduler: Union[LRScheduler, None]: The LR Scheduler to be used to decay the learning rate.
            LR Scheduler must be initialized and wrapped with passed optimizer.
        @param loss_function: LossFunction: A PyTorch loss function.
        @param distribution : dist_math.distribution.Distribution: The distribution of PyTorch to be used to sampled
            actions in action space. (See `action_space`).
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
        @param bootstrap_rounds: int: The number of rounds until which gradients are to be accumulated before
            performing calling optimizer step. Gradients are mean reduced for bootstrap_rounds > 1. Default: 1.
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
        @param variance: Tuple[
                Union[float, np.ndarray, pytorch.Tensor],
                Callable[[Union[float, np.ndarray, pytorch.Tensor], bool, int], float],
            ]: The tuple of variance to be used to sample actions for continuous action space and a method to
            be used to decay it. The passed method have the signature Callable[[float, int], float]. The first
            argument would be the variance value and second value be the boolean, done flag indicating if the state
            is terminal or not and third will be the timestep; returning the updated variance value. Default: None


        **Notes**


        The codes for `apply_norm` are given as follows: -
            - No Normalization: -1; (`"none"`)
            - Min-Max Normalization: 0; (`"min_max"`)
            - Standardization: 1; (`"standardize"`)
            - P-Normalization: 2; (`"p_norm"`)


        The codes for `apply_norm_to` are given as follows:
            - No Normalization: -1; (`["none"]`)
            - On States only: 0; (`["states"]`)
            - On Rewards only: 1; (`["rewards"]`)
            - On TD value only: 2; (`["advantage"]`)
            - On States and Rewards: 3; (`["states", "rewards"]`)
            - On States and TD: 4; (`["states", "advantage"]`)


        If a valid `max_norm_grad` is passed, then gradient clipping takes place else gradient clipping step is
        skipped. If `max_norm_grad` value was invalid, error will be raised from PyTorch.
        If a valid `clip_grad_value` is passed, then gradients will be clipped by value. If `clip_grad_value` value
        was invalid, error will be raised from PyTorch.
        """
        super(A2C, self).__init__()
        setup = InternalCodeSetup()
        device = pytorch.device(device=device)
        dtype = setup.get_torch_dtype(dtype)
        ## The input policy model moved to desired device. @I{# noqa: E266}
        self.policy_model = policy_model.to(device=device, dtype=dtype)
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
        # Check sanity of `bootstrap_rounds`
        assert (
            bootstrap_rounds > 0
        ), "Argument `bootstrap_rounds` must be an integer between 0 and 1"
        ## The input boostrap rounds. @I{# noqa: E266}
        self.bootstrap_rounds = bootstrap_rounds
        ## The input `device` argument; indicating the device name as device type class. @I{# noqa: E266}
        self.device = device
        ## The input `device` argument; indicating the datatype class. @I{# noqa: E266}
        self.dtype = dtype
        if isinstance(apply_norm, str):
            apply_norm = setup.get_apply_norm_mode_code(apply_norm)
        setup.check_validity_of_apply_norm_code(apply_norm)
        ## The input `apply_norm` argument; indicating the normalisation to be used. @I{# noqa: E266}
        self.apply_norm = apply_norm
        if isinstance(apply_norm_to, list):
            apply_norm_to = setup.get_apply_norm_to_mode_code(apply_norm_to)
        setup.check_validity_of_apply_norm_to_code(apply_norm_to)
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
        ## The current variance value. This will be None if `variance` argument was not passed @I{# noqa: E266}
        self.variance_value = None
        ## The variance decay method. This will be None if `variance` argument was not passed @I{# noqa: E266}
        self.variance_decay_fn = None
        ## The boolean flag indicating if variance operations are to be used. @I{# noqa: E266}
        self._operate_with_variance = False
        if variance is not None:
            if len(variance) != 2:
                raise ValueError(
                    "Length of `variance` arg must be 2, "
                    "first argument being the variance value and "
                    "the second being the decay function"
                )
            self.variance_value = variance[0]
            self.variance_decay_fn = variance[1]
            self._operate_with_variance = True
        ## The step counter; counting the total timesteps done so far. @I{# noqa: E266}
        self.step_counter = 0
        ## The list of sampled actions from each timestep from the action distribution. @I{# noqa: E266}
        ## This is cleared after each episode. @I{# noqa: E266}
        self.action_log_probabilities = list()
        ## The list of state values at each timestep.This is cleared after each episode.  @I{# noqa: E266}
        self.states_current_values = list()
        ## The list of rewards from each timestep. This is cleared after each episode. @I{# noqa: E266}
        self.rewards = list()
        ## The list of entropies from each timestep. This is cleared after each episode. @I{# noqa: E266}
        self.entropies = list()
        ## Flag indicating if action space is continuous or discrete. @I{# noqa: E266}
        self.is_continuous_action_space = True
        if isinstance(self.action_space, int):
            self.is_continuous_action_space = False
        # Parameter keys of the model.
        keys = list(dict(self.policy_model.named_parameters()).keys())
        ## The list of gradients from each backward call. @I{# noqa: E266}
        ## This is only used when boostrap_rounds > 1 and is cleared after each boostrap round. @I{# noqa: E266}
        ## The rlpack._C.grad_accumulator.GradAccumulator object for grad accumulation. @I{# noqa: E266}
        self._grad_accumulator = GradAccumulator(keys, bootstrap_rounds)
        ## The normalisation tool to be used for agent. @I{# noqa: E266}
        ## An instance of rlpack.utils.normalization.Normalization. @I{# noqa: E266}
        self._normalization = Normalization(
            apply_norm=apply_norm, eps=eps_for_norm, p=p_for_norm, dim=dim_for_norm
        )

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
        state_current = self._cast_to_tensor(state_current).to(self.device)
        action_values, state_current_value = self.policy_model(state_current)
        distribution = self._create_action_distribution(action_values)
        if self._operate_with_variance:
            self.variance_value = self.variance_decay_fn(
                self.variance_value, done, self.step_counter
            )
        if not self.is_continuous_action_space:
            action = distribution.sample()
        else:
            sample_shape = self._get_action_sample_shape_for_continuous()
            action = distribution.rsample(sample_shape=sample_shape)
        # Accumulate quantities.
        self.action_log_probabilities.append(distribution.log_prob(action))
        self.states_current_values.append(state_current_value)
        self.rewards.append(reward)
        self.entropies.append(distribution.entropy().mean())
        # Call train policy method.
        self._call_to_train_policy_model(done)
        # Backup model every `backup_frequency` steps.
        self._call_to_save()
        # Increment `step_counter` and use policy model to get next action.
        self.step_counter += 1
        with torch.no_grad():
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
        self.policy_model.eval()
        state_current = self._cast_to_tensor(state_current).to(self.device)
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
        save_path = self.save_path
        checkpoint = {
            "policy_model_state_dict": self.policy_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self._operate_with_variance:
            checkpoint["variance_value"] = self.variance_value
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
        if "variance_value" in checkpoint.keys():
            self.variance_value = checkpoint["variance_value"]
            self._operate_with_variance = True
        return

    def _call_to_save(self) -> None:
        """
        Method calling the save method when required. This method is to be overriden by asynchronous methods.
        """
        if (self.step_counter + 1) % self.backup_frequency == 0:
            self.save()
        return

    def _call_to_train_policy_model(self, done: Union[bool, int]) -> None:
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
            loss = self._compute_loss()
            self._run_optimizer(loss)

    def _compute_loss(self) -> pytorch.Tensor:
        """
        Method to compute total loss (from actor and critic).
        @return pytorch.Tensor: The loss tensor.
        """
        self.policy_model.train()
        returns = self._compute_returns()
        # Stack the action log probabilities.
        action_log_probabilities = pytorch.stack(self.action_log_probabilities).to(
            self.device
        )
        # Get entropy values
        entropy = pytorch.tensor(
            self.entropies, dtype=pytorch.float32, device=self.device
        )
        entropy = self._adjust_dims_for_tensor(
            entropy, target_dim=action_log_probabilities.dim()
        )
        # Stack the State values.
        state_current_values = pytorch.stack(self.states_current_values).to(self.device)
        # Compute Advantage Values
        advantage = self._compute_advantage(returns, state_current_values).detach()
        # Adjust dimensions for further calculations
        action_log_probabilities = self._adjust_dims_for_tensor(
            action_log_probabilities, advantage.dim()
        )
        entropy = self._adjust_dims_for_tensor(entropy, advantage.dim())
        # Compute Policy Losses
        policy_losses = (
            -action_log_probabilities * advantage + self.entropy_coefficient * entropy
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

    def _run_optimizer(self, loss) -> None:
        """
        Protected void method to train the model or accumulate the gradients for training.
        - If bootstrap_rounds is passed as 1 (default), model is trained each time the method is called.
        - If bootstrap_rounds > 1, the gradients are accumulated in grad_accumulator and model is trained via
            _train_models method.
        """
        # Clear the buffer values.
        self._clear()
        # Prepare for optimizer step by setting zero grads.
        self.optimizer.zero_grad()
        # Backward call
        loss.backward()
        # Append loss to list
        self.loss.append(loss.item())
        if self.bootstrap_rounds > 1:
            # When `bootstrap_rounds` is greater than 1; accumulate gradients if no. of rounds
            # specified by `bootstrap_rounds` have not been completed and return.
            # If no. of rounds have been completed, perform mean reduction and proceed with optimizer step.
            if len(self._grad_accumulator) < self.bootstrap_rounds:
                self._grad_accumulator.accumulate(self.policy_model.named_parameters())
                return
            else:
                # Perform mean reduction.
                self._grad_mean_reduction()
                # Clear Accumulated Gradient buffer.
                self._grad_accumulator.clear()
        # Clip gradients by norm if requested.
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
        # Take optimizer step.
        self.optimizer.step()
        # Take an LR Scheduler step if required.
        if (
            self.lr_scheduler is not None
            and min([*self.lr_scheduler.get_last_lr()]) > self.lr_threshold
        ):
            self.lr_scheduler.step()

    @pytorch.no_grad()
    def _grad_mean_reduction(self) -> None:
        """
        Performs mean reduction and assigns the policy model's parameter the mean reduced gradients.
        """
        reduced_parameters = self._grad_accumulator.mean_reduce()
        # Assign average parameters to model.
        for key, param in self.policy_model.named_parameters():
            param.grad = reduced_parameters[key] / self.bootstrap_rounds

    def _compute_advantage(
        self, returns: pytorch.Tensor, state_current_values: pytorch.Tensor
    ) -> pytorch.Tensor:
        """
        Computes the advantage from returns and state values
        @param returns: pytorch.Tensor: The discounted returns; computed from _compute_returns method
        @param state_current_values: pytorch.Tensor: The corresponding state values
        @return pytorch.Tensor: The advantage for the given returns and state values
        """
        returns = self._adjust_dims_for_tensor(returns, state_current_values.dim())
        # Apply normalization if required to states.
        if self.apply_norm_to in self._state_norm_codes:
            state_current_values = self._normalization.apply_normalization(
                state_current_values
            )
        # Apply normalization if required to rewards.
        if self.apply_norm_to in self._reward_norm_codes:
            returns = self._normalization.apply_normalization(returns)
        advantage = returns - state_current_values
        if self.apply_norm_to in self._advantage_norm_codes:
            advantage = self._normalization.apply_normalization(advantage)
        return advantage

    def _compute_returns(self) -> pytorch.Tensor:
        """
        Computes the discounted returns iteratively.
        @return pytorch.Tensor: The discounted returns
        """
        total_rewards = len(self.rewards)
        returns = [0] * total_rewards
        r_ = 0
        for idx in range(total_rewards):
            idx = total_rewards - idx - 1
            r_ = self.rewards[idx] + self.gamma * r_
            returns[idx] = r_
        # Convert discounted returns to tensor and move it to correct device and dtype
        returns = pytorch.tensor(returns, dtype=self.dtype, device=self.device)
        return returns

    def _clear(self) -> None:
        """
        Protected void method to clear the lists of rewards, action_log_probs and state_values.
        """
        #
        self.rewards.clear()
        self.action_log_probabilities.clear()
        self.states_current_values.clear()
        self.entropies.clear()

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
    ) -> Distribution:
        """
        Protected static method to create distributions from action logits
        @param action_values: Union[List[pytorch.Tensor], pytorch.Tensor]: The action values from policy model
        @return Distribution: A Distribution object initialized with given action logits
        """
        if not self.is_continuous_action_space:
            distribution = self.distribution(logits=action_values)
        else:
            if isinstance(action_values, pytorch.Tensor):
                action_values_ = action_values.flatten()
                if self._operate_with_variance:
                    action_values_[-1] = self.variance_value**0.5
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
                if self._operate_with_variance:
                    if isinstance(self.variance_value, (float, int)):
                        action_values[-1] = math.sqrt(self.variance_value)
                    elif isinstance(self.variance_value, np.ndarray):
                        action_values[-1] = pytorch.from_numpy(
                            np.sqrt(self.variance_value)
                        ).to(device=self.device, dtype=self.dtype)
                    elif isinstance(self.variance_value, pytorch.Tensor):
                        action_values[-1] = pytorch.sqrt(self.variance_value).to(
                            device=self.device, dtype=self.dtype
                        )
                    else:
                        raise TypeError(
                            f"Invalid datatype passed for `variance` value. Must be either of "
                            f"{float}, {np.ndarray} or {pytorch.Tensor}"
                        )
                distribution = self.distribution(*action_values)
            else:
                raise TypeError(
                    f"Expected `action_values` to be either a tensor or a list of Tensors, "
                    f"received {type(action_values)}"
                )
        return distribution

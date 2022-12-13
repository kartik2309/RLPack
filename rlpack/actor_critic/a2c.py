"""!
@package rlpack.actor_critic
@brief This package implements the Actor-Critic methods.


Currently following methods are implemented:
    - `A2C`: Implemented in rlpack.actor_critic.a2c.A2C. More details can be found
     [here](@ref agents/actor_critic/a2c.md)
"""
import os
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from torch.distributions import Categorical

from rlpack import pytorch
from rlpack.utils import LossFunction, LRScheduler
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
        gamma: float,
        entropy_coefficient: float,
        state_value_coefficient: float,
        lr_threshold: float,
        num_actions: int,
        backup_frequency: int,
        save_path: str,
        bootstrap_rounds: int = 1,
        device: str = "cpu",
        apply_norm: int = -1,
        apply_norm_to: int = -1,
        eps_for_norm: float = 5e-12,
        p_for_norm: int = 2,
        dim_for_norm: int = 0,
        max_grad_norm: Optional[float] = None,
        grad_norm_p: float = 2.0,
    ):
        """!
        @param policy_model: *pytorch.nn.Module*: The policy model to be used. Policy model must return a tuple of
            action logits and state values.
        @param optimizer: pytorch.optim.Optimizer: The optimizer to be used for policy model. Optimizer must be
            initialized and wrapped with policy model parameters.
        @param lr_scheduler: Union[LRScheduler, None]: The LR Scheduler to be used to decay the learning rate.
            LR Scheduler must be initialized and wrapped with passed optimizer.
        @param loss_function: LossFunction: A PyTorch loss function.
        @param gamma: float: The discounting factor for rewards.
        @param entropy_coefficient: float: The coefficient to be used for entropy in policy loss computation.
        @param state_value_coefficient: float: The coefficient to be used for state value in final loss computation.
        @param lr_threshold: float: The threshold LR which once reached LR scheduler is not called further.
        @param num_actions: int: Number of actions for the environment.
        @param backup_frequency: int: The timesteps after which policy model, optimizer states and lr
            scheduler states are backed up.
        @param save_path: str: The path where policy model, optimizer states and lr scheduler states are to be saved.
        @param bootstrap_rounds: int: The number of rounds until which gradients are to be accumulated before
            performing calling optimizer step. Gradients are mean reduced for bootstrap_rounds > 1. Default: 1.
        @param device: str: The device on which models are run. Default: "cpu".
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
        @param grad_norm_p: Optional[float]: The p-value for p-normalization of gradients. Default: 2.0



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
            - On TD value only: 2; (`["td"]`)
            - On States and Rewards: 3; (`["states", "rewards"]`)
            - On States and TD: 4; (`["states", "td"]`)


        If a valid `max_norm_grad` is passed, then gradient clipping takes place else gradient clipping step is
        skipped. If `max_norm_grad` value was invalid, error will be raised from PyTorch.
        """
        super(A2C, self).__init__()
        setup = InternalCodeSetup()
        ## The input policy model moved to desired device. @I{# noqa: E266}
        self.policy_model = policy_model.to(device)
        ## The input optimizer wrapped with policy_model parameters. @I{# noqa: E266}
        self.optimizer = optimizer
        ## The input optional LR Scheduler (this can be None). @I{# noqa: E266}
        self.lr_scheduler = lr_scheduler
        ## The input loss function. @I{# noqa: E266}
        self.loss_function = loss_function
        ## The input discounting factor. @I{# noqa: E266}
        self.gamma = gamma
        ## The input entropy coefficient. @I{# noqa: E266}
        self.entropy_coefficient = entropy_coefficient
        ## The input state value coefficient. @I{# noqa: E266}
        self.state_value_coefficient = state_value_coefficient
        ## The input LR Threshold. @I{# noqa: E266}
        self.lr_threshold = float(lr_threshold)
        ## The input number of actions. @I{# noqa: E266}
        self.num_actions = num_actions
        ## The input model backup frequency in terms of timesteps. @I{# noqa: E266}
        self.backup_frequency = backup_frequency
        ## The input save path for backing up agent models. @I{# noqa: E266}
        self.save_path = save_path
        ## The input boostrap rounds. @I{# noqa: E266}
        self.bootstrap_rounds = bootstrap_rounds
        ## The input `device` argument; indicating the device name. @I{# noqa: E266}
        self.device = device
        if isinstance(apply_norm, str):
            apply_norm = setup.get_apply_norm_mode_code(apply_norm)
        ## The input `apply_norm` argument; indicating the normalisation to be used. @I{# noqa: E266}
        self.apply_norm = apply_norm
        if isinstance(apply_norm_to, (str, list)):
            apply_norm_to = setup.get_apply_norm_to_mode_code(apply_norm_to)
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
        ## The step counter; counting the total timesteps done so far. @I{# noqa: E266}
        self.step_counter = 0
        ## The episode counter; counting the total episodes done so far. @I{# noqa: E266}
        self.episode_counter = 1
        ## The list of sampled actions from each timestep from the action distribution. @I{# noqa: E266}
        ## This is cleared after each episode. @I{# noqa: E266}
        self.action_log_probabilities = list()
        ## The list of state values at each timestep.This is cleared after each episode.  @I{# noqa: E266}
        self.states_current_values = list()
        ## The list of rewards from each timestep. This is cleared after each episode. @I{# noqa: E266}
        self.rewards = list()
        ## The list of entropies from each timestep. This is cleared after each episode. @I{# noqa: E266}
        self.entropies = list()
        ## The list of gradients from each backward call. @I{# noqa: E266}
        ## This is only used when boostrap_rounds > 1 and is cleared after each boostrap round. @I{# noqa: E266}
        self.grad_accumulator = list()
        ## The normalisation tool to be used for agent. @I{# noqa: E266}
        ## An instance of rlpack.utils.normalization.Normalization. @I{# noqa: E266}
        self.__normalization = Normalization(
            apply_norm=apply_norm, eps=eps_for_norm, p=p_for_norm, dim=dim_for_norm
        )
        ## The policy model parameters names. @I{# noqa: E266}
        self.__policy_model_parameter_keys = OrderedDict(
            self.policy_model.named_parameters()
        ).keys()

    def train(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        reward: Union[int, float],
        done: Union[bool, int],
        **kwargs,
    ) -> int:
        """
        The train method to train the agent and underlying policy model.
        @param state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current state returned
        @param reward: Union[int, float]: The reward returned from previous action
        @param done: Union[bool, int]: Flag indicating if episode has terminated or not
        @param kwargs: Other keyword arguments.
        @return int: The action to be taken
        """
        self.policy_model.eval()
        state_current = self._cast_to_tensor(state_current).to(self.device)
        actions_logits, state_current_value = self.policy_model(state_current)
        distribution = self.__create_action_distribution(actions_logits)
        action = distribution.sample()
        self.action_log_probabilities.append(distribution.log_prob(action))
        self.states_current_values.append(state_current_value)
        self.rewards.append(reward)
        self.entropies.append(distribution.entropy().mean())
        self.__call_train_policy_model(done)
        return action.item()

    @pytorch.no_grad()
    def policy(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        **kwargs,
    ) -> int:
        """
        The policy method to evaluate the agent. This runs in pure inference mode.
        @param state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current state returned
            from gym environment
        @param kwargs: Other keyword arguments
        @return int: The action to be taken
        """
        self.policy_model.eval()
        state_current = self._cast_to_tensor(state_current).to(self.device)
        actions_logits, _ = self.policy_model(state_current)
        distribution = self.__create_action_distribution(actions_logits)
        action = distribution.sample().item()
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
        checkpoint_policy = {"state_dict": self.policy_model.state_dict()}
        checkpoint_optimizer = {"state_dict": self.optimizer.state_dict()}
        pytorch.save(
            checkpoint_policy,
            os.path.join(self.save_path, f"policy{custom_name_suffix}.pt"),
        )
        pytorch.save(
            checkpoint_optimizer,
            os.path.join(self.save_path, f"optimizer{custom_name_suffix}.pt"),
        )
        if self.lr_scheduler is not None:
            checkpoint_lr_scheduler = {"state_dict": self.lr_scheduler.state_dict()}
            pytorch.save(
                checkpoint_lr_scheduler,
                os.path.join(self.save_path, f"lr_scheduler{custom_name_suffix}.pt"),
            )
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
        if os.path.isfile(
            os.path.join(self.save_path, f"policy{custom_name_suffix}.pt")
        ):
            checkpoint_policy = pytorch.load(
                os.path.join(self.save_path, f"policy{custom_name_suffix}.pt"),
                map_location="cpu",
            )
            self.policy_model.load_state_dict(checkpoint_policy["state_dict"])
        else:
            raise FileNotFoundError("The Policy model was not found in the given path!")
        if os.path.isfile(
            os.path.join(self.save_path, f"optimizer{custom_name_suffix}.pt")
        ):
            checkpoint_optimizer = pytorch.load(
                os.path.join(self.save_path, f"optimizer{custom_name_suffix}.pt"),
                map_location="cpu",
            )
            self.optimizer.load_state_dict(checkpoint_optimizer["state_dict"])
        if os.path.isfile(
            os.path.join(self.save_path, f"lr_scheduler{custom_name_suffix}.pt")
        ):
            checkpoint_lr_sc = pytorch.load(
                os.path.join(self.save_path, f"lr_scheduler{custom_name_suffix}.pt"),
                map_location="cpu",
            )
            self.lr_scheduler.load_state_dict(checkpoint_lr_sc["state_dict"])
        return

    def __call_train_policy_model(self, done: Union[bool, int]) -> None:
        """
        Private method to call the appropriate method for training policy model based on initialization of A2C agent
        @param done: Union[bool, int]: Flag indicating if episode has terminated or not
        """
        if isinstance(done, bool):
            if done:
                self.__accumulate_gradients()
                self.episode_counter += 1
        elif isinstance(done, int) and done == 1:
            if done == 1:
                self.__accumulate_gradients()
                self.episode_counter += 1
        else:
            raise TypeError(
                f"Expected `done` argument to be of type {bool} or {int} but received {type(done)}!"
            )
        if (
            self.episode_counter % (self.bootstrap_rounds + 1) == 0
            and self.bootstrap_rounds > 1
        ):
            self.__train_models()
            self.episode_counter += 1

    def __accumulate_gradients(self) -> None:
        """
        Private void method to train the model or accumulate the gradients for training.
        - If bootstrap_rounds is passed as 1 (default), model is trained each time the method is called.
        - If bootstrap_rounds > 1, the gradients are accumulated in grad_accumulator and model is trained via
            __train_models method.
        """
        self.policy_model.train()
        returns = self.__compute_returns()
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
        # Compute TD Values
        advantage = self.compute_advantage(returns, state_current_values).detach()
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
        if self.bootstrap_rounds < 2:
            self.optimizer.zero_grad()
        # Backward call
        loss.backward()
        # Append loss to list
        self.loss.append(loss.item())
        if self.bootstrap_rounds < 2:
            # Take optimizer step.
            self.optimizer.step()
            # Clip gradients if requested.
            if self.max_grad_norm is not None:
                pytorch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    max_norm=self.max_grad_norm,
                    norm_type=self.grad_norm_p,
                )
        else:
            self.grad_accumulator.append(
                {
                    k: param.grad.detach().clone()
                    for k, param in self.policy_model.named_parameters()
                },
            )
        self.__clear()

    def __train_models(self) -> None:
        """
        Private method to policy model if boostrap_rounds > 1. In such cases the gradients are accumulated in
        grad_accumulator. This method collects the accumulated gradients and performs mean reduction and runs
        optimizer step.
        """
        policy_model_grads = self.grad_accumulator
        # OrderedDict to store reduced average value.
        policy_model_grads_reduced = OrderedDict()
        self.optimizer.zero_grad()
        # No Grad mode to disable PyTorch Operation tracking.
        with pytorch.no_grad():
            # Perform parameter wise summation.
            for key in self.__policy_model_parameter_keys:
                for policy_model_grad in policy_model_grads:
                    if key not in policy_model_grads_reduced.keys():
                        policy_model_grads_reduced[key] = policy_model_grad[key]
                        continue
                    policy_model_grads_reduced[key] += policy_model_grad[key]
            # Assign average parameters to model.
            for key, param in self.policy_model.named_parameters():
                param.grad = policy_model_grads_reduced[key] / self.bootstrap_rounds
        # Take an optimizer step.
        self.optimizer.step()
        # Clip gradients if requested.
        if self.max_grad_norm is not None:
            pytorch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=self.grad_norm_p,
            )
        # Take an LR Scheduler step if required.
        if (
            self.lr_scheduler is not None
            and min([*self.lr_scheduler.get_last_lr()]) > self.lr_threshold
        ):
            self.lr_scheduler.step()
        # Clear buffers for tracked operations.
        self.__clear()
        # Clear Accumulated Gradient buffer.
        self.grad_accumulator.clear()

    def compute_advantage(
        self, returns: pytorch.Tensor, state_current_values: pytorch.Tensor
    ) -> pytorch.Tensor:
        """
        Computes the advantage from returns and state values
        @param returns: pytorch.Tensor: The discounted returns; computed from __compute_returns method
        @param state_current_values: pytorch.Tensor: The corresponding state values
        @return pytorch.Tensor: The advantage for the given returns and state values
        """
        returns = self._adjust_dims_for_tensor(returns, state_current_values.dim())
        # Apply normalization if required to states.
        if self.apply_norm_to in self.state_norm_codes:
            state_current_values = self.__normalization.apply_normalization(
                state_current_values
            )
        # Apply normalization if required to rewards.
        if self.apply_norm_to in self.reward_norm_codes:
            returns = self.__normalization.apply_normalization(returns)
        advantage = returns - state_current_values
        if self.apply_norm_to in self.advantage_norm_codes:
            advantage = self.__normalization.apply_normalization(advantage)
        return advantage

    def __compute_returns(self) -> pytorch.Tensor:
        """
        Computes the discounted returns iteratively.
        @return pytorch.Tensor: The discounted returns
        """
        rewards_in_descending_timesteps = self.rewards[::-1]
        returns = list()
        r_ = 0
        # Compute discounted returns
        for r in rewards_in_descending_timesteps:
            r_ = r + self.gamma * r_
            returns.insert(0, r_)
        # Convert discounted returns to tensor and move it to correct device.
        returns = pytorch.tensor(returns, dtype=pytorch.float32, device=self.device)
        return returns

    def __clear(self) -> None:
        """
        Private void method to clear the lists of rewards, action_log_probs and state_values.
        """
        #
        self.rewards.clear()
        self.action_log_probabilities.clear()
        self.states_current_values.clear()
        self.entropies.clear()
        self.loss.clear()

    @staticmethod
    def __create_action_distribution(
        actions_logits: pytorch.Tensor,
    ) -> Categorical:
        """
        Private static method to create distributions from action logits
        @param actions_logits: pytorch.Tensor: The action logits from policy model
        @return Categorical: A Categorical object initialized with given action logits
        """
        categorical = Categorical(logits=actions_logits)
        return categorical

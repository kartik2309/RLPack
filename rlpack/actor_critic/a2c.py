"""!
@package actor_critic
@brief This package implements the Actor-Critic methods.


Currently following methods are implemented:
    - A2C: Implemented in rlpack.actor_critic.a2c.A2C. More details can be found [here](@ref agents/a2c.md)
"""

from collections import OrderedDict
from typing import List, Union

import numpy as np
from torch.distributions import Categorical

from rlpack import pytorch
from rlpack.utils import LossFunction, LRScheduler
from rlpack.utils.base.agent import Agent
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
            model_backup_frequency: int,
            save_path: str,
            bootstrap_rounds: int = 1,
            device: str = "cpu",
            apply_norm: int = -1,
            apply_norm_to: int = -1,
            eps_for_norm: float = 5e-12,
            p_for_norm: int = 2,
            dim_for_norm: int = 0,
    ):
        """
        :param policy_model: pytorch.nn.Module: The policy model to be used. Policy model must return a tuple of
            action logits and state values.
        :param optimizer: pytorch.optim.Optimizer: The optimizer to be used for policy model. Optimizer must be
            initialized and wrapped with policy model parameters.
        :param lr_scheduler: Union[LRScheduler, None]: The LR Scheduler to be used to decay the learning rate.
            LR Scheduler must be initialized and wrapped with passed optimizer.
        :param loss_function: LossFunction: A PyTorch loss function.
        :param gamma: float: The discounting factor for rewards.
        :param entropy_coefficient: float: The coefficient to be used for entropy in policy loss computation.
        :param state_value_coefficient: float: The coefficient to be used for state value in final loss computation.
        :param lr_threshold: float: The threshold LR which once reached LR scheduler is not called further.
        :param num_actions: int: Number of actions for the environment.
        :param model_backup_frequency: int: The timesteps after which policy model, optimizer states and lr
            scheduler states are backed up.
        :param save_path: str: The path where policy model, optimizer states and lr scheduler states are to be saved.
        :param bootstrap_rounds: int: The number of rounds until which gradients are to be accumulated before
            performing calling optimizer step. Gradients are mean reduced for bootstrap_rounds > 1. Default: 1.
        :param device: str: The device on which models are run. Default: "cpu".
        :param apply_norm: int: The code to select the normalization procedure to be applied on selected quantities;
            selected by `apply_norm_to`: see below)). Default: -1.
        :param apply_norm_to: int: The code to select the quantity to which normalization is to be applied.
            Default: -1.
        :param eps_for_norm: float: Epsilon value for normalization; for numeric stability. For min-max normalization
            and standardized normalization. Default: 5e-12.
        :param p_for_norm: int: The p value for p-normalization. Default: 2; L2 Norm.
        :param dim_for_norm: int: The dimension across which normalization is to be performed. Default: 0.

        The codes for `apply_norm` are given as follows: -
            - No Normalization: -1
            - Min-Max Normalization: 0
            - Standardization: 1
            - P-Normalization: 2
        The codes for `apply_norm_to` are given as follows:
            No Normalization: -1
            On States only: 0
            On Rewards only: 1
            On Advantage value only: 2
            On States and Rewards: 3
            On States and Advantage: 4
        """
        super(A2C, self).__init__()
        self.policy_model = policy_model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.state_value_coefficient = state_value_coefficient
        self.lr_threshold = float(lr_threshold)
        self.num_actions = num_actions
        self.model_backup_frequency = model_backup_frequency
        self.save_path = save_path
        self.bootstrap_rounds = bootstrap_rounds
        self.device = device
        self.apply_norm = apply_norm
        self.apply_norm_to = apply_norm_to
        self.eps_for_norm = eps_for_norm
        self.p_for_norm = p_for_norm
        self.dim_for_norm = dim_for_norm
        # Initialize counters
        self.step_counter = 0
        self.episode_counter = 1
        # List to save the tuple for Log-Probabilities
        self.action_log_probabilities = list()
        # List to save the tuple State Values.
        self.states_current_values = list()
        # List to save rewards locally
        self.rewards = list()
        # Entropy Accumulation list
        self.entropies = list()
        # Gradient Accumulation list
        self.grad_accumulator = list()
        # Initialize Normalization tool.
        self.normalization = Normalization(
            apply_norm=apply_norm, eps=eps_for_norm, p=p_for_norm, dim=dim_for_norm
        )
        # Keys for named parameters of policy model.
        self.policy_model_parameter_keys = OrderedDict(
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
        :param state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current state returned
        :param reward: Union[int, float]: The reward returned from previous action
        :param done: Union[bool, int]: Flag indicating if episode has terminated or not
        :param kwargs: Other keyword arguments.
        :return: int: The action to be taken
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
        :param state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current state returned
            from gym environment
        :param kwargs: Other keyword arguments
        :return: int: The action to be taken
        """
        self.policy_model.eval()
        state_current = self._cast_to_tensor(state_current).to(self.device)
        actions_logits, _ = self.policy_model(state_current)
        distribution = self.__create_action_distribution(actions_logits)
        action = distribution.sample().item()
        return action

    def __call_train_policy_model(self, done: Union[bool, int]) -> None:
        """
        Private method to call the appropriate method for training policy model based on initialization of A2C agent
        :param done: Union[bool, int]: Flag indicating if episode has terminated or not
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
        if self.bootstrap_rounds < 2:
            self.optimizer.step()
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
            for key in self.policy_model_parameter_keys:
                for policy_model_grad in policy_model_grads:
                    if key not in policy_model_grads_reduced.keys():
                        policy_model_grads_reduced[key] = policy_model_grad[key]
                        continue
                    policy_model_grads_reduced[key] += policy_model_grad[key]
            # Assign average parameters to model
            for key, param in self.policy_model.named_parameters():
                param.grad = policy_model_grads_reduced[key] / self.bootstrap_rounds
        # Take an optimizer step
        self.optimizer.step()
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
        :param returns: pytorch.Tensor: The discounted returns; computed from __compute_returns method
        :param state_current_values: pytorch.Tensor: The corresponding state values
        :return: pytorch.Tensor: The advantage for the given returns and state values
        """
        returns = self._adjust_dims_for_tensor(returns, state_current_values.dim())
        # Apply normalization if required to states.
        if self.apply_norm_to in self.state_norm_codes:
            state_current_values = self.normalization.apply_normalization(
                state_current_values
            )
        # Apply normalization if required to rewards.
        if self.apply_norm_to in self.reward_norm_codes:
            returns = self.normalization.apply_normalization(returns)
        advantage = returns - state_current_values
        if self.apply_norm_to in self.advantage_norm_codes:
            advantage = self.normalization.apply_normalization(advantage)
        return advantage

    def __compute_returns(self) -> pytorch.Tensor:
        """
        Computes the discounted returns iteratively.
        :return: pytorch.Tensor: The discounted returns
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

    @staticmethod
    def __create_action_distribution(
            actions_logits: pytorch.Tensor,
    ) -> Categorical:
        """
        Private static method to create distributions from action logits
        :param actions_logits: pytorch.Tensor: The action logits from policy model
        :return: Categorical: A Categorical object initialized with given action logits
        """
        categorical = Categorical(logits=actions_logits)
        return categorical

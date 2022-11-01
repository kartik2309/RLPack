import logging
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy import ndarray

from rlpack import pytorch
from rlpack._C.memory import Memory
from rlpack.utils.base.agent import Agent
from rlpack.utils.normalization import Normalization

LRScheduler = TypeVar("LRScheduler")
LossFunction = TypeVar("LossFunction")
Activation = TypeVar("Activation")


class DqnAgent(Agent):
    """
    The DqnAgent class which implements the DQN algorithm on arguments. This class inherits from `Agent`
        class, which is the generic base class for all the agents in the project.
    """

    def __init__(
        self,
        target_model: pytorch.nn.Module,
        policy_model: pytorch.nn.Module,
        optimizer: pytorch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        loss_function: LossFunction,
        gamma: float,
        epsilon: float,
        min_epsilon: float,
        epsilon_decay_rate: float,
        epsilon_decay_frequency: int,
        memory_buffer_size: int,
        target_model_update_rate: int,
        policy_model_update_rate: int,
        model_backup_frequency: int,
        lr_threshold: float,
        batch_size: int,
        num_actions: int,
        save_path: str,
        device: str = "cpu",
        prioritization_params: Optional[Dict[str, Any]] = None,
        force_terminal_state_selection_prob: float = 0.0,
        tau: float = 1.0,
        apply_norm: int = -1,
        apply_norm_to: int = -1,
        eps_for_norm: float = 5e-12,
        p_for_norm: int = 2,
        dim_for_norm: int = 0,
    ):
        """
        @:param target_model (nn.Module): The target network for DQN model. This the network which has
            its weights frozen
        @:param policy_model (nn.Module): The policy network for DQN model. This is the network which is trained.
        @:param optimizer (optim.Optimizer): The optimizer wrapped with policy model's parameters.
        @:param lr_scheduler (LRScheduler): The PyTorch LR Scheduler with wrapped optimizer.
        @:param loss_function (LossFunction): The loss function from PyTorch's nn module. Initialized
            instance must be passed.
        @:param gamma (float): The gamma value for agent.
        @:param epsilon (float): The initial epsilon for the agent.
        @:param min_epsilon (float): The minimum epsilon for the agent. Once this value is reached,
            it is maintained for all further episodes.
        @:param epsilon_decay_rate (float): The decay multiplier to decay the epsilon.
        @:param epsilon_decay_frequency (int): The number of timesteps after which the epsilon is decayed.
        @:param memory_buffer_size (int): The buffer size of memory (or replay buffer) for DQN.
        @:param target_model_update_rate (int): The timesteps after which target model's weights are updated with
            policy model weights (weights are weighted as per `tau` (see below)).
        @:param policy_model_update_rate (int): The timesteps after which policy model is trained. This involves
            backpropagation through the policy network.
        @:param model_backup_frequency (int): The timesteps after which models are backed up. This will also
            save optimizer, lr_scheduler and agent_states (epsilon the time of saving and memory).
        @:param lr_threshold (float): The threshold LR which once reached LR scheduler is not called further.
        @:param batch_size (int): The batch size used for inference through target_model and train through policy model.
        @:param num_actions (int): Number of actions for the environment.
        @:param save_path (str): The save path for models (target_model and policy_model), optimizer,
            lr_scheduler and agent_states.
        @:param device (str): The device on which models are run. Default: "cpu"
        @:param prioritization_params (Optional[Dict[str, Any]]): The parameters for prioritization in prioritized
            memory (or relay buffer). Default: None
        @:param force_terminal_state_selection_prob (float): The probability for forcefully selecting a terminal state
            in a batch. Default: 0.0
        @:param tau (float): The weighted update of weights from policy_model to target_model. This is done by formula
            target_weight = tau * policy_weight + (1 - tau) * target_weight/. Default: -1
        @:param apply_norm (int): The code to select the normalization procedure to be applied on selected quantities
            (selected by `apply_norm_to` (see below)). Default: -1
        @:param apply_norm_to (int): The code to select the quantity to which normalization is to be applied.
            Default: -1
        @:param eps_for_norm (int): Epsilon value for normalization (for numeric stability). For min-max normalization
            and standardized normalization. Default: 5e-12
        @:param p_for_norm (int): The p value for p-normalization. Default: 2 (L2 Norm)
        @:param dim_for_norm (int): The dimension across which normalization is to be performed. Default: 0.

        NOTE:
        For prioritization_params, when None (the default) is passed, prioritized memory is not used. To use
            prioritized memory, pass a dictionary with keys `alpha` and `beta`. You can also pass `alpha_decay_rate`
            and `beta_decay_rate` additionally.
        The codes for `apply_norm` are given as follows: -
            - No Normalization: -1
            - Min-Max Normalization: 0
            - Standardization: 1
            - P-Normalization: 2
        The codes for `apply_norm_to` are given as follows:
            No Normalization: -1
            On States only: 0
            On Rewards only: 1
            On TD value only: 2
            On States and Rewards: 3
            On States and TD: 4
        """
        super(DqnAgent, self).__init__()
        self.target_model = target_model.to(device)
        self.policy_model = policy_model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_frequency = epsilon_decay_frequency
        self.memory_buffer_size = memory_buffer_size
        self.target_model_update_rate = target_model_update_rate
        self.policy_model_update_rate = policy_model_update_rate
        self.model_backup_frequency = model_backup_frequency
        self.min_lr = float(lr_threshold)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.save_path = save_path
        self.device = device
        # Process `prioritization_params` and set flags accordingly.
        self.prioritization_strategy_code = (
            prioritization_params.get("prioritization_strategy", 0)
            if prioritization_params is not None
            else 0
        )
        self.prioritization_params = prioritization_params
        self.force_terminal_state_selection_prob = force_terminal_state_selection_prob
        # Sanity check for tau value.
        if tau < 0 or tau > 1:
            ValueError(
                "Invalid value for tau passed! Expected value is between 0 and 1"
            )
        self.tau = float(tau)
        self.apply_norm = apply_norm
        self.apply_norm_to = apply_norm_to
        self.eps_for_norm = eps_for_norm
        self.p_for_norm = p_for_norm
        self.dim_for_norm = dim_for_norm
        self.step_counter = 1
        # Set necessary prioritization parameters. Depending on `prioritization_params`, appropriate values are set.
        self.prioritization_params = self.__process_prioritization_params(
            self.prioritization_params
        )
        # Initialize Memory.
        self.memory = Memory(
            buffer_size=memory_buffer_size,
            device=device,
            prioritization_strategy_code=self.prioritization_strategy_code,
        )
        # Disable gradients for target network.
        for n, p in self.target_model.named_parameters():
            p.requires_grad = False
        # Initialize Normalization tool.
        self.normalization = Normalization(apply_norm=apply_norm)

    def train(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        state_next: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        reward: Union[int, float],
        action: Union[int, float],
        done: Union[bool, int],
        priority: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1e3,
        probability: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1.0,
        weight: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1.0,
    ) -> int:
        """
        - The training method for agent, which accepts a transition from environment and returns an action for next
            transition. Use this method when you intend to train the agent.
        - This method will also run the policy to yield the best action for the given state.
        - For each transition (or experience) being passed, associated priority, probability and weight
            can be passed.

        @:param state_current (Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]): The current
            state in the environment.
        @:param state_next (Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]): The next
            state returned by the environment.
        @:param reward (Union[int, float]): Reward obtained by performing the action for the transition.
        @:param action Union[int, float]: Action taken for the transition
        @:param done Union[bool, int]: Indicates weather episode has terminated or not.
         @:param priority (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The priority of the
            transition (for priority relay memory). Default: 1e3
        @:param probability (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The probability of the transition
            (for priority relay memory). Default: 1.0
        @:param weight (Optional[Union[pytorch.Tensor, np.ndarray, float]]): The important sampling weight
            of the transition (for priority relay memory). Default: 1.0
        @:return (int): The next action to be taken from `state_next`.
        """
        # Insert the sample into memory.
        self.memory.insert(
            state_current,
            state_next,
            reward,
            action,
            done,
            priority,
            probability,
            weight,
        )
        # Train policy model every at every `policy_model_update_rate` steps.
        if (
            self.step_counter % self.policy_model_update_rate == 0
            and len(self.memory) >= self.batch_size
        ):
            self.__train_policy_model()
        # Update target model every `target_model_update_rate` steps.
        if self.step_counter % self.target_model_update_rate == 0:
            self.__update_target_model()
        # Decay epsilon every `epsilon_decay_frequency` steps.
        if self.step_counter % self.epsilon_decay_frequency == 0:
            self.__decay_epsilon()
        # Backup model every `model_backup_frequency` steps.
        if self.step_counter % self.model_backup_frequency == 0:
            self.save()
        # Restart `step_counter` if `it has reached the buffer size.
        if self.step_counter == self.memory_buffer_size:
            self.step_counter = 0
        # If using prioritized memory, anneal alpha and beta.
        if self.prioritization_strategy_code > 0:
            self.__anneal_alpha()
            self.__anneal_beta()
        # Increment `step_counter` and use policy model to get next action.
        self.step_counter += 1
        action = self.policy(state_current)
        return action

    @pytorch.no_grad()
    def policy(self, state_current: Union[ndarray, pytorch.Tensor, List[float]]) -> int:
        """
        The policy for the agent. This runs the inference on policy model with `state_current`
        and uses q-values to obtain the best action.
        @:param state_current (Union[ndarray, pytorch.Tensor, List[float]]): The current state agent is in.
        @:return (int): The action to be taken.
        """
        state_current = self._cast_to_tensor(state_current).to(self.device)
        state_current = pytorch.unsqueeze(state_current, 0)
        p = random.random()
        if p < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            if self.policy_model.training:
                self.policy_model.eval()
            if self.apply_norm_to in self.state_norm_codes:
                state_current = self.normalization.apply_normalization(
                    state_current, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
                )
            q_values = self.policy_model(state_current)
            action_tensor = q_values.argmax(-1)
            action = action_tensor.item()
        return action

    def save(self, custom_name_suffix: Optional[str] = None) -> None:
        """
        This method saves the target_model, policy_model, optimizer, lr_scheduler and agent_states in the supplied
            `save_path` argument in the DQN Agent class' constructor (also called __init__).
        agent_states includes current memory and epsilon values in a dictionary.
        @:param custom_name_suffix Optional[str]: If supplied, additional suffix is added to names of target_model,
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
        checkpoint_target = {
            "state_dict": self.target_model.state_dict(),
        }
        checkpoint_policy = {"state_dict": self.policy_model.state_dict()}
        checkpoint_optimizer = {"state_dict": self.optimizer.state_dict()}

        checkpoint_lr_scheduler = dict()
        if self.lr_scheduler is not None:
            checkpoint_lr_scheduler = {"state_dict": self.lr_scheduler.state_dict()}

        save_memory = True if os.getenv("SAVE_MEMORY", False) == "TRUE" else False
        agent_state = {
            "epsilon": self.epsilon,
            "memory": self.memory.view() if save_memory else None,
        }
        pytorch.save(
            checkpoint_target,
            os.path.join(self.save_path, f"target{custom_name_suffix}.pt"),
        )
        pytorch.save(
            checkpoint_policy,
            os.path.join(self.save_path, f"policy{custom_name_suffix}.pt"),
        )
        pytorch.save(
            checkpoint_optimizer,
            os.path.join(self.save_path, f"optimizer{custom_name_suffix}.pt"),
        )
        if self.lr_scheduler is not None:
            pytorch.save(
                checkpoint_lr_scheduler,
                os.path.join(self.save_path, f"lr_scheduler{custom_name_suffix}.pt"),
            )
        pytorch.save(agent_state, os.path.join(self.save_path, "agent_states.pt"))
        return

    def load(self, custom_name_suffix: Optional[str] = None) -> None:
        """
        This method loads the target_model, policy_model, optimizer, lr_scheduler and agent_states from
            the supplied `save_path` argument in the DQN Agent class' constructor (also called __init__).
        @:param custom_name_suffix Optional[str]: If supplied, additional suffix is added to names of target_model,
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
            os.path.join(self.save_path, f"target{custom_name_suffix}.pt")
        ):
            checkpoint_target = pytorch.load(
                os.path.join(self.save_path, f"target{custom_name_suffix}.pt"),
                map_location="cpu",
            )
            self.target_model.load_state_dict(checkpoint_target["state_dict"])
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
        if os.path.isfile(os.path.join(self.save_path, "agent_state.pt")):
            agent_state = pytorch.load(os.path.join(self.save_path, "agent_state.pt"))
            self.epsilon = agent_state["epsilon"]
            self.memory = agent_state["memory"]
        return

    def __train_policy_model(self) -> None:
        """
        Protected method of the class to train the policy model. This method is called every
            `policy_model_update_rate` timesteps supplied in the DqnAgent class constructor.
        This method will load the random samples from memory (number of samples depend on
            `batch_size` supplied in DqnAgent constructor), and train the policy_model.
        """
        random_experiences = self.__load_random_experiences()
        (
            state_current,
            state_next,
            rewards,
            actions,
            dones,
            priorities,
            probabilities,
            weights,
            random_indices,
        ) = random_experiences
        # Apply normalization if required to states.
        if self.apply_norm_to in self.state_norm_codes:
            state_current = self.normalization.apply_normalization(
                state_current, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
            )
            state_next = self.normalization.apply_normalization(
                state_next, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
            )
        # Apply normalization if required to rewards.
        if self.apply_norm_to in self.reward_norm_codes:
            rewards = self.normalization.apply_normalization(
                rewards, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
            )
        # Set policy model to training mode.
        if not self.policy_model.training:
            self.policy_model.train()
        # Compute target q-values from target model and temporal difference values.
        with pytorch.no_grad():
            q_values_target = self.target_model(state_next)
            td_value = self.temporal_difference(rewards, q_values_target, dones)
        # Apply normalization if required to TD values.
        if self.apply_norm_to in self.td_norm_codes:
            td_value = self.normalization.apply_normalization(
                td_value, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
            )
        # Compute current q-values from policy model.
        q_values_policy = self.policy_model(state_current)
        actions = self._adjust_dims_for_tensor(
            tensor=actions, target_dim=q_values_target.dim()
        )
        # Choose best q-values corresponding to actions
        q_values_gathered = pytorch.gather(q_values_policy, dim=-1, index=actions)
        self.optimizer.zero_grad()
        # If prioritized memory is used, multiply weights from sampling process to TD values.
        if self.prioritization_strategy_code > 0:
            weights_ = self._adjust_dims_for_tensor(weights, td_value.dim())
            td_value = td_value * weights_
        td_value = td_value.detach()
        loss = self.loss_function(q_values_gathered, td_value)
        loss.backward()
        self.loss.append(loss.item())
        if self.prioritization_strategy_code > 0:
            # Proportional-Based prioritization.
            if self.prioritization_strategy_code == 1:
                new_priorities = (
                    pytorch.abs(td_value.cpu()) + self.prioritization_params["error"]
                )
            # Rank-Based prioritization.
            elif self.prioritization_strategy_code == 2:
                _, sorted_td_indices = pytorch.sort(
                    pytorch.abs(td_value.cpu()),
                    dim=0,
                    descending=True
                )
                new_priorities = 1 / (sorted_td_indices + 1)
            else:
                raise NotImplementedError(
                    f"The given prioritization strategy {self.prioritization_strategy_code} has not been implemented"
                )
            self.memory.update_priorities(
                random_indices,
                new_priorities,
                probabilities,
                weights,
            )
        self.optimizer.step()
        if (
            self.lr_scheduler is not None
            and min(self.lr_scheduler.get_last_lr()) > self.min_lr
        ):
            self.lr_scheduler.step()

    @pytorch.no_grad()
    def __update_target_model(self) -> None:
        """
        Protected method of the class to update the target model. This method is called every
            `target_model_update_rate` timesteps supplied in the DqnAgent class constructor.
        """
        policy_parameters = self.policy_model.named_parameters()
        target_parameters = self.target_model.named_parameters()
        target_parameters = OrderedDict(target_parameters)
        for policy_parameter_name, policy_parameter in policy_parameters:
            if policy_parameter_name in target_parameters:
                target_parameters[policy_parameter_name].copy_(
                    (1 - self.tau) * target_parameters[policy_parameter_name]
                    + self.tau * policy_parameter
                )
        return

    @pytorch.no_grad()
    def temporal_difference(
        self, rewards: pytorch.Tensor, q_values: pytorch.Tensor, dones: pytorch.Tensor
    ) -> pytorch.Tensor:
        """
        This method computes the temporal difference for given transitions.

        @:param rewards (pytorch.Tensor): The sampled batch of rewards.
        @:param q_values (pytorch.Tensor): The q-values inferred from target_model.
        @:param dones (pytorch.Tensor): The done values for each transition in the batch.
        @:return (pytorch.Tensor): The TD value for each sample in the batch.
        """
        q_values_max_tuple = pytorch.max(q_values, dim=-1, keepdim=True)
        q_values_max = q_values_max_tuple.values
        rewards = self._adjust_dims_for_tensor(rewards, target_dim=q_values.dim())
        dones = self._adjust_dims_for_tensor(dones, target_dim=q_values.dim())
        td_value = rewards + ((self.gamma * q_values_max) * (1 - dones))
        return td_value

    def __decay_epsilon(self) -> None:
        """
        Protected method to decay epsilon. This method is called every `epsilon_decay_frequency` timesteps and
            decays the epsilon by `epsilon_decay_rate`, both supplied in DqnAgent class' constructor.
        """
        if self.min_epsilon < self.epsilon:
            self.epsilon *= self.epsilon_decay_rate
        if self.min_epsilon > self.epsilon:
            self.epsilon = self.min_epsilon
        return

    def __load_random_experiences(
        self,
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
        This method loads random transitions from memory. This may also include forced terminal states
            if supplied `force_terminal_state_selection_prob` > 0 in DqnAgent constructor for each batch. i.e. if
            force_terminal_state_selection_prob = 0.1, approximately every 1 in 10 batches will have at least
            one terminal state forced by the loader.

        :return: Tuple[
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
           The tuple of tensors as (states_current, states_next, rewards, actions, dones, priorities,
            probabilities, weights, random_indices).

        """
        samples = self.memory.sample(
            self.batch_size,
            self.force_terminal_state_selection_prob,
            alpha=self.prioritization_params["alpha"],
            beta=self.prioritization_params["beta"],
            num_segments=self.prioritization_params["num_segments"],
        )
        return samples

    @staticmethod
    def __anneal_alpha_default_fn(alpha: float, alpha_annealing_factor: float) -> float:
        """
        Protected method to anneal alpha parameter for important sampling weights. This will be called
            every `alpha_annealing_frequency` times. `alpha_annealing_frequency` is a key to be passed in dictionary
            `prioritization_params` argument in the DqnAgent class' constructor. This method is called by default
            to anneal alpha.

        If `alpha_annealing_frequency` is not passed in `prioritization_params`, the annealing of alpha will not take
            place. This method uses another value `alpha_annealing_factor` that must also be passed in
            `prioritization_params`. `alpha_annealing_factor` is typically below 1 to slowly annealed it to
            0 or `min_alpha`.
        """
        alpha *= alpha_annealing_factor
        return alpha

    @staticmethod
    def __anneal_beta_default_fn(beta: float, beta_annealing_factor: float) -> float:
        """
        Protected method to anneal beta parameter for important sampling weights. This will be called
            every `beta_annealing_frequency` times. `beta_annealing_frequency` is a key to be passed in dictionary
            `prioritization_params` argument in the DqnAgent class' constructor.

        If `beta_annealing_frequency` is not passed in `prioritization_params`, the annealing of beta will not take
            place. This method uses another value `beta_annealing_factor` that must also be passed in
            `prioritization_params`. `beta_annealing_factor` is typically above 1 to slowly annealed it to
            1 or `max_beta`
        """
        beta *= beta_annealing_factor
        return beta

    def __anneal_alpha(self):
        if (
            self.prioritization_params["to_anneal_alpha"]
            and (
                self.step_counter
                % self.prioritization_params["alpha_annealing_frequency"]
                == 0
            )
            and self.prioritization_params["alpha"]
            > self.prioritization_params["min_alpha"]
        ):
            self.prioritization_params["alpha"] = self.prioritization_params[
                "alpha_annealing_fn"
            ](
                self.prioritization_params["alpha"],
                *self.prioritization_params["alpha_annealing_fn_args"],
                **self.prioritization_params["alpha_annealing_fn_kwargs"],
            )
            if (
                self.prioritization_params["alpha"]
                < self.prioritization_params["min_alpha"]
            ):
                self.prioritization_params["alpha"] = self.prioritization_params[
                    "min_alpha"
                ]

    def __anneal_beta(self):
        if (
            self.prioritization_params["to_anneal_beta"]
            and (
                self.step_counter
                % self.prioritization_params["beta_annealing_frequency"]
                == 0
            )
            and self.prioritization_params["beta"]
            < self.prioritization_params["max_beta"]
        ):
            self.prioritization_params["beta"] = self.prioritization_params[
                "beta_annealing_fn"
            ](
                self.prioritization_params["beta"],
                *self.prioritization_params["beta_annealing_fn_args"],
                **self.prioritization_params["beta_annealing_fn_kwargs"],
            )
            if (
                self.prioritization_params["beta"]
                > self.prioritization_params["max_beta"]
            ):
                self.prioritization_params["beta"] = self.prioritization_params[
                    "max_beta"
                ]

    def __process_prioritization_params(
        self, prioritization_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Private method to process the prioritization parameters. This includes sanity check and loading of default
            values of mandatory parameters.
        @:param prioritization_params (Dict[str, Any]): The prioritization parameters for when
            we use prioritized memory
        @:return (Dict[str, Any]): The processed prioritization parameters with necessary parameters loaded.
        """
        to_anneal_alpha = False
        to_anneal_beta = False
        if prioritization_params is not None and self.prioritization_strategy_code > 0:
            assert (
                "alpha" in prioritization_params.keys()
            ), "`alpha` must be passed when passing prioritization_params"
            assert (
                "beta" in prioritization_params.keys()
            ), "`beta` must be passed when passing prioritization_params"
        else:
            prioritization_params = dict()
        alpha = float(prioritization_params.get("alpha", -1))
        beta = float(prioritization_params.get("beta", -1))
        min_alpha = float(prioritization_params.get("min_alpha", 1.0))
        max_beta = float(prioritization_params.get("max_beta", 1.0))
        alpha_annealing_frequency = int(
            prioritization_params.get("alpha_annealing_frequency", -1)
        )
        beta_annealing_frequency = int(
            prioritization_params.get("beta_annealing_frequency", -1)
        )
        alpha_annealing_fn = prioritization_params.get("alpha_annealing_fn")
        beta_annealing_fn = prioritization_params.get("beta_annealing_fn")
        if alpha_annealing_fn is None:
            alpha_annealing_fn = self.__anneal_alpha_default_fn
            # Set annealing parameters for alpha.
            alpha_annealing_factor = float(
                prioritization_params.get("alpha_annealing_factor", -1)
            )
            # Check if to anneal alpha based on input parameters.
            if alpha_annealing_frequency != -1 and alpha_annealing_factor != -1:
                to_anneal_alpha = True
            elif alpha_annealing_frequency == -1 and alpha_annealing_factor != -1:
                logging.warning(
                    "alpha_annealing_factor was passed but alpha_annealing_frequency was not passed! "
                    "This will prevent annealing of alpha."
                )
            elif alpha_annealing_frequency != -1 and alpha_annealing_factor == -1:
                logging.warning(
                    "alpha_annealing_frequency was passed but alpha_annealing_factor was not passed! "
                    "This will prevent annealing of alpha."
                )
            alpha_annealing_fn_args = (alpha_annealing_factor,)
            alpha_annealing_fn_kwargs = dict()
        else:
            alpha_annealing_fn = alpha_annealing_fn
            alpha_annealing_fn_args = prioritization_params.get(
                "alpha_annealing_fn_args"
            )
            alpha_annealing_fn_kwargs = prioritization_params.get(
                "alpha_annealing_fn_kwargs"
            )
        if beta_annealing_fn is None:
            beta_annealing_fn = self.__anneal_beta_default_fn
            # Set annealing parameters for beta.
            beta_annealing_factor = float(
                prioritization_params.get("beta_annealing_factor", -1)
            )
            # Check if to anneal beta based on input parameters.
            if beta_annealing_frequency != -1 and beta_annealing_factor != -1:
                to_anneal_beta = True
            elif beta_annealing_frequency == -1 and beta_annealing_factor != -1:
                logging.warning(
                    "beta_annealing_factor was passed but beta_annealing_frequency was not passed! "
                    "This will prevent annealing of beta."
                )
            elif beta_annealing_frequency != -1 and beta_annealing_factor == -1:
                logging.warning(
                    "beta_annealing_frequency was passed but beta_annealing_factor was not passed! "
                    "This will prevent annealing of beta."
                )
            beta_annealing_fn_args = (beta_annealing_factor,)
            beta_annealing_fn_kwargs = dict()
        else:
            beta_annealing_fn_args = prioritization_params.get("beta_annealing_fn_args")
            beta_annealing_fn_kwargs = prioritization_params.get(
                "beta_annealing_fn_kwargs"
            )
        # Error for proportional based prioritized memory.
        error = float(prioritization_params.get("error", 5e-3))
        # Number of segments for rank-based prioritized memory.
        num_segments = prioritization_params.get("num_segments", self.batch_size)
        # Creation of final process dictionary for prioritization_params
        prioritization_params_processed = {
            "to_anneal_alpha": to_anneal_alpha,
            "to_anneal_beta": to_anneal_beta,
            "alpha": alpha,
            "beta": beta,
            "min_alpha": min_alpha,
            "max_beta": max_beta,
            "alpha_annealing_frequency": alpha_annealing_frequency,
            "beta_annealing_frequency": beta_annealing_frequency,
            "alpha_annealing_fn": alpha_annealing_fn,
            "alpha_annealing_fn_args": alpha_annealing_fn_args,
            "alpha_annealing_fn_kwargs": alpha_annealing_fn_kwargs,
            "beta_annealing_fn": beta_annealing_fn,
            "beta_annealing_fn_args": beta_annealing_fn_args,
            "beta_annealing_fn_kwargs": beta_annealing_fn_kwargs,
            "error": error,
            "num_segments": num_segments,
        }
        return prioritization_params_processed

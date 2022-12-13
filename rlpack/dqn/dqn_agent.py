"""!
@package rlpack.dqn
@brief This package implements the DQN methods.


Currently following classes have been implemented:
    - `Dqn`: This class is a helper class that selects the correct variant of DQN agent based on argument
        `prioritization_params`.
    - `DqnAgent`: Implemented as rlpack.dqn.dqn_agent.DqnAgent this class implements the basic DQN methodology, i.e.
        without prioritization. It also acts as a base class for DQN agents with prioritization strategies.
    - `DqnProportionalPrioritizationAgent`: Implemented as
        rlpack.dqn.dqn_proportional_prioritization_agent.DqnProportionalPrioritizationAgent this class implements the
         DQN with proportional prioritization.
    - `DqnRankBasedPrioritizationAgent`: Implemented as
        rlpack.dqn.dqn_rank_based_prioritization_agent.DqnRankBasedPrioritizationAgent; this class implements the
        DQN with rank prioritization.
"""


import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from rlpack import pytorch
from rlpack._C.memory import Memory
from rlpack.utils import LossFunction, LRScheduler
from rlpack.utils.base.agent import Agent
from rlpack.utils.normalization import Normalization
from rlpack.utils.internal_code_setup import InternalCodeSetup


class DqnAgent(Agent):
    """
    This class implements the basic DQN methodology, i.e. DQN without prioritization. This class also acts as a base
    class for other DQN variants all of which override the method `__apply_prioritization_strategy` to implement
    their prioritization strategy.
    """

    def __init__(
        self,
        target_model: pytorch.nn.Module,
        policy_model: pytorch.nn.Module,
        optimizer: pytorch.optim.Optimizer,
        lr_scheduler: Union[LRScheduler, None],
        loss_function: LossFunction,
        gamma: float,
        epsilon: float,
        min_epsilon: float,
        epsilon_decay_rate: float,
        epsilon_decay_frequency: int,
        memory_buffer_size: int,
        target_model_update_rate: int,
        policy_model_update_rate: int,
        backup_frequency: int,
        lr_threshold: float,
        batch_size: int,
        num_actions: int,
        save_path: str,
        device: str = "cpu",
        prioritization_params: Optional[Dict[str, Any]] = None,
        force_terminal_state_selection_prob: float = 0.0,
        tau: float = 1.0,
        apply_norm: Union[int, str] = -1,
        apply_norm_to: Union[int, List[str]] = -1,
        eps_for_norm: float = 5e-12,
        p_for_norm: int = 2,
        dim_for_norm: int = 0,
        max_grad_norm: Optional[float] = None,
        grad_norm_p: float = 2.0,
    ):
        """
        @param target_model: nn.Module: The target network for DQN model. This the network which has
            its weights frozen.
        @param policy_model: nn.Module: The policy network for DQN model. This is the network which is trained.
        @param optimizer: optim.Optimizer: The optimizer wrapped with policy model's parameters.
        @param lr_scheduler: Union[LRScheduler, None]: The PyTorch LR Scheduler with wrapped optimizer.
        @param loss_function: LossFunction: The loss function from PyTorch's nn module. Initialized
            instance must be passed.
        @param gamma: float: The gamma value for agent.
        @param epsilon: float: The initial epsilon for the agent.
        @param min_epsilon: float: The minimum epsilon for the agent. Once this value is reached,
            it is maintained for all further episodes.
        @param epsilon_decay_rate: float: The decay multiplier to decay the epsilon.
        @param epsilon_decay_frequency: int: The number of timesteps after which the epsilon is decayed.
        @param memory_buffer_size: int: The buffer size of memory; or replay buffer for DQN.
        @param target_model_update_rate: int: The timesteps after which target model's weights are updated with
            policy model weights: weights are weighted as per `tau`: see below)).
        @param policy_model_update_rate: int: The timesteps after which policy model is trained. This involves
            backpropagation through the policy network.
        @param backup_frequency: int: The timesteps after which models are backed up. This will also
            save optimizer, lr_scheduler and agent_states: epsilon the time of saving and memory.
        @param lr_threshold: float: The threshold LR which once reached LR scheduler is not called further.
        @param batch_size: int: The batch size used for inference through target_model and train through policy model
        @param num_actions: int: Number of actions for the environment.
        @param save_path: str: The save path for models: target_model and policy_model, optimizer,
            lr_scheduler and agent_states.
        @param device: str: The device on which models are run. Default: "cpu".
        @param prioritization_params: Optional[Dict[str, Any]]: The parameters for prioritization in prioritized
            memory: or relay buffer). Default: None.
        @param force_terminal_state_selection_prob: float: The probability for forcefully selecting a terminal state
            in a batch. Default: 0.0.
        @param tau: float: The weighted update of weights from policy_model to target_model. This is done by formula
            target_weight = tau * policy_weight +: 1 - tau) * target_weight/. Default: -1.
        @param apply_norm: Union[int, str]: The code to select the normalization procedure to be applied on
            selected quantities; selected by `apply_norm_to`: see below)). Direct string can also be
            passed as per accepted keys. Refer below in Notes to see the accepted values. Default: -1
        @param apply_norm_to: Union[int, List[str]]: The code to select the quantity to which normalization is
            to be applied. Direct list of quantities can also be passed as per accepted keys. Refer
            below in Notes to see the accepted values. Default: -1.
        @param eps_for_norm: float: Epsilon value for normalization: for numeric stability. For min-max normalization
            and standardized normalization. Default: 5e-12.
        @param p_for_norm: int: The p value for p-normalization. Default: 2: L2 Norm.
        @param dim_for_norm: int: The dimension across which normalization is to be performed. Default: 0.
        @param max_grad_norm: Optional[float]: The max norm for gradients for gradient clipping. Default: None
        @param grad_norm_p: Optional[float]: The p-value for p-normalization of gradients. Default: 2.0.



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
        super(DqnAgent, self).__init__()
        setup = InternalCodeSetup()
        ## The input target model. This model's parameters are frozen. @I{# noqa: E266}
        self.target_model = target_model.to(device)
        ## The input policy model. @I{# noqa: E266}
        self.policy_model = policy_model.to(device)
        ## The input optimizer wrapped with policy_model parameters. @I{# noqa: E266}
        self.optimizer = optimizer
        ## The input optional LR Scheduler (this can be None). @I{# noqa: E266}
        self.lr_scheduler = lr_scheduler
        ## The input loss function. @I{# noqa: E266}
        self.loss_function = loss_function
        ## The input discounting factor. @I{# noqa: E266}
        self.gamma = gamma
        ## The input exploration factor. @I{# noqa: E266}
        self.epsilon = epsilon
        ## The input minimum exploration factor after decays. @I{# noqa: E266}
        self.min_epsilon = min_epsilon
        ## The input epsilon decay rate. @I{# noqa: E266}
        self.epsilon_decay_rate = epsilon_decay_rate
        ## The input epsilon decay frequency in terms of timesteps. @I{# noqa: E266}
        self.epsilon_decay_frequency = epsilon_decay_frequency
        ## The input argument `memory_buffer_size`; indicating the buffer size used. @I{# noqa: E266}
        self.memory_buffer_size = memory_buffer_size
        ## The input argument `target_model_update_rate`; indicating the update rate of target model. @I{# noqa: E266}
        ## A soft copy of parameters takes place form policy_model to target model as per the update rate @I{# noqa: E266}
        self.target_model_update_rate = target_model_update_rate
        ## The input argument `policy_model_update_rate`; indicating the update rate of policy model. @I{# noqa: E266}
        ## Optimizer is called every `policy_model_update_rate`. @I{# noqa: E266}
        self.policy_model_update_rate = policy_model_update_rate
        ## The input model backup frequency in terms of timesteps. @I{# noqa: E266}
        self.backup_frequency = backup_frequency
        ## The input LR Threshold. @I{# noqa: E266}
        self.lr_threshold = float(lr_threshold)
        ## The batch size to be used when training policy model. @I{# noqa: E266}
        ## Corresponding number of samples are drawn from @ref memory as per the prioritization strategy @I{# noqa: E266}
        self.batch_size = batch_size
        ## The input number of actions. @I{# noqa: E266}
        self.num_actions = num_actions
        ## The input save path for backing up agent models. @I{# noqa: E266}
        self.save_path = save_path
        ## The input `device` argument; indicating the device name. @I{# noqa: E266}
        self.device = device
        # Set necessary prioritization parameters. Depending on `prioritization_params`, appropriate values are set.
        ## The prioritization strategy code. @I{# noqa: E266}
        self.__prioritization_strategy_code = (
            prioritization_params.get("prioritization_strategy_code", 0)
            if prioritization_params is not None
            else 0
        )
        ## The input prioritization parameters. @I{# noqa: E266}
        self.prioritization_params = prioritization_params
        ## The input `force_terminal_state_selection_prob`. @I{# noqa: E266}
        ## This indicates the probability to force at least one terminal state sample in a batch.  @I{# noqa: E266}
        self.force_terminal_state_selection_prob = force_terminal_state_selection_prob
        # Sanity check for tau value.
        if tau < 0 or tau > 1:
            ValueError(
                "Invalid value for tau passed! Expected value is between 0 and 1"
            )
        ## The input `tau`; indicating the soft update used to update @ref target_model parameters. @I{# noqa: E266}
        self.tau = float(tau)
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
        ## The step counter; counting the total timesteps done so far up to @ref memory_buffer_size. @I{# noqa: E266}
        ## Once `buffer_size` is reached, this will restarted. @I{# noqa: E266}
        self.step_counter = 1
        ## The instance of @ref rlpack._C.memory.Memory used for Replay buffer. @I{# noqa: E266}
        self.memory = Memory(
            buffer_size=memory_buffer_size,
            device=device,
            prioritization_strategy_code=self.__prioritization_strategy_code,
            batch_size=self.batch_size,
        )
        # Disable gradients for target network.
        for n, p in self.target_model.named_parameters():
            p.requires_grad = False
        ## The normalisation tool to be used for agent. @I{# noqa: E266}
        ## An instance of rlpack.utils.normalization.Normalization. @I{# noqa: E266}
        self.__normalization = Normalization(
            apply_norm=apply_norm, eps=eps_for_norm, p=p_for_norm, dim=dim_for_norm
        )

    def train(
        self,
        state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        state_next: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]],
        reward: Union[int, float],
        action: Union[int, float],
        done: Union[bool, int],
        priority: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1.0,
        probability: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1.0,
        weight: Optional[Union[pytorch.Tensor, np.ndarray, float]] = 1.0,
    ) -> int:
        """
        - The training method for agent, which accepts a transition from environment and returns an action for next
            transition. Use this method when you intend to train the agent.
        - This method will also run the policy to yield the best action for the given state.
        - For each transition (or experience) being passed, associated priority, probability and weight
            can be passed.

        @param state_current: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The current
            state in the environment.
        @param state_next: Union[pytorch.Tensor, np.ndarray, List[Union[float, int]]]: The next
            state returned by the environment.
        @param reward: Union[int, float]: Reward obtained by performing the action for the transition.
        @param action: Union[int, float]: Action taken for the transition
        @param done: Union[bool, int]: Indicates weather episode has terminated or not.
        @param priority: Optional[Union[pytorch.Tensor, np.ndarray, float]]: The priority of the
            transition: for priority relay memory). Default: 1.0
        @param probability: Optional[Union[pytorch.Tensor, np.ndarray, float]]: The probability of the transition
           : for priority relay memory). Default: 1.0
        @param weight: Optional[Union[pytorch.Tensor, np.ndarray, float]]: The important sampling weight
            of the transition: for priority relay memory). Default: 1.0
        @return int: The next action to be taken from `state_next`.
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
        # Backup model every `backup_frequency` steps.
        if self.step_counter % self.backup_frequency == 0:
            self.save()
        # Restart `step_counter` if `it has reached the buffer size.
        if self.step_counter == self.memory_buffer_size:
            self.step_counter = 0
        # If using prioritized memory, anneal alpha and beta.
        if self.__prioritization_strategy_code > 0:
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
        @param state_current: Union[ndarray, pytorch.Tensor, List[float]]: The current state agent is in.
        @return int: The action to be taken.
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
                state_current = self.__normalization.apply_normalization(state_current)
            q_values = self.policy_model(state_current)
            action_tensor = q_values.argmax(-1)
            action = action_tensor.item()
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
        checkpoint_target = {
            "state_dict": self.target_model.state_dict(),
        }
        checkpoint_policy = {"state_dict": self.policy_model.state_dict()}
        checkpoint_optimizer = {"state_dict": self.optimizer.state_dict()}
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
            checkpoint_lr_scheduler = {"state_dict": self.lr_scheduler.state_dict()}
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
            state_current = self.__normalization.apply_normalization(state_current)
            state_next = self.__normalization.apply_normalization(state_next)
        # Apply normalization if required to rewards.
        if self.apply_norm_to in self.reward_norm_codes:
            rewards = self.__normalization.apply_normalization(rewards)
        # Set policy model to training mode.
        if not self.policy_model.training:
            self.policy_model.train()
        # Compute target q-values from target model and temporal difference values.
        with pytorch.no_grad():
            q_values_target = self.target_model(state_next)
            td_value = self.temporal_difference(rewards, q_values_target, dones)
        # Apply normalization if required to TD values.
        if self.apply_norm_to in self.td_norm_codes:
            td_value = self.__normalization.apply_normalization(td_value)
        # Compute current q-values from policy model.
        q_values_policy = self.policy_model(state_current)
        actions = self._adjust_dims_for_tensor(
            tensor=actions, target_dim=q_values_target.dim()
        )
        # Choose best q-values corresponding to actions
        q_values_gathered = pytorch.gather(q_values_policy, dim=-1, index=actions)
        self.optimizer.zero_grad()
        # If prioritized memory is used, multiply weights from sampling process to TD values.
        if self.__prioritization_strategy_code > 0:
            weights_ = self._adjust_dims_for_tensor(weights, td_value.dim())
            td_value = td_value * weights_
        td_value = td_value.detach()
        loss = self.loss_function(q_values_gathered, td_value)
        loss.backward()
        self.loss.append(loss.item())
        # Clip gradients if requested.
        if self.max_grad_norm is not None:
            pytorch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=self.grad_norm_p,
            )
        self.__apply_prioritization_strategy(
            td_value, random_indices
        )
        self.optimizer.step()
        if (
            self.lr_scheduler is not None
            and min(self.lr_scheduler.get_last_lr()) > self.lr_threshold
        ):
            self.lr_scheduler.step()

    def __apply_prioritization_strategy(
        self,
        td_value: pytorch.Tensor,
        random_indices: pytorch.Tensor,
    ) -> None:
        """
        Void private method that applies the relevant prioritization strategy for the DQN.
        @param td_value: pytorch.Tensor: The computed TD value.
        @param random_indices: The indices of randomly sampled transitions.
        """
        return

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

        @param rewards: pytorch.Tensor: The sampled batch of rewards.
        @param q_values: pytorch.Tensor: The q-values inferred from target_model.
        @param dones: pytorch.Tensor: The done values for each transition in the batch.
        @return pytorch.Tensor: The TD value for each sample in the batch.
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

        @return Tuple[
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
            self.force_terminal_state_selection_prob,
            alpha=self.prioritization_params["alpha"],
            beta=self.prioritization_params["beta"],
            num_segments=self.prioritization_params["num_segments"],
        )
        return samples

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
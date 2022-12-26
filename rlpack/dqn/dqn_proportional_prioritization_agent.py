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


from typing import Any, Dict, List, Optional, Union

from rlpack import pytorch
from rlpack.dqn.dqn_agent import DqnAgent
from rlpack.utils import LossFunction, LRScheduler


class DqnProportionalPrioritizationAgent(DqnAgent):
    """
    This class implements the DQN with Proportional prioritization strategy.
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
        bootstrap_rounds: int = 1,
        device: str = "cpu",
        dtype: str = "float32",
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
        clip_grad_value: Optional[float] = None,
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
        @param bootstrap_rounds: int: The number of rounds until which gradients are to be accumulated before
            performing calling optimizer step. Gradients are mean reduced for bootstrap_rounds > 1. Default: 1.
        @param device: str: The device on which models are run. Default: "cpu".
        @param dtype: str: The datatype for model parameters. Default: "float32"
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
        @param clip_grad_value: Optional[float]: The gradient value for clipping gradients by value. Default: None


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

        If a valid `clip_grad_value` is passed, then gradients will be clipped by value. If `clip_grad_value` value
        was invalid, error will be raised from PyTorch.
        """
        super(DqnProportionalPrioritizationAgent, self).__init__(
            target_model,
            policy_model,
            optimizer,
            lr_scheduler,
            loss_function,
            gamma,
            epsilon,
            min_epsilon,
            epsilon_decay_rate,
            epsilon_decay_frequency,
            memory_buffer_size,
            target_model_update_rate,
            policy_model_update_rate,
            backup_frequency,
            lr_threshold,
            batch_size,
            num_actions,
            save_path,
            bootstrap_rounds,
            device,
            dtype,
            prioritization_params,
            force_terminal_state_selection_prob,
            tau,
            apply_norm,
            apply_norm_to,
            eps_for_norm,
            p_for_norm,
            dim_for_norm,
            max_grad_norm,
            grad_norm_p,
            clip_grad_value,
        )

    def _apply_prioritization_strategy(
        self,
        td_value: pytorch.Tensor,
        random_indices: pytorch.Tensor,
    ) -> None:
        """
        Void private method that applies the relevant prioritization strategy for the DQN.
        @param td_value: pytorch.Tensor: The computed TD value.
        @param random_indices: The indices of randomly sampled transitions.
        """
        if self.__prioritization_strategy_code == 1:
            new_priorities = (
                pytorch.abs(td_value.cpu()) + self.prioritization_params["error"]
            )
            self.memory.update_priorities(
                random_indices,
                new_priorities,
            )

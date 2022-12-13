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


from typing import Any, Callable, Dict, Optional, Union

from rlpack import pytorch
from rlpack.dqn.dqn_agent import DqnAgent
from rlpack.dqn.dqn_proportional_prioritization_agent import (
    DqnProportionalPrioritizationAgent,
)
from rlpack.dqn.dqn_rank_based_prioritization_agent import (
    DqnRankBasedPrioritizationAgent,
)
from rlpack.utils import LossFunction, LRScheduler
from rlpack.utils.internal_code_setup import InternalCodeSetup


class Dqn:
    """
    This is a helper class that selects the correct the variant of DQN implementations based on prioritization
    strategy determined by the argument `prioritization_params`.
    """

    def __new__(
        cls,
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
        prioritization_params: Optional[Dict[str, Any]] = None,
        force_terminal_state_selection_prob: float = 0.0,
        tau: float = 1.0,
        apply_norm: int = -1,
        apply_norm_to: int = -1,
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
        @param bootstrap_rounds: int: The number of rounds until which gradients are to be accumulated before
            performing calling optimizer step. Gradients are mean reduced for bootstrap_rounds > 1. Default: 1.
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
        @param grad_norm_p: Optional[float]: The p-value for p-normalization of gradients. Default: 2.0



        **Notes**


        For prioritization_params, when None: the default is passed, prioritized memory is not used. To use
            prioritized memory, pass a dictionary with keys `alpha` and `beta`. You can also pass `alpha_decay_rate`
            and `beta_decay_rate` additionally.


        The code for prioritization strategies are:
            - Uniform: 0; `uniform`
            - Proportional: 1; `proportional`
            - Rank-Based: 2; `rank-based`


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
        if prioritization_params is None:
            prioritization_params = dict()
        prioritization_strategy = prioritization_params.get(
            "prioritization_strategy", "uniform"
        )
        setup = InternalCodeSetup()
        prioritization_strategy_code = setup.get_prioritization_code(
            prioritization_strategy=prioritization_strategy
        )
        prioritization_params = cls.__process_prioritization_params(
            prioritization_params=prioritization_params,
            prioritization_strategy_code=prioritization_strategy_code,
            anneal_alpha_default_fn=cls.__anneal_alpha_default_fn,
            anneal_beta_default_fn=cls.__anneal_beta_default_fn,
            batch_size=batch_size,
        )
        args_ = (
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
        )
        # Select the appropriate DQN variant.
        if prioritization_strategy_code == 0:
            dqn = DqnAgent(*args_)
        elif prioritization_strategy_code == 1:
            dqn = DqnProportionalPrioritizationAgent(*args_)
        elif prioritization_strategy_code == 2:
            dqn = DqnRankBasedPrioritizationAgent(*args_)
        else:
            raise NotImplementedError(
                f"The provided prioritization strategy {prioritization_strategy} is not supported or is invalid!"
            )
        return dqn

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

        @param alpha: float: The input alpha value to anneal.
        @param alpha_annealing_factor: float: The annealing factor to be used to anneal alpha.
        @return float: Annealed alpha.
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

        @param beta: float: The input beta value to anneal.
        @param beta_annealing_factor: float: The annealing factor to be used to anneal beta.
        @return float: Annealed beta.
        """
        beta *= beta_annealing_factor
        return beta

    @staticmethod
    def __process_prioritization_params(
        prioritization_params: Dict[str, Any],
        prioritization_strategy_code: int,
        anneal_alpha_default_fn: Callable[[float, float], float],
        anneal_beta_default_fn: Callable[[float, float], float],
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Private method to process the prioritization parameters. This includes sanity check and loading of default
            values of mandatory parameters.
        @param prioritization_params: Dict[str, Any]: The prioritization parameters for when
            we use prioritized memory.
        @param prioritization_strategy_code: int: The prioritization code corresponding to the given
            prioritization strategy string.
        @param anneal_alpha_default_fn: Callable[[float, float], float]: The default annealing function for alpha.
        @param anneal_beta_default_fn: Callable[[float, float], float]: The default annealing function for beta.
        @param batch_size: int: The requested batch size; used in rank-based prioritization to determine the number of
            segments.
        @return Dict[str, Any]: The processed prioritization parameters with necessary parameters loaded.
        """
        to_anneal_alpha = False
        to_anneal_beta = False
        if prioritization_params is not None and prioritization_strategy_code > 0:
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
        alpha_annealing_fn = prioritization_params.get(
            "alpha_annealing_fn", anneal_alpha_default_fn
        )
        beta_annealing_fn = prioritization_params.get(
            "beta_annealing_fn", anneal_beta_default_fn
        )
        # Check if to anneal alpha based on input parameters.
        if alpha_annealing_frequency != -1:
            to_anneal_alpha = True
        # Get args and kwargs for to pass to alpha_annealing_fn.
        alpha_annealing_fn_args = prioritization_params.get(
            "alpha_annealing_fn_args", tuple()
        )
        alpha_annealing_fn_kwargs = prioritization_params.get(
            "alpha_annealing_fn_kwargs", dict()
        )
        # Check if to anneal beta based on input parameters.
        if beta_annealing_frequency != -1:
            to_anneal_beta = True
        # Get args and kwargs for to pass to beta_annealing_fn.
        beta_annealing_fn_args = prioritization_params.get(
            "beta_annealing_fn_args", tuple()
        )
        beta_annealing_fn_kwargs = prioritization_params.get(
            "beta_annealing_fn_kwargs", dict()
        )
        # Error for proportional based prioritized memory.
        error = float(prioritization_params.get("error", 5e-3))
        # Number of segments for rank-based prioritized memory.
        num_segments = prioritization_params.get("num_segments", batch_size)
        # Creation of final process dictionary for prioritization_params
        prioritization_params_processed = {
            "prioritization_strategy_code": prioritization_strategy_code,
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

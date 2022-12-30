"""!
@package rlpack.actor_critic
@brief This package implements the Actor-Critic methods.


Currently following methods are implemented:
    - `AC`: Implemented in rlpack.actor_critic.ac.AC. More details
        can be found [here](@ref agents/actor_critic.ac.md)
    - `A2C`: Implemented in rlpack.actor_critic.a2c.A2C. More details can be found
     [here](@ref agents/actor_critic/a2c.md)
     - `A3C`: Implemented in rlpack.actor_critic.a3c.A3C. More details can be found
     [here](@ref agents/actor_critic/a3c.md)

Following packages are part of actor_critic:
    - `utils`: A package utilities for actor_critic package.
"""


from typing import List, Optional, Tuple, Type, Union

from rlpack import pytorch, pytorch_distributed, pytorch_distributions
from rlpack.actor_critic.utils.actor_critic_agent import ActorCriticAgent
from rlpack.exploration.utils.exploration import Exploration
from rlpack.utils import LossFunction, LRScheduler
from rlpack.utils.exceptions import AgentError


class A3C(ActorCriticAgent):
    """
    The A3C class implements the asynchronous Actor-Critic method. This uses PyTorch's multiprocessing for
    gradient all-reduce operations. In this implementation master process' policy model is assumed to be
    the global model as well.
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
                element can be an empty list, if you wish to sample the default no. of samples.
        @param backup_frequency: int: The timesteps after which policy model, optimizer states and lr
            scheduler states are backed up.
        @param exploration_tool: Union[Exploration, None]: Exploration tool to be used to explore the environment.
            These tools can be found in `rlpack.exploration`.
        @param save_path: str: The path where policy model, optimizer states and lr scheduler states are to be saved.
        @param rollout_accumulation_size: Union[int, None]: The size of rollout buffer before performing optimizer
            step. Whole rollout buffer is used to fit the policy model and is cleared. By default, after every episode.
             Default: None.
        @param grad_accumulation_rounds: int: The number of rounds until which gradients are to be accumulated before
            performing calling optimizer step. Gradients are mean reduced for grad_accumulation_rounds > 1. Default: 1.
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
        super(A3C, self).__init__(
            policy_model,
            optimizer,
            lr_scheduler,
            loss_function,
            distribution,
            gamma,
            entropy_coefficient,
            state_value_coefficient,
            lr_threshold,
            action_space,
            backup_frequency,
            save_path,
            rollout_accumulation_size,
            grad_accumulation_rounds,
            exploration_tool,
            device,
            dtype,
            apply_norm,
            apply_norm_to,
            eps_for_norm,
            p_for_norm,
            dim_for_norm,
            max_grad_norm,
            grad_norm_p,
            clip_grad_value,
        )
        if not pytorch_distributed.is_initialized():
            raise AgentError("A3C can only be launched in distributed setting!")
        ## The callable method for gradient reduction in distributed environment.@I{# noqa: E266}
        self._distributed_grad_reduce_method = self._async_grad_all_reduce
        ## The callable method for parameter scattering in distributed environment. @I{# noqa: E266}
        self._distributed_param_scatter_method = self._async_param_scatter
        ## The process rank for A3C worker. @I{# noqa: E266}
        self._process_rank = pytorch_distributed.get_rank()
        ## The world size for A3C workers @I{# noqa: E266}
        self._world_size = pytorch_distributed.get_world_size()

    def _call_to_save(self) -> None:
        """
        Method calling the save method when required. Only saved from first process (process with rank 0).
        """
        if (
            (self.step_counter + 1) % self.backup_frequency == 0
        ) and self._process_rank == self._master_process_rank:
            self.save()

    @pytorch.no_grad()
    def _async_grad_all_reduce(self) -> None:
        """
        Asynchronously averages the gradients across the world_size (number of processes) using non-blocking
        all-reduce method.
        """
        for param in self.policy_model.parameters():
            if param.requires_grad:
                pytorch_distributed.reduce(
                    param.grad,
                    dst=self._master_process_rank,
                    op=pytorch_distributed.ReduceOp.SUM,
                    async_op=True,
                )
                param.grad /= self._world_size

    @pytorch.no_grad()
    def _async_param_scatter(self):
        for param in self.policy_model.parameters():
            if self._process_rank != self._master_process_rank:
                scatter_list = None
            else:
                scatter_list = [param] * self._world_size
            pytorch_distributed.scatter(
                param,
                scatter_list=scatter_list,
                src=self._master_process_rank,
                async_op=True,
            )

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


from abc import ABC
from datetime import timedelta
from typing import List, Optional, Tuple, Type, Union

from rlpack import pytorch, pytorch_distributed, pytorch_distributions
from rlpack.actor_critic.utils.actor_critic_agent import ActorCriticAgent
from rlpack.exploration.utils.exploration import Exploration
from rlpack.utils import LossFunction, LRScheduler
from rlpack.utils.base.model import Model
from rlpack.utils.exceptions import AgentError
from rlpack.utils.normalization import Normalization


class A3C(ActorCriticAgent, ABC):
    """
    The A3C class implements the asynchronous Actor-Critic method. This uses PyTorch's multiprocessing for
    gradient all-reduce operations. In this implementation master process' policy model is assumed to be
    the global model as well.
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
            gae_lambda,
            exploration_steps,
            grad_accumulation_rounds,
            training_frequency,
            exploration_tool,
            device=device,
            dtype=dtype,
            normalization_tool=normalization_tool,
            apply_norm_to=apply_norm_to,
            max_grad_norm=max_grad_norm,
            grad_norm_p=grad_norm_p,
            clip_grad_value=clip_grad_value,
            timeout=timeout,
        )
        if not pytorch_distributed.is_initialized():
            raise AgentError("A3C can only be launched in distributed setting!")

    def _set_attribute_custom_values(self) -> None:
        # The process rank for A3C worker.
        self._process_rank = pytorch_distributed.get_rank()
        # The world size for A3C workers.
        self._world_size = pytorch_distributed.get_world_size()
        # The process group to current World.
        self._process_group = pytorch_distributed.group.WORLD
        # Disable gradient processing, optimizer step and lr scheduler step for all workers.
        if self._process_rank != self._master_process_rank:
            # The flag indicating weather to perform gradient processing (normalization and clipping).
            # Initialized to False for workers.
            self._perform_grad_processing = False
            # The flag indicating weather to take optimizer step. Initialized to False for workers.
            self._take_optimizer_step = False
            # The flag indicating weather to take LR scheduler step. Initialized to False for workers.
            self._take_lr_scheduler_step = False

    def _call_to_save(self) -> None:
        """
        Method calling the save method when required. Only saved from first process (process with rank 0).
        """
        if (
            (self.step_counter + 1) % self.backup_frequency == 0
        ) and self._process_rank == self._master_process_rank:
            self.save()

    @pytorch.no_grad()
    def _call_to_extend_transitions(self) -> None:
        """
        Method to extend the transitions with all gather. Here the method does nothing and returns immediately.
        """
        return

    @pytorch.no_grad()
    def _share_gradients(self) -> None:
        """
        Asynchronously averages the gradients across the world_size (number of processes) using non-blocking
        reduce method to share gradients to master process. Here the method is implemented with reduce operation
        at master process.
        """
        for param in self.policy_model.parameters():
            if param.requires_grad:
                self._process_group.reduce(
                    tensor=param.grad,
                    root=self._master_process_rank,
                    op=pytorch_distributed.ReduceOp.SUM,
                )
                param.grad /= self._world_size

    @pytorch.no_grad()
    def _share_parameters(self):
        """
        Method to share parameters from a single model. This must typically implement scatter collective communication
        operation. Here the method is implemented by scattering parameters from master process to other processes.
        """
        for param in self.policy_model.parameters():
            self._process_group.broadcast(tensor=param, root=self._master_process_rank)

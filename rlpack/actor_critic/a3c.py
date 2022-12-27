"""!
@package rlpack.actor_critic
@brief This package implements the Actor-Critic methods.


Currently following methods are implemented:
    - `A2C`: Implemented in rlpack.actor_critic.a2c.A2C. More details can be found
     [here](@ref agents/actor_critic/a2c.md)
     - `A3C`: Implemented in rlpack.actor_critic.a3c.A3C. More details can be found
     [here](@ref agents/actor_critic/a3c.md)
"""


from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np

from rlpack import pytorch, pytorch_distributed, pytorch_distributions
from rlpack.actor_critic.utils.actor_critic_agent import ActorCriticAgent
from rlpack.utils import LossFunction, LRScheduler


class A3C(ActorCriticAgent):
    """
    The A3C class implements the asynchronous Actor-Critic method. This uses PyTorch's multiprocessing for
    gradient all-reduce operations.
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
        bootstrap_rounds: int = 1,
        add_gaussian_noise: bool = False,
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
                Callable[[Union[float, np.ndarray, pytorch.Tensor], bool, int], float],
            ]
        ] = None,
        max_timesteps: int = 1000,
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
        @param add_gaussian_noise: bool: Parameter indicating whether to add gaussian noise from standard normal
            distribution; N(0, 1) is used. Default: False.
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
        @param max_timesteps: int: The maximum timesteps the environment will run for. This is used for memory
            preemptive allocation for improved efficiency.


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
            bootstrap_rounds,
            add_gaussian_noise,
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
            variance,
            max_timesteps,
        )

    def _call_to_save(self) -> None:
        """
        Method calling the save method when required. Only saved from first process (process with rank 0).
        """
        if (
            (self.step_counter + 1) % self.backup_frequency == 0
        ) and pytorch_distributed.get_rank() == 0:
            self.save()

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
        # Asynchronous gradient mean reduction.
        self._async_gradients()
        # Clip gradients if requested.
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
    def _async_gradients(self) -> None:
        """
        Asynchronously averages the gradients across the world_size (number of processes) using non-blocking
        all-reduce method.
        """
        world_size = pytorch_distributed.get_world_size()
        for param in self.policy_model.parameters():
            if param.requires_grad:
                pytorch_distributed.all_reduce(
                    param.grad, op=pytorch_distributed.ReduceOp.SUM
                )
                param.grad /= world_size

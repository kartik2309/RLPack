from collections import OrderedDict
from typing import Optional, Union

from rlpack import dist, pytorch
from rlpack.actor_critic.a2c import A2C
from rlpack.utils import LossFunction, LRScheduler


class A3C(A2C):
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
        super(A3C, self).__init__(
            policy_model,
            optimizer,
            lr_scheduler,
            loss_function,
            gamma,
            entropy_coefficient,
            state_value_coefficient,
            lr_threshold,
            num_actions,
            backup_frequency,
            save_path,
            bootstrap_rounds,
            device,
            apply_norm,
            apply_norm_to,
            eps_for_norm,
            p_for_norm,
            dim_for_norm,
            max_grad_norm,
            grad_norm_p,
        )

    def _call_to_save(self) -> None:
        if (self.step_counter % self.backup_frequency == 0) and dist.get_rank() == 0:
            self.save()

    def _accumulate_gradients(self) -> None:
        """
        Protected void method to train the model or accumulate the gradients for training.
        - If bootstrap_rounds is passed as 1 (default), model is trained each time the method is called.
        - If bootstrap_rounds > 1, the gradients are accumulated in grad_accumulator and model is trained via
            _train_models method.
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
        if self.bootstrap_rounds < 2:
            self.optimizer.zero_grad()
        # Backward call
        loss.backward()
        # Append loss to list
        self.loss.append(loss.item())
        if self.bootstrap_rounds < 2:
            self._async_gradients()
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
            self._grad_accumulator.append(
                {
                    k: param.grad.detach().clone()
                    for k, param in self.policy_model.named_parameters()
                },
            )
        self._clear()

    def _train_models(self) -> None:
        """
        Protected method to policy model if boostrap_rounds > 1. In such cases the gradients are accumulated in
        grad_accumulator. This method collects the accumulated gradients and performs mean reduction and runs
        optimizer step.
        """
        policy_model_grads = self._grad_accumulator
        # OrderedDict to store reduced average value.
        policy_model_grads_reduced = OrderedDict()
        self.optimizer.zero_grad()
        # No Grad mode to disable PyTorch Operation tracking.
        with pytorch.no_grad():
            # Perform parameter wise summation.
            for key in self._policy_model_parameter_keys:
                for policy_model_grad in policy_model_grads:
                    if key not in policy_model_grads_reduced.keys():
                        policy_model_grads_reduced[key] = policy_model_grad[key]
                        continue
                    policy_model_grads_reduced[key] += policy_model_grad[key]
            # Assign average parameters to model.
            for key, param in self.policy_model.named_parameters():
                param.grad = policy_model_grads_reduced[key] / self.bootstrap_rounds
        # Clip gradients if requested.
        if self.max_grad_norm is not None:
            pytorch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=self.grad_norm_p,
            )
        # Synchronize gradients across different processes.
        self._async_gradients()
        # Take an optimizer step.
        self.optimizer.step()
        # Take an LR Scheduler step if required.
        if (
            self.lr_scheduler is not None
            and min([*self.lr_scheduler.get_last_lr()]) > self.lr_threshold
        ):
            self.lr_scheduler.step()
        # Clear buffers for tracked operations.
        self._clear()
        # Clear Accumulated Gradient buffer.
        self._grad_accumulator.clear()

    @pytorch.no_grad()
    def _async_gradients(self):
        """
        Asynchronously averages the gradients across the world_size (number of processes) using non-blocking
        all-reduce method.
        """
        world_size = dist.get_world_size()
        for param in self.policy_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
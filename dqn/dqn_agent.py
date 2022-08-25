import os
import random
import time
from collections import OrderedDict
from typing import List, Optional, TypeVar, Union

import torch
from numpy import ndarray

from utils.base.agent import Agent
from utils.memory import Memory
from utils.normalization import Normalization

LRScheduler = TypeVar("LRScheduler")
LossFunction = TypeVar("LossFunction")
Activation = TypeVar("Activation")

MEMORY_SHUFFLE_SEED_MIN = 1223372036854775807
MEMORY_SHUFFLE_SEED_MAX = 9223372036854775807


class DqnAgent(Agent):
    def __init__(
        self,
        target_model: torch.nn.Module,
        policy_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        loss_function: LossFunction,
        gamma: float,
        epsilon: float,
        min_epsilon: float,
        epsilon_decay_rate: int,
        epsilon_decay_frequency: int,
        memory_buffer_size: int,
        target_model_update_rate: int,
        policy_model_update_rate: int,
        model_backup_frequency: int,
        min_lr: float,
        batch_size: int,
        num_actions: int,
        save_path: str,
        device: str = "cpu",
        force_terminal_state_selection_prob: float = 0.0,
        tau: float = 1.0,
        apply_norm: int = -1,
        apply_norm_to: int = -1,
        eps_for_norm: float = 5e-12,
        p_for_norm: int = 2,
        dim_for_norm: int = 0,
    ):
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
        self.min_lr = float(min_lr)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.save_path = save_path
        self.device = device
        self.force_terminal_state_selection_prob = force_terminal_state_selection_prob

        if tau < 0 or tau > 1:
            ValueError(
                "Invalid value for tau passed! Expected value is between 0 and 1"
            )

        self.tau = tau
        self.apply_norm = apply_norm
        self.apply_norm_to = apply_norm_to
        self.eps_for_norm = eps_for_norm
        self.p_for_norm = p_for_norm
        self.dim_for_norm = dim_for_norm

        self.memory = Memory(buffer_size=memory_buffer_size, device=self.device)
        self.generator = torch.Generator(self.device)
        self.step_counter = 1

        for n, p in self.target_model.named_parameters():
            p.requires_grad = False

        self.normalization = Normalization(apply_norm=apply_norm)

    def train(
        self,
        state_current: Union[ndarray, torch.Tensor, List[float]],
        state_next: Union[ndarray, torch.Tensor, List[float]],
        reward: Union[int, float],
        action: Union[int, float],
        done: Union[bool, int],
    ) -> int:
        state_current = self.memory.cast_to_tensor(state_current)
        state_next = self.memory.cast_to_tensor(state_next)

        if isinstance(done, bool):
            done = int(done)

        self.memory.append(state_current, state_next, reward, action, done)

        if (
            self.step_counter % self.policy_model_update_rate == 0
            and len(self.memory) >= self.batch_size
        ):
            self.__train_policy_model()

        if self.step_counter % self.target_model_update_rate == 0:
            self.__update_target_model()

        if self.step_counter % self.epsilon_decay_frequency == 0:
            self.__decay_epsilon()

        if self.step_counter % self.model_backup_frequency == 0:
            self.save()

        if len(self.memory) == self.memory_buffer_size:
            self.loss.clear()
            self.step_counter = 0

        self.step_counter += 1
        state_current = torch.unsqueeze(state_current, 0)
        action = self.policy(state_current)
        return action

    @torch.no_grad()
    def policy(self, state_current: Union[ndarray, torch.Tensor, List[float]]) -> int:
        if not isinstance(state_current, torch.Tensor):
            state_current = self.memory.cast_to_tensor(state_current)
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
        if custom_name_suffix is None:
            custom_name_suffix = ""
        checkpoint_target = {
            "state_dict": self.target_model.state_dict(),
        }
        checkpoint_policy = {"state_dict": self.policy_model.state_dict()}
        checkpoint_optimizer = {"state_dict": self.optimizer.state_dict()}

        checkpoint_lr_scheduler = dict()
        if self.lr_scheduler is not None:
            checkpoint_lr_scheduler = {"state_dict": self.lr_scheduler.state_dict()}

        agent_state = {"epsilon": self.epsilon}

        torch.save(
            checkpoint_target,
            os.path.join(self.save_path, f"target{custom_name_suffix}.pt"),
        )
        torch.save(
            checkpoint_policy,
            os.path.join(self.save_path, f"policy{custom_name_suffix}.pt"),
        )
        torch.save(
            checkpoint_optimizer,
            os.path.join(self.save_path, f"optimizer{custom_name_suffix}.pt"),
        )
        if self.lr_scheduler is not None:
            torch.save(
                checkpoint_lr_scheduler,
                os.path.join(self.save_path, f"lr_scheduler{custom_name_suffix}.pt"),
            )

        torch.save(agent_state, os.path.join(self.save_path, "agent_state.pt"))
        return

    def load(self, custom_name_suffix: Optional[str] = None) -> None:
        if custom_name_suffix is None:
            custom_name_suffix = ""
        checkpoint_target = torch.load(
            os.path.join(self.save_path, f"target{custom_name_suffix}.pt"),
            map_location="cpu",
        )
        checkpoint_policy = torch.load(
            os.path.join(self.save_path, f"policy{custom_name_suffix}.pt"),
            map_location="cpu",
        )
        checkpoint_optimizer = torch.load(
            os.path.join(self.save_path, f"optimizer{custom_name_suffix}.pt"),
            map_location="cpu",
        )

        checkpoint_lr_sc = None
        if os.path.isfile(
            os.path.join(self.save_path, f"lr_scheduler{custom_name_suffix}.pt")
        ):
            checkpoint_lr_sc = torch.load(
                os.path.join(self.save_path, f"lr_scheduler{custom_name_suffix}.pt"),
                map_location="cpu",
            )

        self.target_model.load_state_dict(checkpoint_target["state_dict"])
        self.policy_model.load_state_dict(checkpoint_policy["state_dict"])
        self.optimizer.load_state_dict(checkpoint_optimizer["state_dict"])
        if self.lr_scheduler is not None and checkpoint_lr_sc is not None:
            self.lr_scheduler.load_state_dict(checkpoint_lr_sc["state_dict"])

        agent_state = torch.load(os.path.join(self.save_path, "agent_state.pt"))
        self.epsilon = agent_state["epsilon"]
        return

    def __train_policy_model(self) -> None:
        random_experiences = self.__load_random_experiences()
        state_current = random_experiences.stack_current_states()
        state_next = random_experiences.stack_next_states()
        rewards = random_experiences.stack_rewards()
        actions = random_experiences.stack_actions()
        dones = random_experiences.stack_dones()

        if self.apply_norm_to in self.state_norm_codes:
            state_current = self.normalization.apply_normalization(
                state_current, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
            )
            state_next = self.normalization.apply_normalization(
                state_next, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
            )

        if self.apply_norm_to in self.reward_norm_codes:
            rewards = self.normalization.apply_normalization(
                rewards, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
            )

        if not self.policy_model.training:
            self.policy_model.train()

        with torch.no_grad():
            q_values_target = self.target_model(state_next)
            td_value = self.__temporal_difference(rewards, q_values_target, dones)

        if self.apply_norm_to in self.td_norm_codes:
            td_value = self.normalization.apply_normalization(
                td_value, self.eps_for_norm, self.p_for_norm, self.dim_for_norm
            )

        q_values_policy = self.policy_model(state_current)
        q_values_gathered = torch.gather(q_values_policy, dim=-1, index=actions)
        self.optimizer.zero_grad()

        loss = self.loss_function(q_values_gathered, td_value.detach())
        loss.backward()
        self.loss.append(loss.item())
        self.optimizer.step()
        if (
            self.lr_scheduler is not None
            and min(self.lr_scheduler.get_last_lr()) > self.min_lr
        ):
            self.lr_scheduler.step()

    @torch.no_grad()
    def __update_target_model(self) -> None:
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

    def __temporal_difference(
        self, rewards: torch.Tensor, q_values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        q_values_max_tuple = torch.max(q_values, dim=-1, keepdim=True)
        q_values_max = q_values_max_tuple.values
        td_value = rewards + ((self.gamma * q_values_max) * (1 - dones))
        return td_value

    def __decay_epsilon(self) -> None:
        if self.min_epsilon < self.epsilon:
            self.epsilon *= self.epsilon_decay_rate
        if self.min_epsilon > self.epsilon:
            self.epsilon = self.min_epsilon
        return

    def __load_random_experiences(self) -> Memory:
        random.seed(time.time())
        self.generator.manual_seed(
            random.randint(MEMORY_SHUFFLE_SEED_MIN, MEMORY_SHUFFLE_SEED_MAX)
        )

        force_sample_terminal_state = False
        if (
            random.random() < self.force_terminal_state_selection_prob
            and self.memory.has_terminal_state()
        ):
            force_sample_terminal_state = True

        local_batch_size = self.batch_size
        if force_sample_terminal_state:
            local_batch_size -= 1
        random_indices = torch.randint(
            low=0,
            high=len(self.memory) - 1,
            size=(local_batch_size,),
            generator=self.generator,
            requires_grad=False,
            device=self.device,
        ).tolist()

        if force_sample_terminal_state:
            random_indices.append(self.memory.get_random_terminal_sample_index())
        random_experience = Memory(buffer_size=self.batch_size, device=self.device)
        random_experience.append(*self.memory[random_indices])
        return random_experience
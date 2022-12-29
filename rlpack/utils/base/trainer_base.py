"""!
@package rlpack.utils.base
@brief This package implements the base classes to be used across rlpack


Currently following base classes have been implemented:
    - `Agent`: Base class for all agents, implemented as rlpack.utils.base.agent.Agent.
    - `TrainerBase`: BAse class for all trainers, implemented as rlpack.utils.base.trainer.Trainer.

Following packages are part of utils:
    - `registers`: A package for base classes for registers. Implemented in rlpack.utils.base.registers.
"""


import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from rlpack import SummaryWriter, pytorch_distributed
from rlpack.utils import GenericFuncSignature
from rlpack.utils.base.agent import Agent


class TrainerBase(ABC):
    """
    This class is the abstract base class of all trainer classes which implements methods to train an agent. This
    class implements basic utilities useful for all trainer classes.
    """

    def __init__(
        self,
        mode: str,
        agent: Agent,
        env: gym.Env,
        save_path: str,
        summary_writer: Union[SummaryWriter, None] = None,
        is_distributed: bool = False,
    ):
        """
        The initializer method (class constructor) for TrainerBase.
        @param mode: bool: Current mode of operation of Trainer (training/evaluation).
        @param agent: Agent: The RLPack Agent to be trained or evaluated.
        @param env: gym.Env: The gym environment to be used for training or evaluated.
        @param save_path: str: The path where agent and logs are saved.
        @param summary_writer: Union[SummaryWriter, None]: An instance of SummaryWriter for tensorboard
            logging. Default: None
        @param is_distributed: bool: Flag indicating if current setting is distributed or not. If set to True and
            is not distributed setting (i.e. dist.init_process_group) has not been called yet, may raise an
            error. Default: False
        """
        ## The mode in which trainer will be run (training or evaluation). @I{# noqa: E266}
        self.mode = mode
        ## The input RLPack agent to be run. @I{# noqa: E266}
        self.agent = agent
        ## The gym environment on which the agent will run. @I{# noqa: E266}
        self.env = env
        ## The input `summary_writer` for tensorboard logging. @I{# noqa: E266}
        self.summary_writer = summary_writer
        ## The input `is_distributed` indicating if TrainerBase is launched in multiprocessing setting. @I{# noqa: E266}
        self.is_distributed = is_distributed
        ## The python logger for logging metrics. This is saved in input `save_path` as trainer.log. @I{# noqa: E266}
        self.py_logger = self._configure_logger(save_path)
        ## The list of rewards at each timestep. @I{# noqa: E266}
        self.rewards = list()
        ## The cumulative rewards after each episode. @I{# noqa: E266}
        self.cumulative_rewards = list()
        ## The basic quantities to log. @I{# noqa: E266}
        self._log_quantities_base = (
            "return",
            "reward",
        )
        ## The quantities from agents to log. @I{# noqa: E266}
        self._log_quantities_agent = (
            "epsilon",
            "loss",
            "variance_value",
            "prioritization_params",
        )
        ## The prioritization quantities from agents to log. @I{# noqa: E266}
        self._log_quantities_prioritization = (
            "alpha",
            "beta",
        )
        ## The possible names for train mode. @I{# noqa: E266}
        self._possible_train_names = ("train", "training")
        ## The possible names for evaluation mode. @I{# noqa: E266}
        self._possible_eval_names = ("eval", "evaluate", "evaluation")
        ## The quantities that can be logged in current session. @I{# noqa: E266}
        self._loggable_quantities = self.get_loggable_quantities()
        ## The prioritization quantities that can be logged in current session. @I{# noqa: E266}
        self._loggable_prioritization_quantities = (
            self.get_loggable_prioritization_quantities()
        )
        ## The best cumulative reward acquired so far. @I{# noqa: E266}
        self._best_cumulative_reward_value = None

    @abstractmethod
    def train_agent(self, *args, **kwargs) -> Any:
        """
        Abstract method to train agents
        @param args: Positional arguments for training agents.
        @param kwargs: Keyword arguments for training agents.
        @return: Any: Return any object.
        """
        pass

    @abstractmethod
    def evaluate_agent(self, *args, **kwargs) -> Any:
        """
        Abstract method to evaluating agents
        @param args: Positional arguments for evaluating agents.
        @param kwargs: Keyword arguments for evaluating agents.
        @return: Any: Return any object.
        """
        pass

    def get_loggable_quantities(self) -> List[str]:
        """
        Gets the list of quantities that can be logged in current session given the agent by checking
        if loggable quantities are present in the agent's attribute
        @return List[str]: list of quantities that can be logged in current session given the agent.
        """
        loggable = [
            quantity
            for quantity in self._log_quantities_agent
            if hasattr(self.agent, quantity)
        ]
        return loggable

    def get_loggable_prioritization_quantities(self) -> List[str]:
        """
        Gets the list of prioritization quantities that can be logged in current session given the agent by checking
        if prioritization quantities are present in the agent's "prioritization_params" attribute's key.
        @return List[str]: list of quantities that can be logged in current session given the agent.
        """
        if "prioritization_params" in self._loggable_quantities:
            loggable_prioritization_params = [
                prioritization_param
                for prioritization_param in self._log_quantities_prioritization
                if prioritization_param
                in getattr(self.agent, "prioritization_params").keys()
            ]
            return loggable_prioritization_params
        return list()

    def get_loggable_quantities_by_current_value(
        self,
    ) -> Dict[str, Union[int, float, List[float]]]:
        """
        Obtains the current value of loggable quantities from agent.
        @return Dict[str, Union[int, float, List[float]]]: The dictionary of current values for each loggable
            quantity.
        """
        loggable_quantities_by_current_value = {
            loggable: getattr(self.agent, loggable)
            for loggable in self._loggable_quantities
        }
        return loggable_quantities_by_current_value

    def get_loggable_prioritization_quantities_by_current_value(
        self,
    ) -> Dict[str, Union[int, float]]:
        """
        Obtains the current value of loggable prioritization quantities from agent.
        @return Dict[str, Union[int, float, List[float]]]: The dictionary of current values for each loggable
            prioritization quantity.
        """
        if not len(self._loggable_prioritization_quantities) > 0:
            return dict()
        loggable_prioritization_quantities_by_current_value = {
            loggable: getattr(self.agent, "prioritization_params")[loggable]
            for loggable in self._loggable_prioritization_quantities
        }
        return loggable_prioritization_quantities_by_current_value

    def log_agent_info_with_py_logger(self, episode: int) -> None:
        """
        Adds agent's loggable quantities to Python logger.
        @param episode: int: The current episode for which logging is being done.
        """
        loggable_by_value = {
            **self.get_loggable_quantities_by_current_value(),
            **self.get_loggable_prioritization_quantities_by_current_value(),
        }
        for key, value in loggable_by_value.items():
            message = f"{key} at episode {episode}: "
            if isinstance(value, list):
                value = self._list_mean(value)
                message = f"Mean {message}"
            if value is None:
                continue
            self.py_logger.info(f"{message}{value}")
        return

    def log_agent_info_with_summary_writer(self, episode: int) -> None:
        """
        Adds agent's loggable quantities to Tensorboard logger.
        @param episode: int: The current episode for which logging is being done.
        """
        if self.summary_writer is None:
            return
        loggable_by_value = {
            **self.get_loggable_quantities_by_current_value(),
            **self.get_loggable_prioritization_quantities_by_current_value(),
        }
        for key, value in loggable_by_value.items():
            if isinstance(value, list):
                value = self._list_mean(value)
            if value is None:
                continue
            self.summary_writer.add_scalar(
                tag=f"{self.mode}/{key}", scalar_value=value, global_step=episode
            )
        return

    def log_cumulative_rewards_with_py_logger(self, episode: int) -> None:
        """
        Computes average cumulative rewards accumulated so far and logs them with Python logger
        @param episode: int:  The current episode for which logging is being done.
        """
        mean_reward = self._list_mean(self.cumulative_rewards)
        self.py_logger.info(
            f"Mean Cumulative reward at episode {episode}: {mean_reward}"
        )

    def log_reward_with_summary_writer(
        self, reward: float, episode: int, timestep: int
    ) -> None:
        """
        Computes average cumulative rewards accumulated so far and logs them.
        @param reward: float: The reward obtained at the given timestep.
        @param episode: int:  The current episode for which logging is being done.
        @param timestep: int: The current timestep of the given episode.
        """
        if self.summary_writer is None:
            return
        self.summary_writer.add_scalar(
            tag=f"{self.mode}/reward",
            scalar_value=reward,
            global_step=episode * timestep,
        )

    def log_cumulative_rewards_with_summary_writer(self, episode: int) -> None:
        """
        Computes average cumulative rewards accumulated so far and logs them with Tensorboard logger.
        @param episode: int:  The current episode for which logging is being done.
        """
        if self.summary_writer is None:
            return
        mean_reward = self._list_mean(self.rewards)
        self.summary_writer.add_scalar(
            tag=f"{self.mode}/cumulative_rewards",
            scalar_value=mean_reward,
            global_step=episode,
        )

    def log_returns_with_py_logger(self, episode: int) -> None:
        """
        Computes average returns and logs them with Python logger.
        @param episode: int:  The current episode for which logging is being done.
        """
        mean_returns = self._list_mean(self._compute_returns())
        self.py_logger.info(f"Mean Returns at episode {episode}: {mean_returns}")

    def log_returns_with_summary_writer(self, episode: int) -> None:
        """
        Computes average returns and logs them with Tensorboard logger.
        @param episode: int:  The current episode for which logging is being done.
        """
        if self.summary_writer is None:
            return
        mean_returns = self._list_mean(self._compute_returns())
        self.summary_writer.add_scalar(
            tag=f"{self.mode}/returns", scalar_value=mean_returns, global_step=episode
        )

    def header_line(self) -> None:
        """
        Logs header line for block separation.
        """
        self.py_logger.info(f"\n{'~' * 60}")

    def append_reward(self, reward: float) -> None:
        """
        Append the reward to current list of rewards (TrainerBase.rewards). Ideally this must be called at each
        timestep.
        @param reward: float: The current reward
        """
        self.rewards.append(reward)

    def fill_cumulative_reward(self) -> None:
        """
        Populates the cumulative rewards (TrainerBase.cumulative_rewards) list by computing cumulative rewards at
        the moment. This will sum the accumulated rewards and hence ideally must be called after each episode.
        Post episode, Trainer.clear_rewards method can be called to clear the accumulated rewards
        """
        self.cumulative_rewards.append(sum(self.rewards))

    def clear_rewards(self) -> None:
        """
        Clear the rewards accumulated so far
        """
        self.rewards.clear()

    def clear_cumulative_rewards(self) -> None:
        """
        Clears the cumulative rewards accumulated so far.
        """
        self.cumulative_rewards.clear()

    def clear_agent_loss(self) -> None:
        """
        Clears the agent's loss accumulated so far
        """
        self.agent.loss.clear()

    def save_agent_with_custom_suffix(self, custom_suffix: str) -> None:
        """
        Saves the agent with given custom suffix if obtained cumulative reward of the agent is found to be best so
        far.
        @param custom_suffix: str: The custom suffix to add to agent's name while saving.
        """
        updated_flag = self._update_best_reward_value()
        if updated_flag:
            self.agent.save(custom_suffix)

    def save_agent(self) -> None:
        """
        Call to `agent.save` method. This method executes the save method atomically with only process 0 when
        there is a multiprocessing or distributed setting.
        """
        self._execute_func_atomically_(self.is_distributed, self.agent.save)

    def is_eval(self) -> bool:
        """
        Check if environment is to be run in evaluation mode or not.
        @return bool: True if evaluation mode is set.
        """
        return self.mode in self._possible_eval_names

    def is_train(self) -> bool:
        """
        Check if environment is to be run in training mode or not.
        @return bool: True if training mode is set.
        """
        return self.mode in self._possible_train_names

    def _update_best_reward_value(self) -> bool:
        """
        Updates the current best cumulative reward value (TrainerBase._best_cumulative_reward_value) if new
        cumulative reward is found to be higher than current value.
        @return bool: Flag indicating if update has occured or not.
        """
        mean_cumulative_rewards = self._list_mean(self.cumulative_rewards)
        updated_flag = False
        if self._best_cumulative_reward_value is None:
            self._best_cumulative_reward_value = mean_cumulative_rewards
            updated_flag = True
        if self._best_cumulative_reward_value < mean_cumulative_rewards:
            self._best_cumulative_reward_value = mean_cumulative_rewards
            updated_flag = True
        return updated_flag

    def _compute_returns(self) -> List[float]:
        """
        Computes returns for accumulated rewards.
        @return List[float]: The list of returns at each timestep.
        """
        return self._compute_returns_helper(
            rewards=self.rewards, gamma=self.agent.gamma
        )

    @staticmethod
    def _execute_func_atomically_(
        is_distributed: bool, func: GenericFuncSignature, proc: int = 0, *args, **kwargs
    ) -> None:
        """
        Helper function to execute the given function atomically with only one given process.
        @param is_distributed: Flag indicating if current setting is distributed or not. If set to True and is not
            distributed setting (i.e. dist.init_process_group) has not been called yet, will raise an error.
        @param func: GenericFuncSignature: The function to be executed. This function must be void and
            should not return anything.
        @param proc: int: The process id (local rank) which will execute the `func`. Default: 0
        @param args: Other positional arguments for `func`.
        @param kwargs: Other keyword arguments for `fund`.
        """
        if is_distributed:
            process_rank = pytorch_distributed.get_rank()
            if process_rank == proc:
                func(*args, **kwargs)
        else:
            func(*args, **kwargs)

    @staticmethod
    def _compute_returns_helper(rewards, gamma) -> List[float]:
        """
        Helper function to compute returns for given rewards.
        @return List[float]: The list of returns at each timestep.
        """
        total_rewards = len(rewards)
        returns = [0] * total_rewards
        r_ = 0
        for idx in range(total_rewards):
            idx = total_rewards - idx - 1
            r_ = rewards[idx] + gamma * r_
            returns[idx] = r_
        return returns

    @staticmethod
    def _list_mean(x: List[Union[float, int]]) -> Union[None, float]:
        """
        This function computes the mean of the input list.
        @param x: List[Union[float, int]]: The list for which mean is to be computed
        @return Union[None, float]: The mean value.
        """
        if x:
            return sum(x) / len(x)
        return None

    @staticmethod
    def _reshape_func_default(
        x: np.ndarray, shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        This is the default reshape function. If `new_shape` has been set in config, input states are reshaped
            to new shapes, else returns the input as it is. Default behavior is not perform any reshaping.
        @param x: np.ndarray: The input numpy array to reshape.
        @param shape: Optional[Tuple[int, ...]]: The new shape to which we want states to be reshaped. Default: None.
        @return np.ndarray: The reshaped (or unchanged) array.
        """
        if shape is not None:
            x = np.reshape(x, newshape=shape)
        return x

    @staticmethod
    def _configure_logger(save_path: str) -> logging.Logger:
        """
        @param save_path: str: The path to save log file. A file named `trainer.log` will be created and metrics
        will be logged into the file.
        @return logging.Logger: The logger instance for logging
        """
        logging.basicConfig(
            filename=os.path.join(save_path, "trainer.log"), level=logging.INFO
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

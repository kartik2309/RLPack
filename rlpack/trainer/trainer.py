"""!
@package rlpack.trainer
@brief This package implements the training methods for agents.


This package implements Trainer class as rlpack.trainer.trainer.Trainer. This class couples the
agent we selected with the environment we pass/select. It provides basic methods for training and evaluating an agent.
This class also logs rewards and other metrics on the screen.
"""


import logging
from typing import Callable, Optional, Tuple, Union

import gym
import numpy as np

from rlpack import SummaryWriter
from rlpack.utils.base.agent import Agent
from rlpack.utils.base.trainer_base import TrainerBase


class Trainer(TrainerBase):
    """
    This class is a generic class to train or evaluate any agent in any environment. This class provides necessary
    framework to perform experimentation and analyse the results. It inherits from rlpack.utils.base.TrainerBase.
    """

    def __init__(
        self,
        mode: str,
        agent: Agent,
        env: gym.Env,
        save_path: str,
        num_episodes: int,
        max_timesteps: Union[int, None] = None,
        custom_suffix: Union[str, None] = None,
        reshape_func: Optional[
            Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]
        ] = None,
        new_shape: Union[Tuple[int, ...], None] = None,
        is_distributed: bool = False,
        summary_writer: Union[SummaryWriter, None] = None,
    ):
        """
        Initialization method for Trainer.
        @param mode: bool: Current mode of operation of Trainer (training/evaluation).
        @param agent: Agent: The RLPack Agent to be trained or evaluated.
        @param env: gym.Env: The gym environment to be used for training or evaluated.
        @param save_path: str: The path where agent and logs are saved.
        @param num_episodes: int: The number of episodes to run for training or evaluation.
        @param max_timesteps: Union[str, None]: The maximum number of timesteps to be run on the environment. This
            will override the gym environment's TimeWrapper if passed. Default: None.
        @param custom_suffix: str: The custom suffix to add to agent's name while saving for the agent that
            receives the highest cumulative rewards observed thus far.
        @param reshape_func: Optional[Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]]: The function to reshape
            the input states. Default: None; Default behavior is to not do any reshaping unless `new_shape` is passed.
        @param new_shape: Union[Tuple[int, ...], None]: The new shape for observations (states) using `reshape_func`.
            If reshape_func was passed None and a valid `new_shape` was passed, default reshape func
            TrainerBase._reshape_func_default is used to perform reshaping.
        @param is_distributed: bool: Flag indicating if current setting is distributed or not. If set to True and
            is not distributed setting (i.e. dist.init_process_group) has not been called yet, may raise an
            error. Default: False
        @param summary_writer: SummaryWriter: The instance of SummaryWriter to perform tensorboard
            logging. Default: None
        """

        super(Trainer, self).__init__(
            mode=mode,
            agent=agent,
            env=env,
            save_path=save_path,
            summary_writer=summary_writer,
            is_distributed=is_distributed,
        )
        self.num_episodes = num_episodes
        self.custom_suffix = custom_suffix
        ## The input reshape function for states. @I{# noqa: E266}
        self.reshape_func = (
            reshape_func if reshape_func is not None else self._reshape_func_default
        )
        # Check input `reshape_func` and set attributes accordingly.
        if not isinstance(new_shape, (list, type(None))):
            raise TypeError(
                f"`new_shape` must be a {list} or {tuple} of new shape or {type(None)}"
            )
        ## The new shape requested in config to be used with @ref reshape_func. @I{# noqa: E266}
        self.new_shape = new_shape
        self.max_timesteps = max_timesteps
        if max_timesteps is not None:
            self.env.spec.max_episode_steps = max_timesteps
        else:
            self.max_timesteps = self.env.spec.max_episode_steps

    def train_agent(
        self,
        metrics_logging_frequency: int,
        render: bool = False,
        load: bool = False,
    ) -> None:
        """
        Method to train the agent in the specified environment. `mode` must be set as mode='train' to run evaluation
        during class initialization.
        @param metrics_logging_frequency: int: The logging frequency for rewards.
        @param render: bool: Indicates if we wish to render the environment (in animation). Default: False.
        @param load: bool: Indicates weather to load a previously saved model or train a new one. If set true,
            config must be `save_path` or set or environment variable SAVE_PATH must be set.
        """
        if not self.is_train():
            logging.warning("Currently operating in Evaluation Mode")
            return
        if load:
            self.agent.load(self.custom_suffix)
        # Start episodic loop
        for ep in range(1, self.num_episodes + 1):
            observation_current, _ = self.env.reset()
            action = self.env.action_space.sample()
            done = False
            # Start timestep loop
            for timestep in range(1, self.max_timesteps + 1):
                if render:
                    self.env.render()
                observation_next, reward, terminated, truncated, info = self.env.step(
                    action=action
                )
                if terminated or truncated:
                    done = True
                state_current = self.reshape_func(observation_current, self.new_shape)
                state_next = self.reshape_func(observation_next, self.new_shape)
                action = self.agent.train(
                    state_current=state_current,
                    state_next=state_next,
                    action=action,
                    reward=reward,
                    done=done,
                )
                self.log_reward_with_summary_writer(
                    reward=reward, episode=ep, timestep=timestep
                )
                self.append_reward(reward)
                observation_current = observation_next
                # Break the loop once done
                if done:
                    break
            self.fill_cumulative_reward()
            self.log_cumulative_rewards_with_summary_writer(episode=ep)
            self.log_returns_with_summary_writer(episode=ep)
            self.log_agent_info_with_summary_writer(episode=ep)
            self.clear_rewards()
            if ep % metrics_logging_frequency == 0:
                # If mean reward obtained is higher than moving average of rewards so far, save the agent with
                # custom suffix. If no custom suffix is present, save with the suffix `_best`.
                self.save_agent_with_custom_suffix(self.custom_suffix)
                self.header_line()
                self.log_cumulative_rewards_with_py_logger(ep)
                self.log_agent_info_with_py_logger(ep)
                self.clear_cumulative_rewards()
                self.clear_agent_loss()
        self.env.close()

    def evaluate_agent(self) -> None:
        """
        Method to evaluate a trained model. This method renders the environment and loads the model from
            `save_path`.
        `mode` must be set as mode='eval' to run evaluation during class initialization.
        """
        if not self.is_eval():
            logging.warning("Currently operating in Training Mode")
            return
        # Load agent's necessary objects (models, states etc.)
        self.agent.load(self.custom_suffix)
        # Start episodes.
        for ep in range(1, self.num_episodes + 1):
            observation, _ = self.env.reset()
            done = False
            # Start episodic loop
            for timestep in range(1, self.max_timesteps + 1):
                self.env.render()
                observation = self.reshape_func(observation, self.new_shape)
                action = self.agent.policy(observation)
                observation_next, reward, terminated, truncated, info = self.env.step(
                    action=action
                )
                if terminated or truncated:
                    done = True
                self.log_reward_with_summary_writer(
                    reward=reward, episode=ep, timestep=timestep
                )
                self.append_reward(reward)
                if done:
                    break
            self.fill_cumulative_reward()
            self.header_line()
            self.log_cumulative_rewards_with_summary_writer(episode=ep)
            self.log_returns_with_summary_writer(episode=ep)
            self.log_agent_info_with_summary_writer(episode=ep)
            self.log_returns_with_py_logger(ep)
            self.log_cumulative_rewards_with_py_logger(ep)
            self.clear_rewards()
        self.env.close()

"""!
@package rlpack.environments
@brief This package implements the gym environment to couple it with selected environment.


This package implements Environments class as rlpack.environments.environments.Environments. This class couples the
agent we selected with the environment we pass/select. It provides basic methods for training and evaluating an agent.
This class also logs rewards and other metrics on the screen.
"""


import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np

from rlpack import dist
from rlpack.utils.base.agent import Agent


class Environments:
    """
    This class is a generic class to train any agent in any environment.
    """

    def __init__(
        self,
        agent: Agent,
        config: Dict[str, Any],
        reshape_func: Optional[
            Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]
        ] = None,
    ):
        """
        @param agent: Agent: The agent to be trained and/or evaluated in the environment specified in `config`.
        @param config: Dict[str, Any]: The configuration setting for experiment.
        @param reshape_func: Optional[Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]]: The function to reshape
            the input states. Default: None. Default behavior is to not do any reshaping.
        """
        ## The input RLPack agent to be run. @I{# noqa: E266}
        self.agent = agent
        ## The input config for setup. @I{# noqa: E266}
        self.config = config
        ## The input reshape function for states. @I{# noqa: E266}
        if reshape_func is None:
            self.reshape_func = self.__reshape_func_default
        else:
            self.reshape_func = reshape_func
        new_shape = self.config.get("new_shape")
        if not isinstance(new_shape, list):
            raise TypeError("`new_shape` must be a list of new shape")
        ## The new shape requested in config to be used with @ref reshape_func. @I{# noqa: E266}
        self.new_shape = tuple(new_shape) if new_shape is not None else None
        render_mode = None
        if self.is_eval() and config["render"]:
            render_mode = "human"
        ## The gym environment on which the agent will run. @I{# noqa: E266}
        if self.config.get("env") is None:
            assert self.config.get("env_name") is not None, (
                "Either `env` (for gym environment) or `env_name` (for env name registered with gym)"
                " must be passed in config"
            )
            self.env = gym.make(self.config["env_name"], render_mode=render_mode)
        else:
            self.env = config["env"]
        self.env.spec.max_episode_steps = self.config["max_timesteps"]

    def train_agent(
        self,
        render: bool = False,
        load: bool = False,
        plot: bool = False,
        verbose: int = -1,
        distributed_mode: bool = False,
    ) -> None:
        """
        Method to train the agent in the specified environment.
        @param render: bool: Indicates if we wish to render the environment (in animation). Default: False.
        @param load: bool: Indicates weather to load a previously saved model or train a new one. If set true,
            config must be `save_path` or set or environment variable SAVE_PATH must be set.
        @param plot: bool: Indicates if to plot the training progress. If set True, rewards and episodes are
            recorded and plot is saved in `save_path`.
        @param verbose: bool: Indicates the verbose level. Refer notes for more details. This also refers to values
            logged on screen. If you want to disable the logging on screen, set logging level to WARNING. Default: -1
        config must have set mode='train' to run evaluation.
        Rewards are logged on console every `reward_logging_frequency` set in the console.

        **Notes**


        Verbose levels:
            - -1: Log everything.
            - 0: Log episode wise rewards.
            - 1: Log model level losses.
            - 2: Log Agent specific values.
        """
        assert -2 <= verbose <= 2, "Argument `verbose` must be in range [-1, 2]."
        if not self.is_train():
            logging.info("Currently operating in Evaluation Mode")
            return
        if load:
            self.agent.load(self.config.get("custom_suffix", "_best"))
        else:
            if distributed_mode:
                if dist.get_rank() == 0:
                    self.__remove_log_file()
            else:
                self.__remove_log_file()
        log = list()
        highest_mv_avg_reward, timestep = 0.0, 0
        rewards_collector = {k: list() for k in range(self.config["num_episodes"])}
        rewards = list()
        for ep in range(self.config["num_episodes"]):
            observation_current, _ = self.env.reset()
            action = self.env.action_space.sample()
            scores = 0
            for timestep in range(self.config["max_timesteps"]):
                if render:
                    self.env.render()
                observation_next, reward, done, info, _ = self.env.step(action=action)
                action = self.agent.train(
                    state_current=self.reshape_func(
                        observation_current, self.new_shape
                    ),
                    state_next=self.reshape_func(observation_next, self.new_shape),
                    action=action,
                    reward=reward,
                    done=done,
                )
                scores += reward
                if plot:
                    rewards_collector[ep].append(reward)
                observation_current = observation_next
                if done:
                    break
            if timestep == self.config["max_timesteps"]:
                logging.info(
                    f"Maximum timesteps of {timestep} reached in the episode {ep}"
                )
            rewards.append(scores)
            if ep % self.config["reward_logging_frequency"] == 0:
                reward_log_message = "~" * 60
                # Log Mean Reward in the episode cycle
                if verbose <= 0:
                    mean_reward = self.__list_mean(rewards)
                    reward_log_message = (
                        f"Average Reward after {ep} episodes: {mean_reward}"
                    )
                    if distributed_mode:
                        reward_log_message = (
                            f"{reward_log_message} from process {dist.get_rank()}"
                        )
                    logging.info(reward_log_message)
                    log.append(f"{reward_log_message}\n")
                    if highest_mv_avg_reward < mean_reward:
                        self.agent.save(
                            custom_name_suffix=self.config.get("suffix", "_best")
                        )
                        highest_mv_avg_reward = mean_reward
                if verbose <= 1:
                    # Log Mean Loss in the episode cycle
                    mean_loss = self.__list_mean(self.agent.loss)
                    if len(self.agent.loss) > 0:
                        log_mean_message = (
                            f"Average Loss after {ep} episodes: {mean_loss}"
                        )
                        if distributed_mode:
                            log_mean_message = (
                                f"{log_mean_message} from process {dist.get_rank()}"
                            )
                        logging.info(log_mean_message)
                        log.append(f"{log_mean_message}\n")
                if verbose <= 2:
                    # Log current epsilon value
                    if hasattr(self.agent, "epsilon"):
                        log_epsilon_message = (
                            f"Epsilon after {ep} episodes: {self.agent.epsilon}"
                        )
                        if distributed_mode:
                            log_epsilon_message = (
                                f"{log_epsilon_message} from process {dist.get_rank()}"
                            )
                        logging.info(log_epsilon_message)
                        log.append(f"{log_epsilon_message}\n")
                    # Log current alpha and beta values - for prioritized relay (DQN)
                    if hasattr(self.agent, "prioritization_params"):
                        if hasattr(self.agent, "prioritization_params"):
                            if "alpha" in self.agent.prioritization_params.keys():
                                log_alpha_message = f"Alpha after {ep} episodes: {self.agent.prioritization_params['alpha']}"
                                if distributed_mode:
                                    log_alpha_message = f"{log_alpha_message} from process {dist.get_rank()}"
                                logging.info(log_alpha_message)
                                log.append(f"{log_alpha_message}\n")
                            if "beta" in self.agent.prioritization_params.keys():
                                log_beta_message = f"Beta after {ep} episodes: {self.agent.prioritization_params['beta']}"
                                if distributed_mode:
                                    log_beta_message = f"{log_beta_message} from process {dist.get_rank()}"
                                logging.info(log_beta_message)
                                log.append(f"{log_beta_message}\n")
                    if not distributed_mode:
                        closing_message = f"{'~' * len(reward_log_message)}\n"
                    else:
                        length = int(len(reward_log_message) / 2)
                        closing_message = (
                            f"{'~' * length} Process {dist.get_rank()} {'~' * length}\n"
                        )
                    logging.info(closing_message)
                    log.append(f"{closing_message}\n")
                    if not distributed_mode:
                        self.__write_log_file(log)
                    else:
                        if dist.get_rank() == 0:
                            self.__write_log_file(log)
                rewards.clear()
        self.env.close()
        if not distributed_mode:
            self.agent.save()
        else:
            if dist.get_rank() == 0:
                self.agent.save()
        if plot:
            if not distributed_mode:
                self.__generate_plot(rewards_collector)
            else:
                if dist.get_rank() == 0:
                    self.__generate_plot(rewards_collector)

    def evaluate_agent(self) -> None:
        """
        Method to evaluate a trained model. This method renders the environment and loads the model from
            `save_path`.
        config must have set mode='eval' to run evaluation.
        """
        if not self.is_eval():
            logging.info("Currently operating in Training Mode")
            return

        # Load agent's necessary objects (models, states etc.)
        self.agent.load(self.config.get("custom_suffix", "_best"))
        # Temporarily save epsilon before setting it 0.0
        epsilon = self.agent.epsilon
        rewards = list()
        self.agent.epsilon, timestep, ep, score = 0.0, 0, 0, 0
        for ep in range(self.config["num_episodes"]):
            observation, _ = self.env.reset()
            score = 0
            for timestep in range(self.config["max_timesteps"]):
                action = self.agent.policy(
                    self.reshape_func(observation, self.new_shape)
                )
                observation, reward, done, info, _ = self.env.step(action=action)
                score += reward
                if done:
                    break
            rewards.append(score)
        if timestep == self.config["max_timesteps"]:
            logging.info("Max timesteps was reached!")
        if ep < 1:
            # When only single episode was performed and evaluated.
            logging.info(
                f"Total Reward after {timestep} timesteps: {score}",
            )
        else:
            # When more than one episode was performed and evaluated.
            logging.info(
                f'Average Rewards after {self.config["num_episodes"]} episodes: {self.__list_mean(rewards)}'
            )
        # Restore epsilon value of the agent.
        self.agent.epsilon = epsilon
        self.env.close()

    def is_eval(self) -> bool:
        """
        Check if environment is to be run in evaluation mode or not.
        @return bool: True if evaluation mode is set.
        """
        possible_eval_names = ("eval", "evaluate", "evaluation")
        return self.config["mode"] in possible_eval_names

    def is_train(self) -> bool:
        """
        Check if environment is to be run in training mode or not.
        @return bool: True if training mode is set.
        """
        possible_train_names = ("train", "training")
        return self.config["mode"] in possible_train_names

    def __remove_log_file(self):
        if os.path.isfile(
            os.path.join(self.config["agent_args"]["save_path"], "log.txt")
        ):
            os.remove(os.path.join(self.config["agent_args"]["save_path"], "log.txt"))

    def __write_log_file(self, log):
        with open(
            os.path.join(self.config["agent_args"]["save_path"], "log.txt"),
            "a+",
        ) as f:
            for line in log:
                f.write(line)

    def __generate_plot(self, rewards_collector):
        rewards_to_plot = [
            sum(rewards_collector[k]) / len(rewards_collector[k])
            for k in range(self.config["num_episodes"])
        ]
        plt.plot(range(self.config["num_episodes"]), rewards_to_plot)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Lunar Lander - Rewards vs. Episodes")
        plt.savefig(os.path.join(self.agent.save_path, "EpisodeVsReward.jpeg"))

    @staticmethod
    def __reshape_func_default(
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
    def __list_mean(x: List[Union[float, int]]) -> Union[None, float]:
        """
        This function computes the mean of the input list.
        @param x: List[Union[float, int]]: The list for which mean is to be computed
        @return Union[None, float]: The mean value.
        """
        if x:
            return sum(x) / len(x)
        return None

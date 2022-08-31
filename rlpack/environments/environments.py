import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np

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
        @:param agent (Agent): The agent to be trained and/or evaluated in the environment specified in `config`.
        @:param config (Dict[str, Any]): The configuration setting for experiment.
        @:param reshape_func (Optional[Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]]): The function to reshape
            the input states. Default: None. Default behavior is to not do any reshaping.
        """
        self.agent = agent
        self.config = config
        if reshape_func is None:
            self.reshape_func = self.__reshape_func_default
        else:
            self.reshape_func = reshape_func
        self.new_shape = tuple(self.config.get("new_shape"))
        render_mode = None
        if self.is_eval():
            render_mode = "human"
        self.env = gym.make(
            self.config["env_name"], new_step_api=True, render_mode=render_mode
        )
        self.env.spec.max_episode_steps = self.config["max_timesteps"]

    def train_agent(
        self, render: bool = False, load: bool = False, plot: bool = False
    ) -> None:
        """
        Method to train the agent in the specified environment.

        @:param render (bool): Indicates if we wish to render the environment (in animation). Default: False
        @:param load (bool): Indicates weather to load a previously saved model or train a new one. If set true,
            config must be `save_path` or set or environment variable SAVE_PATH must be set.
        @:param plot (bool): Indicates if to plot the training progress. If set True, rewards and episodes are
            recorded and plot is saved in `save_path`.

        config must have set mode='train' to run evaluation.
        Rewards are logged on console every `reward_logging_frequency` set in the console.
        """
        if not self.is_train():
            logging.info("Currently operating in Evaluation Mode")
            return
        if load:
            self.agent.load()
        highest_mv_avg_reward, timestep = 0.0, 0
        rewards_collector = {k: list() for k in range(self.config["num_episodes"])}
        rewards = list()
        for ep in range(self.config["num_episodes"]):
            observation_current = self.env.reset()
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
                # Log Mean Reward in the episode cycle
                mean_reward = self.__list_mean(rewards)
                reward_log_message = (
                    f"Average Reward after {ep} episodes: {mean_reward}"
                )
                logging.info(reward_log_message)
                if highest_mv_avg_reward < mean_reward:
                    self.agent.save(
                        custom_name_suffix=f'_{self.config.get("suffix", "best")}'
                    )
                    highest_mv_avg_reward = mean_reward
                # Log Mean Loss in the episode cycle
                mean_loss = self.__list_mean(self.agent.loss)
                if len(self.agent.loss) > 0:
                    logging.info(f"Average Loss after {ep} episodes: {mean_loss}")
                logging.info(f"{'~' * len(reward_log_message)}\n")
                rewards.clear()
        self.env.close()
        self.agent.save()
        if plot:
            rewards_to_plot = [
                sum(rewards_collector[k]) / len(rewards_collector[k])
                for k in range(self.config["num_episodes"])
            ]
            plt.plot(range(self.config["num_episodes"]), rewards_to_plot)
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.title("Lunar Lander - Rewards vs. Episodes")
            plt.savefig(os.path.join(self.agent.save_path, "EpisodeVsReward.jpeg"))

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
        self.agent.load()
        # Temporarily save epsilon before setting it 0.0
        epsilon = self.agent.epsilon
        rewards = list()
        self.agent.epsilon, timestep, score, ep = 0.0, 0, 0, 0
        observation = self.env.reset()
        for ep in range(self.config["num_episodes"]):
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
        @:returns (bool): True if evaluation mode is set.
        """
        possible_eval_names = ("eval", "evaluate", "evaluation")
        return self.config["mode"] in possible_eval_names

    def is_train(self) -> bool:
        """
        Check if environment is to be run in training mode or not.
        @:returns (bool): True if training mode is set.
        """
        possible_train_names = ("train", "training")
        return self.config["mode"] in possible_train_names

    @staticmethod
    def __reshape_func_default(
        x: np.ndarray, shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        This is the default reshape function. If `new_shape` has been set in config, input states are reshaped
            to new shapes, else returns the input as it is. Default behavior is not perform any reshaping.

        @:param x (np.ndarray): The input numpy array to reshape.
        @:param shape (Optional[Tuple[int, ...]]): The new shape to which we want states to be reshaped. Default: None
        @:return (np.ndarray): The reshaped (or unchanged) array.
        """
        if shape is not None:
            x = np.reshape(x, newshape=shape)
        return x

    @staticmethod
    def __list_mean(x: List[Union[float, int]]) -> Union[None, float]:
        """
        This function computes the mean of the input list.
        @:param x (List[Union[float, int]]): The list for which mean is to be computed
        @:return (Union[None, float]): The mean value.
        """
        if x:
            return sum(x) / len(x)
        return None

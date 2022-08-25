import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import matplotlib.pyplot as plt
import numpy as np

from rlpack.utils.base.agent import Agent


class Environments:
    def __init__(
        self,
        agent: Agent,
        config: Dict[str, Any],
        reshape_func: Optional[Callable] = None,
    ):
        self.agent = agent
        self.config = config

        if reshape_func is None:
            self.reshape_func = self.__reshape_func_default
        else:
            self.reshape_func = reshape_func

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
        if not self.is_train():
            logging.info("Currently operating in Evaluation Mode")
            return
        if load:
            self.agent.load()

        timestep = 0
        highest_mv_avg_reward = 0
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
                    state_current=self.reshape_func(observation_current),
                    state_next=self.reshape_func(observation_next),
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
                    self.agent.save(custom_name_suffix="_best")
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
        if not self.is_eval():
            logging.info("Currently operating in Training Mode")
            return
        self.agent.load()
        epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        timestep = 0
        score = 0
        observation = self.env.reset()
        for timestep in range(self.config["max_timesteps"]):
            action = self.agent.policy(self.reshape_func(observation))
            observation, reward, done, info, _ = self.env.step(action=action)
            score += reward
            if done:
                break

        if timestep == self.config["max_timesteps"]:
            logging.info("Max timesteps was reached!")
        logging.info(
            f"Total Reward after {timestep} timesteps: {score}",
        )
        self.agent.epsilon = epsilon
        self.env.close()

    def is_eval(self):
        possible_eval_names = ("eval", "evaluate", "evaluation")
        return self.config["mode"] in possible_eval_names

    def is_train(self):
        possible_train_names = ("train", "training")
        return self.config["mode"] in possible_train_names

    @staticmethod
    def __reshape_func_default(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def __list_mean(x: List[Union[float, int]]) -> Union[None, float]:
        if x:
            return sum(x) / len(x)
        return None

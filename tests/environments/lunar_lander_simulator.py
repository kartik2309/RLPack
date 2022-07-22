import gym
import os
import numpy as np
import yaml
import logging
from typing import Union, Dict, TypeVar, Any
import matplotlib.pyplot as plt
import time

ReshapeFunction = TypeVar("ReshapeFunction")
RLPackAgent = TypeVar("RLPackAgent")


class LunarLander:
    def __init__(
            self,
            agent: RLPackAgent,
            config_path: Union[str, None] = None,
            config_dict: Union[None, Dict[str, Any]] = None,
            reshape_func: Union[None, ReshapeFunction] = None,
    ):
        time.sleep(10)
        self.agent = agent
        if config_path is not None:
            if config_dict is not None:
                logging.warning(
                    "Arguments `config_path` and `config_dict` were passed. "
                    "`config_dict` will be overwritten by the file read from `config_path`! "
                    "Try using only one of them to not receive this warning."
                )
            with open(config_path) as f:
                config_dict = yaml.load(f, yaml.Loader)

        elif config_dict is not None:
            config_dict = config_dict
        else:
            raise ValueError("Either of the Arguments `config_path` or `config_dict` must be passed!")

        self.config_dict = config_dict
        if reshape_func is None:
            self.reshape_func = self.reshape_func_default
        else:
            self.reshape_func = reshape_func

        self.env = gym.make("LunarLander-v2")
        self.env.spec.max_episode_steps = self.config_dict["max_timesteps"]

        self.agent = agent(
            model_name=self.config_dict["model_name"],
            model_args=self.config_dict["model_args"],
            agent_args=self.config_dict["agent_args"],
            optimizer_args=self.config_dict["optimizer_args"],
            activation_args=self.config_dict["activation_args"],
            lr_scheduler_args=self.config_dict["lr_scheduler_args"],
            device=self.config_dict["device"],
        )

    @staticmethod
    def reshape_func_default(x: np.ndarray) -> np.ndarray:
        return x

    def train_agent(self, render: bool = False, load: bool = False, plot: bool = False) -> None:

        if not self.__is_train():
            logging.info("Currently operating in Evaluation Mode")
            return

        if load:
            self.agent.load()

        timestep = 0
        rewards_collector = {k: list() for k in range(self.config_dict["num_episodes"])}
        rewards = list()

        for ep in range(self.config_dict["num_episodes"]):
            observation_current = self.env.reset()
            action = self.env.action_space.sample()
            scores = 0

            for timestep in range(self.config_dict["max_timesteps"]):

                if render:
                    self.env.render()

                observation_next, reward, done, info = self.env.step(action=action)

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

            if timestep == self.config_dict["max_timesteps"]:
                logging.info(f"Maximum timesteps of {timestep} reached at episode {ep}")
            rewards.append(scores)

            if ep % self.config_dict["reward_print_frequency"] == 0:
                logging.info(
                    f"Average Reward after {ep} episodes: {sum(rewards) / len(rewards)}"
                )
                rewards.clear()

        if plot:
            rewards_to_plot = [
                sum(rewards_collector[k]) / len(rewards_collector[k])
                for k in range(self.config_dict["num_episodes"])
            ]

            plt.plot(range(self.config_dict["num_episodes"]), rewards_to_plot)
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.title("Lunar Lander - Rewards vs. Episodes")
            plt.savefig(os.path.join(self.agent.save_path, "EpisodeVsReward.png"))

        self.env.close()
        self.agent.save()
        self.agent.finish()

    def evaluate_agent(self) -> None:

        if not self.__is_eval():
            logging.info("Currently operating in Training Mode")
            return

        self.agent.load()
        score = 0
        timesteps = 0
        _ = self.env.reset()
        action = self.env.action_space.sample()

        for _ in range(self.config_dict["max_timesteps"]):
            self.env.render()
            observation, reward, done, info = self.env.step(action=action)
            score += reward
            timesteps += 1
            if done:
                break
            action = self.agent.policy(self.reshape_func(observation))

        if timesteps == self.config_dict["max_timesteps"]:
            logging.info("Max timesteps was reached!")
        logging.info(
            f"Total Reward after {timesteps} timesteps: {score}",
        )
        self.env.close()

    def __is_eval(self):
        possible_eval_names = ("eval", "evaluate", "evaluation")
        return self.config_dict["mode"] in possible_eval_names

    def __is_train(self):
        possible_train_names = ("train", "training")
        return self.config_dict["mode"] in possible_train_names

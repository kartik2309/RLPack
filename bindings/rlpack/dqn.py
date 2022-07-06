import sys

from .lib import RLPack

import os.path
import numpy as np
import re
from typing import Dict, Any, Optional


class DQN:
    def __init__(
            self,
            model_name: str,
            model_args: Dict[str, Any],
            agent_args: Dict[str, Any],
            optimizer_args: Dict[str, Any],
            activation_args: Optional[Dict[str, Any]] = None,
            device: str = "cpu",
    ):
        """
        :param model_name: str: The model name for the model.
        :param model_args: Dict[str, Any]: The model specific arguments.
        :param agent_args: Dict[str, Any]: The agent specific arguments.
        :param optimizer_args: Dict[str, Any]: The optimizer specific arguments.
        :param activation_args: Dict[str, Any]: The activation specific arguments.
        :param device: str: The device strings.
        """
        assert (
                model_name is not None
                and model_args is not None
                and agent_args is not None
                and optimizer_args is not None
        ), "One or more of the mandatory arguments were passed as None"

        if activation_args is None:
            activation_args = dict()

        agent_args = self.__save_path_checks(agent_args)
        agent_args = self.__save_path_correction(agent_args, model_name)

        self.get_dqn_agent = RLPack.GetDqnAgent(
            model_name, model_args, activation_args, agent_args, optimizer_args, device
        )
        self.device = device

    def train(
            self,
            state_current: np.ndarray,
            state_next: np.ndarray,
            reward: float,
            action: int,
            done: bool,
    ) -> int:

        return self.get_dqn_agent.train(
            state_current.astype(np.float32),
            state_next.astype(np.float32),
            float(reward),
            int(action),
            bool(done),
            state_current.shape,
            state_next.shape,
        )

    def policy(self, state_current: np.ndarray) -> int:
        return self.get_dqn_agent.policy(
            state_current.astype(np.float32), state_current.shape
        )

    def save(self):
        self.get_dqn_agent.save()

    def load(self):
        self.get_dqn_agent.load()

    @staticmethod
    def __save_path_checks(agent_args):
        if "save_path" not in agent_args.keys():
            save_path = os.environ.get("RLPACK_SAVE_PATH")
            if save_path is None:
                raise ValueError(
                    "Expected save_path key to be in input argument dict 'agent_args' "
                    "or set as environment variable RLPACK_SAVE_PATH"
                )
            agent_args["save_path"] = save_path

        return agent_args

    @staticmethod
    def __save_path_correction(agent_args, model_name):
        if not os.path.isdir(agent_args["save_path"]):

            model_save_path_split = os.path.split(agent_args["save_path"])
            model_file_name = model_save_path_split[-1]
            parent_dir = "".join(model_save_path_split[:-1])
            if not os.path.isdir(parent_dir):
                os.makedirs(parent_dir)

            if ".pth" or ".pt" in agent_args["save_path"]:
                model_file_name_reg_search = re.search(r"\.pt.*", model_file_name)

                if model_file_name_reg_search is not None:
                    parent_dir = "".join(model_save_path_split[:-1])
                    model_file_name = model_file_name.replace(
                        model_file_name_reg_search.group(), ""
                    )
                    agent_args["save_path"] = os.path.join(parent_dir, model_file_name)
                else:
                    os.mkdir(os.path.join(parent_dir, model_file_name))
            else:
                agent_args["save_path"] = os.path.join(parent_dir, model_name)

        else:
            agent_args["save_path"] = os.path.join(agent_args["save_path"], model_name)

        return agent_args


from .lib import RLPack

import os.path
import numpy as np
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
        assert (
            model_name is not None
            and model_args is not None
            and agent_args is not None
            and optimizer_args is not None
        ), "One or more of the mandatory arguments were passed as None"

        if activation_args is None:
            activation_args = dict()

        if "save_path" not in agent_args.keys():
            save_path = os.environ.get("RLPACK_SAVE_PATH")
            if save_path is None:
                raise ValueError(
                    "Expected save_path key to be in input argument dict 'agent_args' "
                    "or set as environment variable RLPACK_SAVE_PATH")
            agent_args["save_path"] = save_path

        model_filename = os.path.split(agent_args["save_path"])[-1]
        if ".pt" not in model_filename:
            if not os.path.isdir(agent_args["save_path"]):
                os.makedirs(agent_args["save_path"])
            agent_args["save_path"] += f'{model_name}.pt'

        self.get_dqn_agent = RLPack.GetDqnAgent(
            model_name,
            model_args,
            activation_args,
            agent_args,
            optimizer_args,
            device
        )

    def train(
        self,
        state_current: np.ndarray,
        state_next: np.ndarray,
        reward: float,
        action: int,
        done: bool,
    ) -> int:
        return self.get_dqn_agent.train(
            state_current,
            state_next,
            reward,
            action,
            done,
            state_current.shape,
            state_next.shape,
        )

    def policy(self, state_current: np.ndarray) -> int:
        return self.get_dqn_agent.policy(state_current, state_current.shape)

    def save(self):
        self.get_dqn_agent.policy.save()

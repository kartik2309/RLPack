from .lib import RLPack
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
    ):
        assert (model_name is not None and
                model_args is not None and
                agent_args is not None and
                optimizer_args is not None), (
            "One of the mandatory arguments were passed as None"
        )

        if activation_args is None:
            activation_args = dict()
        self.get_dqn_agent = RLPack.GetDqnAgent(model_name, model_args, activation_args, agent_args, optimizer_args)

    def train(
            self,
            state_current: np.ndarray,
            state_next: np.ndarray,
            reward: float,
            action: int,
            done: bool
    ) -> int:
        return self.get_dqn_agent.train(
            state_current.astype(dtype=np.double),
            state_next.astype(dtype=np.double),
            reward,
            action,
            done,
            state_current.shape,
            state_next.shape
        )

    def policy(self, state_current: np.ndarray) -> int:
        return self.get_dqn_agent.policy(state_current, state_current.shape)

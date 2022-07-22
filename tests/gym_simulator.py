import numpy as np
from rlpack.dqn import DQN
from environments.lunar_lander_simulator import LunarLander


def reshape_func(x: np.ndarray) -> np.ndarray:
    x = x.reshape((1, 8))

    return x


def simulate_lunar_lander():
    lunar_lander = LunarLander(
        agent=DQN,
        config_path="configs/dlqn1d.yaml",
        reshape_func=reshape_func
    )

    lunar_lander.train_agent(load=False, plot=True)
    # lunar_lander.evaluate_agent()


if __name__ == "__main__":
    simulate_lunar_lander()

from rlpack.dqn import DQN
from environments.lunar_lander_simulator import LunarLander


def simulate_lunar_lander():
    lunar_lander = LunarLander(
        agent=DQN,
        config_path="configs/dlqn1d.yaml"
    )

    lunar_lander.train_agent()


if __name__ == "__main__":
    simulate_lunar_lander()

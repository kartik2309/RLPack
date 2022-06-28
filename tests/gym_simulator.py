import gym
import numpy as np
import yaml
from rlpack.dqn import DQN


def lunar_lander_shape_correction(array: np.ndarray, mode='train') -> np.ndarray:
    array = array.reshape((2, 4))
    array = array.T
    if mode == "train":
        return array
    array = np.expand_dims(array, axis=0)
    return array


def simulate_lunar_lander() -> None:

    agent = DQN(
        model_name=yaml_dict["model_name"],
        model_args=yaml_dict["model_args"],
        agent_args=yaml_dict["agent_args"],
        optimizer_args=yaml_dict["optimizer_args"],
        activation_args=None,
    )
    rewards = list()

    env = gym.make("LunarLander-v2")

    if yaml_dict["mode"] == "train":
        for _ in range(yaml_dict["num_episodes"]):
            observation_current = env.reset()
            action = env.action_space.sample()

            for _ in range(yaml_dict["max_timesteps"]):
                # env.render()

                observation_next, reward, done, info = env.step(action=action)
                if not done:

                    rewards.append(reward)
                    observation_current = lunar_lander_shape_correction(observation_current)
                    observation_next = lunar_lander_shape_correction(observation_next)
                    action = agent.train(
                        state_current=observation_current,
                        state_next=observation_next,
                        action=action,
                        reward=reward,
                        done=done,
                    )

                    observation_current = observation_next
                else:
                    print("Reward:", sum(rewards) / len(rewards))
                    rewards.clear()
                    break
        env.close()

    else:
        _ = env.reset()
        action = env.action_space.sample()
        for _ in range(yaml_dict["max_timesteps"]):
            env.render()
            observation, reward, done, info = env.step(action=action)
            if done:
                break
            action = agent.policy(observation)
        env.close()


if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        yaml_dict = yaml.load(f, Loader=yaml.Loader)

    simulate_lunar_lander()

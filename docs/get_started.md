# Get started {#get_started}

Starting out with RLPack is very straight forward! In this tutorial we will see a basic example of
[DQN](@ref agents/dqn.md) algorithm applied to
[LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment provided by gym.

### Setup the config file

It is not mandatory, however we recommend using a config file (using yaml) to keep track of arguments. It
also makes it easy to run experiments. For our experiment setup, we can use the following config:

```python
{
    "mode": 'train'
    "env_name": 'LunarLander-v2'
    "model_name": 'mlp'
    "activation_name": 'relu'
    "agent_name": 'dqn'
    "optimizer_name": 'adam'
    "lr_scheduler_name": 'step_lr'
    "loss_function_name": 'huber_loss'
    "num_episodes": 5000
    "max_timesteps": 1000
    "reward_logging_frequency": 100
    "new_shape": [1, 8]
    "custom_suffix": "_best"
    "render": false

    "model_args": {
        "num_actions": 4,
        "sequence_length": 1,
        "hidden_sizes": [8, 64, 128, 256, 512],
        "dropout": 0.1
    }

    "activation_args": {

    }

    "agent_args": {
        "gamma": 0.99,
        "epsilon": 1,
        "min_epsilon": 0.01,
        "num_actions": 4,
        "memory_buffer_size": 16384,
        "target_model_update_rate": 64,
        "policy_model_update_rate": 4,
        "lr_threshold": 1e-5,
        "backup_frequency": 10000,
        "batch_size": 64,
        "epsilon_decay_rate": 0.995,
        "epsilon_decay_frequency": 1024,
        "prioritization_params": null,
        "apply_norm": "none",
        "apply_norm_to": ["none"],
        "tau": 0.83,
        "force_terminal_state_selection_prob": 0.7,
        "save_path": "/path/to/directory/to/save",
        "device": 'cuda'
    }

    "optimizer_args": {
        "lr": 0.001,
        "weight_decay": 0.01,
    }

    "lr_scheduler_args": {
        "step_size": 64,
        "gamma": 0.9999,
    }

    "loss_function_args": {
    }
}
```

You can also set the environment variable `SAVE_PATH` and not explicitly set the path in yaml file inside
`agent_args`. You can find a detailed explanation of the keys in yaml presented above
in the [overview](@ref overview.md). You can find detailed explanation for `agent_args` [here](@ref agents/dqn.md) and
for `model_args` [here](@ref models/in_built/mlp.md).

## Setup RLPack Simulator.

```python
import os
import yaml
from rlpack.simulator import Simulator

# Optional, to set SAVE_PATH environment variable. 
os.environ["SAVE_PATH"] = 'models/'
os.makedirs(f"{os.getenv('SAVE_PATH')}, exist_ok=True")

# Read from a yaml file with all the arguments for easy use.
with open("config.yaml", "rb") as f:
    config = yaml.load(f, yaml.Loader)

# Pass the config dictionary to Simulator.
simulator = Simulator(config=config)
simulator.run()
```

That's it! You will have the experiment up and running for the specified environment.

### Using custom models.

With RLPack, it is easy for you to pass your custom model. You just have to disable model related arguments in
config file. An updated config file would look like (shown as an update from previously shown config file):

```python
{
  "mode": 'train'
    "env_name": 'LunarLander-v2'
    # "model_name": 'mlp'
    # "activation_name": 'relu'
    "agent_name": 'dqn'
    "optimizer_name": 'adam'
    "lr_scheduler_name": 'step_lr'
    "loss_function_name": 'huber_loss'
    "num_episodes": 5000
    "max_timesteps": 1000
    "reward_logging_frequency": 100
    "new_shape": [1, 8]
    "custom_suffix": "_best"
    "render": false

    # "model_args": {
    #   "num_actions": 4,
    #   "sequence_length": 1,
    #   "hidden_sizes": [8, 64, 128, 256, 512],
    #   "dropout": 0.1
    # }

    # "activation_args": {
    # 
    # }

    "agent_args": {
      "gamma": 0.99,
      "epsilon": 1,
      "min_epsilon": 0.01,
      "num_actions": 4,
      "memory_buffer_size": 16384,
      "target_model_update_rate": 64,
      "policy_model_update_rate": 4,
      "lr_threshold": 1e-5,
      "backup_frequency": 10000,
      "batch_size": 64,
      "epsilon_decay_rate": 0.995,
      "epsilon_decay_frequency": 1024,
      "prioritization_params": null,
      "apply_norm": "none",
      "apply_norm_to": ["none"],
      "tau": 0.83,
      "force_terminal_state_selection_prob": 0.7,
      "save_path": "/path/to/directory/to/save",
      "device": 'cuda'
    }

    "optimizer_args": {
      "lr": 0.001,
      "weight_decay": 0.01,
    }

    "lr_scheduler_args": {
      "step_size": 64,
      "gamma": 0.9999,
    }

    "loss_function_args": {
    }
}
```

Now you can define your own custom PyTorch model. As an example:

```python
from rlpack import pytorch


class Base(pytorch.nn.Module):

    def __init__(self, dropout: float = 0.5):
        super(Base, self).__init__()
        self.linear1 = pytorch.Linear(in_features=8, out_features=32)
        self.linear2 = pytorch.Linear(in_features=32, out_features=64)
        self.linear_head = pytorch.Linear(in_features=64, out_features=4)
        self.activation = pytorch.nn.ReLU()
        self.dropout = pytorch.nn.Dropout(dropout)
        self.flatten = pytorch.nn.Flatten()

    def forward(self, x: pytorch.Tensor) -> pytorch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear_head(x)
        x = self.dropout(x)
        x = self.flatten(x)

        return x
```

Make sure that the output of the model corresponds to the number of actions on for your environment. In our example,
we have four actions.

Now we run the simulator as follows:

```python
import os
import yaml
from rlpack.simulator import Simulator

# Optional, to set SAVE_PATH environment variable. 
os.environ["SAVE_PATH"] = 'models/'
os.makedirs(f"{os.getenv('SAVE_PATH')}, exist_ok=True")

# Read from a yaml file with all the arguments for easy use.
with open("config.yaml", "rb") as f:
    config = yaml.load(f, yaml.Loader)

# Initialize your custom models and pass them via config. 
target_model = Base()
policy_model = Base()
# Refer to the model arguments for the agent being run. For DQN, we have 
# `target_model` and `policy_model`.
config["agent_args"]["target_model"] = target_model
config["agent_args"]["policy_model"] = policy_model
# Pass the config dictionary to Simulator.
simulator = Simulator(config=config)
simulator.run()
```

For model arguments of each agent, you can refer to [agents](@ref agents/index.md). Since [DQN](@ref agents/dqn.md)
expects `target_model` and `policy_model` as an argument, we modify the config accordingly. Once done, the
experiment will run as usual!

### Using custom gym environments

Using custom gym is again as easy as using your custom model. All you have to do is to pass your initialized
environment with the key `env` in the config.

```python
# Optional, to set SAVE_PATH environment variable. 
os.environ["SAVE_PATH"] = 'models/'
os.makedirs(f"{os.getenv('SAVE_PATH')}, exist_ok=True")

# Read from a yaml file with all the arguments for easy use.
with open("config.yaml", "rb") as f:
    config = yaml.load(f, yaml.Loader)
# Pass your environment 
config["env"] = CustomEnv()
# Pass the config dictionary to Simulator.
simulator = Simulator(config=config)
simulator.run()
```
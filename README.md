# RLPack

Implementation of RL Algorithms with PyTorch. Heavy workloads have
been optimized with C++ backend.

# Installation

To install RLPack, simply follow the following steps: <br>

* Clone and install
    ```zsh
    git clone https://github.com/kartik2309/RLPack.git
    cd RLPack 
    pip install .
    ```
  This will install the package in your python environment.

# Getting started 

* To get started, you can run the following snippet:
  ```python 
  import os
  import yaml
  from rlpack.simulator import Simulator

  # Optional, to set SAVE_PATH environment variable. 
  os.environ["SAVE_PATH"] = 'models/'
  os.makedirs(f"{os.getenv('SAVE_PATH')}, exist_ok=True")
  
  # Read from a yaml file with all the arguments for easy use.
  with open("config.yaml", "rb) as f:
    yaml.load(f, yaml.Loader)
  
  # Pass the config dictionary to Simulator.
  simulator = Simulator(config=config)
  simulator.run()
  ```
  
  `Simulator` accepts a config dictionary which must contain all mandatory arguments.

* An example of basic config.yaml can be as follows: 
  ```yaml
  mode: 'train'
  env_name: 'LunarLander-v2'
  model_name: 'mlp'
  agent_name: 'dqn'
  num_episodes: 5000
  max_timesteps: 1000
  reward_logging_frequency: 100
  new_shape: [ 1, 8 ]
  custom_suffix: "_best"
  render: false
  
  # Model arguments for mlp model
  model_args: {
  num_actions: 4,
  sequence_length: 1,
  hidden_sizes: [ 8, 64, 128, 256, 512 ],
  dropout: 0.1
  }
  
  # Activation args. Relevant arguments can be passed for a particular activation
  activation_args: {
  activation: "relu"
  }
  
  # Agent args for dqn agent.
  agent_args: {
  gamma: 0.99,
  epsilon: 1,
  min_epsilon: 0.01,
  num_actions: 4,
  memory_buffer_size: 1048576,
  target_model_update_rate: 64,
  policy_model_update_rate: 4,
  lr_threshold: 1e-5,
  model_backup_frequency: 10000,
  batch_size: 64,
  epsilon_decay_rate: 0.995,
  epsilon_decay_frequency: 1024,
  prioritization_params: {
  beta: 0.4,
  max_beta: 1.0,
  beta_annealing_frequency: 2048,
  beta_annealing_factor: 1.001,
  alpha: 0.6,
  min_alpha: 0.01,
  alpha_annealing_frequency: 2048,
  alpha_annealing_factor: 0.999,
  prioritization_strategy: "rank-based",
  },
  apply_norm: "none",
  apply_norm_to: [ "none" ],
  tau: 0.83,
  force_terminal_state_selection_prob: 0.7,
  
  optimizer_args: {
  optimizer: "adam",
  lr: 0.001,
  weight_decay: 0.01,
  }
  
  lr_scheduler_args: {
  "scheduler": "step_lr",
  "step_size": 64,
  "gamma": 0.9999,
  }
  
  loss_function_args: {
  "loss_function": "huber_loss"
  }
  
  device: 'cuda'
  ```
  This config will train a dqn model in CUDA with rank-based prioritization.

# License
RLPack is released under [MIT LICENSE](LICENSE.md).  

Please raise an issue on GitHub to report a bug. Pull Requests can be raised after discussion on the raise issue.


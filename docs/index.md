# RLPack Documentation

RLPack provides an easy-to-use interface for Deep RL Algorithms. 
RLPack is built on top `PyTorch` and leverages accelerators available on your machine to train 
the agents' models. Heavier computational requirements outside the scope of PyTorch have been
implemented and optimized with C++ backend. OpenAI's `gym` is used to simulate the environment 
for training.

## General structure

RLPack runs `Simulator` defined in `rlpack.simulator`. This class provides an easy-to-use 
interface to train agents. `Simulator` accepts a dictionary (keyword argument `config`) which 
must contain parameters specific to the agent you are about to train or evaluate. In general, the
config dictionary must contain some mandatory keys as discussed below: 
- `mode`: Represents the mode in which the agent is to be trained. This must either be 'train' or 
'eval'.
- `env_name`: Environment name on which the agent is to be trained. It must be the string name of the 
environment as per OpenAI's `gym`. It launches the gym environment to be trained on
- `model_name`: The model intended to be used in the agent. This must be a string. Currently implemented 
models and further details on them is provided [here](models/index.md). Note that custom models can also 
be implemented and used and hence it's not a mandatory argument.
- `activation_name`: The name of activation to be used. This must be a string. This is only used if an in-built
model is used (selecting with `model_name`). Keywords for `activation_name` and further details on them can be found 
[here](models/activations.md).
- `optimizer_name`: This is the name of the optimizer to be used. This must be a string. This is a mandatory key to be 
passed. Keywords for `optimizer_name` and further details on them can be found [here](models/optimizers.md).
- `loss_function_name`: The name of loss function to be used. This must a string. This is a mandatory key to be 
passed. Keywords for `loss_function_name` and further details on them can be found [here](models/loss_functions.md).
- `lr_scheduler_name`: The name of Learning-Rate Scheduler to be used. This is an optional argument. Keywords for 
`lr_scheduler_name` and further details on them can be found [here](models/lr_schedulers.md).
- `agent_name`: The agent intended to be trained. This must be a string. Currently implemented
  models and further details on them is provided [here](agents/index.md).
- `num_episodes`: The number of episodes up to which the agent is to be trained. This must be an integer 
value.
- `max_timesteps`: The maximum timesteps to be allowed per episode. This must be an integer.
- `reward_logging_frequency`: The frequency of logging rewards on the screen in terms of number of episodes. 
After specified number of episodes are passed, the mean is calculated and logged. This must be an integer.
- `model_args`: If using an in-built model (selected with `model_name`), the corresponding arguments for them
must be provided. This must be a dictionary with each keyword argument and their corresponding values. 
- `activation_args`: If using an in-built model (selected with `model_name` and `activation_name`), the activation 
arguments to be used if required. Necessary arguments for the provided activation with `activation_name` must be 
passed in this dictionary as keyword arguments. More details can be found [here](models/activations.md)
- `agent_args`: The arguments for the agent that has been selected via `agent_name` arguments. It must a dictionary 
with all the necessary that corresponds to the mandatory arguments that needs to be passed for the selected agent. 
More details on each implemented agent and their corresponding arguments can be found [here](agents/index.md).
- `optimizer_args`: The optimizer to be used to train the model in the agent. Necessary arguments for the provided 
activation with `optimizer_name` must be passed in this dictionary as keyword arguments. Further details on how they 
are used of in-built models and values supported is provided [here](models/optimizers.md).
- `loss_function_args`: The loss function keyword arguments to be used depending on the loss function selected by 
`loss_function_name`. Further details on how they are used of in-built models and values supported is 
provided [here](models/loss_functions.md).
- `lr_scheduler_args`: If `lr_scheduler_name` is passed in config dictionary, this argument must be passed. Necessary
arguments for the provided activation with `lr_scheduler_name` must be passed in this dictionary as keyword arguments.
Further details on how they are used of in-built models and values supported is provided [here](models/lr_schedulers.md).
- `device`: The device on which we want the model within the agent to be trained. This relies on PyTorch's support for
device argument. Recommended devices are `cpu`, `cuda` and `mps`. 

The config dictionary to be passed into `Simulator` must contain all the necessary arguments as mentioned. Once prepared 
a dictionary named `config` we can write the following: 
```python
config = {
  # Mandatory arguments to be passed.
}
from rlpack.simulator import Simulator
simulator = Simulator(config=config)
```

Once initialized, `Simulator` prepares the agents, models, optimizers, lr_schedulers and initializes the 
environment. Once done, we can call `run` method, which can run the agent as we desired with config.
```python
simulator.run()
```
`run` method can provide callbacks which can enable us to pass custom models to be used inside the agent. 

## Agents

RLPack implements variety of agents to be easily used via config dict for training and evaluating. Learn more about 
them [here](agents/index.md) 


## Supported Models and Tools

RLPack implements variety of models and corresponding tools to be used for training and evaluating which can be easily 
used via config dictionary. Learn more about them [here](agents/index.md) 

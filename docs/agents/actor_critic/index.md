# Actor Critic Methods {#actor_critic}

Actor-Critic methods are provided in RLPack via rlpack.actor_critic package. [In-Built](@ref models/in_built/index.md)
can be used with Actor Critic agents to train an agent on the fly. Currently following variants have been implemented: 

| Method                                 | Description                      | Keyword |
|----------------------------------------|----------------------------------|---------|
| [A2C](@ref agents/actor_critic/a2c.md) | Synchronous Actor Critic Method  | `"a2c"` |
| [A3C](@ref agents/actor_critic/a3c.md) | Asynchronous Actor Critic Method | "`a3c`" |

Actor-Critic methods implemented in RLPack support both continuous and discrete action spaces. To support both types of 
action spaces simultaneously, RLPack provides an argument `distribution` for actor critic methods to sample actions
from. Currently, following distributions are available and accessible by keyword when using simulators.

| Distribution          | Description                                                                                                                                                                | Keyword                 |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| `Normal`              | The Normal distribution (for continuous action spaces). More info can be found [here](https://pytorch.org/docs/stable/distributions.html#normal).                          | `"normal"`              |
| `LogNormal`           | The LogNormal distribution (for continuous action spaces). More info can be found [here](https://pytorch.org/docs/stable/distributions.html#lognormal).                    | `"log_normal"`          |
| `multivariate_normal` | The Multivariate Normal distribution (for continuous action spaces). More info can be found [here] (https://pytorch.org/docs/stable/distributions.html#multivariatenormal) | `"multivariate_normal"` |
| `Categorical`         | The Categorical distribution (for discrete action spaces). More info can be found [here](https://pytorch.org/docs/stable/distributions.html#categorical).                  | `"categorical"`         |
| `Binomial`            | The Binomial distribution (for discrete action spaces). More info can be found [here](https://pytorch.org/docs/stable/distributions.html#binomial).                        | `"binomial"`            |
| `Bernoulli`           | The Bernoulli distribution (for discrete action spaces). More info can be found [here](https://pytorch.org/docs/stable/distributions.html#bernoulli)                       | `"bernoulli"`           |

Since Actor Critic implementations in RLPack support both continuous and discrete action spaces, all methods have an 
argument `action_space`. You must pass this argument in the following way:
- **Continuous case**: In this case, you must the argument as a list of two elements. 
  - First element of the list will have output features desired from policy model. This is directly unpacked into 
  the distribution object selected with `distribution`, hence make sure to check the arguments for the distribution 
  you are passing. Generally the policy model in continuous case must output the statistics for probability 
  distribution selected.
  - Second element of the list will have the output shape of the action as list. This shape is used to sample
  from the given distribution. 
  - An example: If we have passed normal distribution and action is expected to be of shape `(1,)`, then the 
  `action_space` would be `[2, [1]]`, where 2 represents the no. of features in output of the policy model (for loc 
  (mean) and scale (standard deviation), and `[1]` is the desired action shape.
- **Discrete case**: In this case, you can simply pass an integer representing the number of actions for the environment.
For example if agent can take four actions in the environment, we simply pass `4`.
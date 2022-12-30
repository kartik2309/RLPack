# Actor Critic Methods {#actor_critic}

Actor-Critic methods are provided in RLPack via rlpack.actor_critic package. [In-Built](@ref models/in_built/index.md)
can be used with Actor Critic agents to train an agent on the fly. Currently following variants have been implemented: 

| Method                                 | Description                      | Keyword |
|----------------------------------------|----------------------------------|--------|
| [AC](@ref agents/actor_critic/ac.md)   | Actor Critic Uni-Agent method    | `"ac"` |
| [A2C](@ref agents/actor_critic/a2c.md) | Synchronous Actor Critic Method  | `"a2c"` |
| [A3C](@ref agents/actor_critic/a3c.md) | Asynchronous Actor Critic Method | `"a3c"` |


Since Actor Critic implementations in RLPack support both continuous and discrete action spaces, all methods have an 
argument `action_space`. You must pass this argument in the following way:
- **Continuous case**: In this case, you must the argument as a tuple of two elements. 
  - First element of the tuple will have output features desired from policy model. This is directly unpacked into 
  the distribution object selected with `distribution`, hence make sure to check the arguments for the distribution 
  you are passing. Generally the policy model in continuous case must output the statistics for probability 
  distribution selected.
  - Second element of the tuple will have the output shape of the action as a list. This shape is used to sample
  from the given distribution. You must know the default shape of sample from the given distribution. If you wish
  to sample with default shape from the distribution, you can pass an empty list or `None`. An example of default
  shape is [Normal distribution](https://pytorch.org/docs/stable/distributions.html#normal) where by-default a scalar
  value is sampled, hence a tensor of shape `(1,)` is sampled.
  - An example: If we have passed normal distribution and action is expected to be of shape `(1,)`, then the 
  `action_space` would be `[2, []]`, where 2 represents the no. of features in output of the policy model (for loc 
  (mean) and scale (standard deviation), and `[]` is the desired shape of sample. This will sample a value of shape
  `(1,)` from normal distribution. This will also be equivalent to `[2, None]`.
- **Discrete case**: In this case, you can simply pass an integer representing the number of actions for the environment.
For example if agent can take four actions in the environment, we simply pass `4`.
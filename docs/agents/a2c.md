@page a2c
# Synchronized Actor-Critic (A2C)

A2C is implemented as `rlpack.actor_critic.a2c.A2C`. It inherits from
`Agent` class defined in `rlpack.utils.base.agent.py`.

A2C implements the synchronous Actor-Critic agent which supports gradient accumulation and mean reduction 
of gradients for training the model. Unlike A3C, only a single agent interacts with environment and mean reduced 
gradients are synchronously used to train the policy model.

#### agent_name: "a2c".

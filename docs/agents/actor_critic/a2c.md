# Synchronized Actor-Critic (A2C) {#a2c}

`A2C` is implemented as `rlpack.actor_critic.a2c.A2C`. It inherits from
`ActorCriticAgent` class defined as `rlpack.actor_critic.utils.actor_critic_agent.ActorCriticAgent`.

A2C implements the synchronous Actor-Critic agent which supports gradient accumulation and mean reduction 
of gradients for training the model. Unlike A3C, only a single agent interacts with environment and mean reduced 
gradients are synchronously used to train the policy model. For more
information refer [here](https://arxiv.org/abs/1602.01783)

<h4> Keyword: <kbd> Keyword: `agent_name: "a2c"` </kbd> </h4>

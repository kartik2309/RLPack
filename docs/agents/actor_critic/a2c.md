# Advantage Actor-Critic (A2C) {#a2c}

`A2C` is implemented as `rlpack.actor_critic.a2c.A2C`. It inherits from
`ActorCriticAgent` class defined as `rlpack.actor_critic.utils.actor_critic_agent.ActorCriticAgent`.

A2C implements the synchronous Actor-Critic agent which supports gradient accumulation and mean reduction 
of gradients for training the model. Unlike A3C, each actor waits for completion of interaction with the environment 
after which gradient mean reduction takes place synchronously. For more information refer 
[here](https://arxiv.org/abs/1602.01783)

Note that to run simulation with A2C, make sure to use rlpack.simulator_distributed.SimulatorDistributed, as this will
run the simulation in the given environment in distributed setting.

<h4> Keyword: <kbd> Keyword: `agent_name: "a2c"` </kbd> </h4>

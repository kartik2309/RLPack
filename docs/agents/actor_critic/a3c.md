# Asynchronized Actor-Critic (A3C) {#a3c}

`A3C` is implemented as `rlpack.actor_critic.a3c.A3C`. It inherits from
`A2C` class defined as `rlpack.actor_critic.a2c.A2C`.

A3C implements the asynchronous Actor-Critic agent which supports gradient accumulation and mean reduction
of gradients for training the model similar to [A2C](@ref agents/actor_critic/a2c.md). Along with this A3C supports
multiple actors which interact with the environment at the same time. Gradients of all actors' policy model are 
asynchronously reduced and mean gradient is populated for each actor's policy model before optimizer step.

<h4> Keyword: <kbd> Keyword: `agent_name: "a3c"` </kbd> </h4>

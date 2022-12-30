# Actor-Critic (AC) {#ac}

`Actor-Critic (AC)` is implemented as `rlpack.actor_critic.ac.AC`. It inherits from
`ActorCriticAgent` class defined as `rlpack.actor_critic.utils.actor_critic_agent.ActorCriticAgent`.

This version of Actor-Critic methods is the basic version of Actor Critic with uni-agent setting, so there is only 
actor interacting with the agent. This is primarily useful to run a trained agent from A2C or A3C for evaluation in 
a single process but can also be used to train agents where computational resources are limited.


<h4> Keyword: <kbd> Keyword: `agent_name: "ac"` </kbd> </h4>

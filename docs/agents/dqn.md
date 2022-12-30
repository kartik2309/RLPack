# Deep Q-Networks {#dqn}
`DQN` is interfaced via `rlpack.dqn.dqn.Dqn`. This class provides access to required DQN variant based on 
input configuration. DQN implementation in RLPack is based on Dual Q-Networks with a trainable policy network and a 
frozen target network.  

<h4> Keyword: <kbd> agent_name: "dqn" </kbd> </h4>


## Implementations
There are currently three supported implementations for DQN: 

-  **Uniform-Sampling based DQN**

    This variant is implemented as rlpack.dqn.dqn_agent.DqnAgent. In this DQN variant, we populate the memory
(also called transition buffer) with transitions and sample them as per uniform distribution. This 
essentially means that each transition we store in the buffer has equal probability of being sampled for next 
batch. To use this variant, set `prioritization_params=None` in `agent_args`. 

- **Proportional based DQN**

  This variant is implemented as rlpack.dqn.dqn_proportional_prioritization_agent.DqnProportionalPrioritizationAgent. 
In this DQN variant, we populate the memory (also called transition buffer) with transitions with each sample having a 
fixed priority (generally same of all samples initially) and sample them as per this priority. Probabilities and weights 
are computed from directly using TD Errors and updates are made accordingly. This was proposed 
by [Tom Schaul et al. (ICLR 2016)](https://arxiv.org/pdf/1511.05952.pdf). To use this variant, 
set `prioritization_params={..., "prioritization_strategy": "proportional"}` 

- **Rank-based DQN**

  This variant is implemented as rlpack.dqn.dqn_rank_based_prioritization_agent.DqnRankBasedPrioritizationAgent. 
In this DQN variant, we populate the memory (also called transition buffer) with transitions with each sample having 
a fixed priority (generally same of all samples initially) and sample them as per this priority. Probabilities 
and weights are computed by sorting TD Errors to find each transition's ranks; updates are then made 
accordingly. This was proposed by [Tom Schaul et al. (ICLR 2016)](https://arxiv.org/pdf/1511.05952.pdf). To use this 
set `prioritization_params={..., "prioritization_strategy": "rank-based"}`.

## Prioritization Parameters

For Proportional based DQN and  Rank-based DQN, it is mandatory that we set `prioritization_params` in `agent_args`.
`prioritization_params` is a dictionary that must define some prioritization parameters we intend to use for training. 
Mandatory arguments for `prioritization_params` are: 
    
- `alpha`: The alpha value for prioritization. Alpha indicates the level of prioritization for transitions from memory, 
higher value implying more aggressive prioritization.
- `beta`: The beta value for bias correction. Beta indicates the level of bias correction by adjusting Important
Sampling weights. The higher values of beta indicate the greater bias correction.
- `min_alpha`: The lower limit of `alpha`. This is the minimum value beta can reach during training. Default is set to 
0.
- `max_beta`: The upper limit of `beta`. This is the maximum value beta can reach during training. Default is set to 1.
- `alpha_annealing_frequency`: This parameter indicates the number of timesteps after which the alpha is annealed. 
  - If `alpha_annealing_fn` is passed, this function is called every `alpha_annealing_frequency` timesteps. 
  - If `None` is passed, `alpha` is not annealed.
- `beta_annealing_frequency`: This parameter indicates the number of timesteps after which the beta is annealed.
    - If `beta_annealing_fn` is passed, this function is called every `beta_annealing_frequency` timesteps.
    - If `None` is passed, `beta` is not annealed.
- `alpha_annealing_fn`: The annealing function (or scheduler) to be used to anneal alpha. If this function requires extra
arguments, they must be passed with another key `alpha_annealing_fn_args` as tuple or `alpha_annealing_fn_kwargs` as a 
dictionary of keyword arguments. The function definition's first argument must be alpha. 
- `beta_annealing_fn`: The annealing function (or scheduler) to be used to anneal beta. If this function requires extra
  arguments, they must be passed with another key `beta_annealing_fn_args` as tuple or `beta_annealing_fn_kwargs` as a
  dictionary of keyword arguments. The function definition's first argument must be beta.
- `alpha_annealing_factor`: This is an optional argument that can be used, if not external `alpha_annealing_fn` 
is not supplied. This loads the default annealing function which decays alpha by the supplied factor.
- `beta_annealing_factor`: This is an optional argument that can be used, if not external `beta_annealing_fn`
is not supplied. This loads the default annealing function which decays beta by the supplied factor. 
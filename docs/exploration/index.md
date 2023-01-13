# Exploration {#explorations}

RLPack provides in-built tools to perform exploration in the environment. These tools are accessible by keywords
when using RLPack simulators (`rlpack.simualator.Simulator` or `rlpack.simulator_distributed.SimulatorDistributed`). 
They are implemented in `rlpack.exploration`. All classes inherit from the base class 
`rlpack.exploration.utils.exploration.Exploration`.

Currently following exploration tools have been implemented: 

| Exploration Tool            | Description                                                                                                                                                                   | Keyword             |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `GaussianNoise`             | The tool to add gaussian noise to the desired output. Implemented as rlpack.exploration.gaussian_noise.GaussianNoise.                                                         | `"gaussian_noise"`  |
| `StateDependentExploration` | The tool to add SDE noise to actions. Note that this is a learnable exploration tool. Implemented as rlpack.exploration.state_dependent_exploration.StateDependentExploration | `"state_dependent"` |

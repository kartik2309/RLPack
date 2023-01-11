"""!
@package rlpack.exploration
@brief This package implements the exploration tools for RLPack to explore the environment


Currently following classes have been implemented:
    - `GaussianNoise`: This class implements the gaussian noise exploration tool with optional weights for
        samples and annealing of distribution parameters. Implemented as
        rlpack.exploration.gaussian_noise.GaussianNoise


Following packages are part of exploration:
    - `utils`: A package utilities for exploration package.
"""

from rlpack.exploration.gaussian_noise_exploration import GaussianNoiseExploration
from rlpack.exploration.state_dependent_exploration import StateDependentExploration

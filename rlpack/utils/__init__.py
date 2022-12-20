"""!
@package rlpack.utils
@brief This package implements the basic utilities to be used across rlpack.


Currently following classes have been implemented:
    - `Normalization`: Normalization tool implemented as rlpack.utils.normalization.Normalization with
        support for regular normalization methods.
    - `SanityCheck`: Sanity check for arguments when using Simulator from rlpack.simulator.Simulator. Class is
        implemented as rlpack.utils.sanity_check.SanityCheck.
    - `Setup`: Sets up the simulator to run the agent with environment. Implemented as rlpack.utils.setup.Setup.

Following packages are part of utils:
    - `base`: A package for base class, implemented as utils.base

Following TypeVars have been defined:
    - `LRScheduler`: The Typing variable for LR Schedulers.
    - `LossFunction`: The Typing variable for Loss Functions.
    - `Activation`: The Typing variable for Activations.
"""


from typing import TypeVar

## The Typing variable for LR Schedulers. @I{# noqa: E266}
LRScheduler = TypeVar("LRScheduler")
## The Typing variable for Loss Functions. @I{# noqa: E266}
LossFunction = TypeVar("LossFunction")
## The Typing variable for Activations. @I{# noqa: E266}
Activation = TypeVar("Activation")
## The Typing variable for Distribution. @I{# noqa: E266}
Distribution = TypeVar("Distribution")

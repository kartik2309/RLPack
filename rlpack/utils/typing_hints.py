"""!
@package rlpack.utils
@brief This package implements the basic utilities to be used across rlpack.


Currently following classes have been implemented:
    - `Normalization`: Normalization tool implemented as rlpack.utils.normalization.Normalization with
        support for regular normalization methods.
    - `SanityCheck`: Sanity check for arguments when using Simulator from rlpack.simulator.Simulator. Class is
        implemented as rlpack.utils.sanity_check.SanityCheck.
    - `Setup`: Sets up the simulator to run the agent with environment. Implemented as rlpack.utils.setup.Setup.
    - `InternalCodeSetup`: For internal use to check/validate arguments and to retrieve codes for internal use.
        Implemented as rlpack.utils.internal_code_setup.InternalCodeSetup.

Following packages are part of utils:
    - `base`: A package for base class, implemented as rlpack.utils.base

Following typing hints have been defined:
    - `LRScheduler`: The Typing variable for LR Schedulers.
    - `LossFunction`: Typing hint for loss functions for RLPack. Implemented as
        rlpack.utils.typing_hints.LossFunction.
    - `Activation`: Typing hint for activation functions for RLPack. Implemented as
        rlpack.utils.typing_hints.Activation.
    - `RunFuncSignature`: Typing hint for function signatures to be launched in
        rlpack.simulator_distributed.SimulatedDistributed in distributed mode. Implemented as
        rlpack.utils.typing_hints.RunFuncSignature.
    - `GenericFuncSignature`: Typing hint for generic void function signatures. Implemented as
        rlpack.utils.typing_hints.GenericFuncSignature.
"""


from typing import Any, Dict, Protocol

from rlpack import pytorch


class LRScheduler(pytorch.optim.lr_scheduler._LRScheduler):
    """
    The Typing variable for LR Schedulers.
    """

    def __init__(self, *args, **kwargs):
        """
        __init__ method to define the initialization parameters of Loss Function.
        @param *args: Positional arguments for the initialization.
        @param **kwargs: Keyword arguments for the initialization
        """
        super().__init__(*args, **kwargs)


class LossFunction(pytorch.nn.Module):
    """
    The Typing variable for Loss Functions.
    """

    def __init__(self, *args, **kwargs):
        """
        __init__ method to define the initialization parameters of Loss Function.
        @param *args: Positional arguments for the initialization.
        @param **kwargs: Keyword arguments for the initialization
        """
        super().__init__()

    def __call__(self, *args, **kwargs):
        """
        __call__ method to define the signature for the loss function.
        @param *args: Positional arguments for the callable.
        @param **kwargs: Keyword arguments for the callable
        """
        return


class Activation(pytorch.nn.Module):
    """
    The Typing variable for Activations.
    """

    def __init__(self, *args, **kwargs):
        """
        __init__ method to define the initialization parameters of Activation.
        @param *args: Positional arguments for the initialization.
        @param **kwargs: Keyword arguments for the initialization
        """
        super(Activation, self).__init__()

    def __call__(self, *args, **kwargs):
        """
        __call__ method to define the signature for the activation.
        @param *args: Positional arguments for the callable.
        @param **kwargs: Keyword arguments for the callable
        """
        return


class RunFuncSignature(Protocol):
    """
    Typing hint for function to be spawned in multiprocess.
    """

    def __call__(
        self, process_rank: int, world_size: int, config: Dict[str, Any], **kwargs
    ) -> None:
        """
        __call__ method to define the signature for the callable.
        @param process_rank: int: The process rank of the initialized process.
        @param world_size: int: Total number of processes launched or to be launched.
        @param config: Dict[str, Any]: The configuration to be used.
        @param kwargs: Other keyword arguments corresponding to
            rlpack.environments.environments.Environments.train method.
        """
        return


class GenericFuncSignature(Protocol):
    """
    Typing hint for a generic function.
    """

    def __call__(self, *args, **kwargs) -> None:
        """
        __call__ method to define the signature for the callable.
        @param *args: Positional arguments for the callable.
        @param **kwargs: Keyword arguments for the callable
        """
        return

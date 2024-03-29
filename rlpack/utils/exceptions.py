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

Following exceptions have been defined:
    - `AgentError`: For error happening in Agent's initialization. Implemented as rlpack.utils.exceptions.AgentError.

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


class AgentError(Exception):
    """
    Exception for when error occurs in initializing agents.
    """

    def __init__(self, message: str):
        """
        Initialization method for AgentError
        @param message: str: The error message
        """
        super(AgentError, self).__init__(message)
        self.message = message

    def __repr__(self) -> str:
        """
        repr method to return the error message
        @return: str: The error message in string.
        """
        return self.message

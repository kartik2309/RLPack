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
    - `base`: A package for base class, implemented as utils.base

Following TypeVars have been defined:
    - `LRScheduler`: The Typing variable for LR Schedulers.
    - `LossFunction`: The Typing variable for Loss Functions.
    - `Activation`: The Typing variable for Activations.
"""


from typing import List, Union

from rlpack import pytorch
from rlpack.utils.base.internal_code_register import InternalCodeRegister


class InternalCodeSetup(InternalCodeRegister):
    def __init__(self):
        super(InternalCodeSetup, self).__init__()

    def get_apply_norm_mode_code(self, apply_norm: str) -> int:
        """
        This method retrieves the apply_norm code from the given string. This code is to be supplied to agents.
        @param apply_norm: str: The apply_norm string, specifying the normalization techniques to be used.
            *See the notes below to see the accepted values.
        @return (int): The code corresponding to the supplied valid apply_norm.

        * NOTE
        The value accepted for `apply_norm` are:
            - "none": No normalization
            - "min_max": Min-Max normalization
            - "standardize": Standardization.
            - "p_norm": P-Normalization
        """
        if apply_norm not in self.norm_mode_codes.keys():
            raise ValueError("Invalid or unsupported value for `apply_norm` passed")
        return self.norm_mode_codes[apply_norm]

    def get_apply_norm_to_mode_code(self, apply_norm_to: List[str]) -> int:
        """
        This method retrieves the apply_norm code_to from the given string. This code is to be supplied to agents.
        @param apply_norm_to: List[str]: The apply_norm_to list, specifying the quantities on which we wish to
            apply normalization specified by `apply_norm`
            *See the notes below to see the accepted values.
        @return int: The code corresponding to the supplied valid apply_norm_to.

        *NOTE
        The value accepted for `apply_norm_to` are:
            - ["none"]: Don't apply normalization to any quantity.
            - ["states"]: Apply normalization to states.
            - ["rewards"]: Apply normalization to rewards.
            - ["td"]: Apply normalization for TD values.
            - ["states", "rewards"]: Apply normalization to states and rewards.
            - ["states", "td"]: Apply normalization to states and TD values.
        """
        apply_norm_to = tuple(sorted(apply_norm_to))
        if apply_norm_to not in self.norm_to_mode_codes.keys():
            raise ValueError("Invalid or unsupported value for `apply_norm_to` passed")
        return self.norm_to_mode_codes[apply_norm_to]

    def get_prioritization_code(self, prioritization_strategy: str) -> int:
        """
        This method retrieves the prioritization code for corresponding strategy passed as string
            in prioritized parameters.
        @param prioritization_strategy: str: A dictionary containing memory prioritization parameters for
            agents that may use it
            *See the notes below to see the accepted values.
        @return int: The prioritization code for corresponding string value.

        *NOTE:
        The accepted values for `prioritization_strategy` are as follows:
            - "uniform": No prioritization is done, i.e., uniform sampling takes place; Code: 0.
            - "proportional": Proportional prioritization takes place when sampling transition; Code: 1.
            - "rank-based": Rank based prioritization takes place when sampling transitions; Code: 2.
        """
        if prioritization_strategy not in self.prioritization_strategy_codes.keys():
            raise NotImplementedError(
                f"The provided prioritization strategy {prioritization_strategy} is not supported or is invalid!"
            )
        code = self.prioritization_strategy_codes[prioritization_strategy]
        return code

    def get_torch_dtype(self, dtype: str) -> pytorch.dtype:
        """
        Check if the input given strong for torch datatype is valid. If valid, return the datatype class from
        PyTorch.
        @param dtype: str: The datatype string.
        :return: pytorch.dtype: The datatype class for given string.
        """
        if dtype not in self.pytorch_dtype_map.keys():
            raise ValueError(
                "Invalid or unsupported datatype has been passed! Either pass 'float64' or 'float32'"
            )
        return self.pytorch_dtype_map[dtype]

    def check_validity_of_apply_norm_code(self, apply_norm: int) -> None:
        """
        Check if validity of the `apply_norm` code. Raises ValueError if code is invalid.
        @param apply_norm: int: `apply_norm` code to check
        """
        if apply_norm in list(self.norm_mode_codes.values()):
            return
        raise ValueError("Invalid value of `apply_norm` code was received!")

    def check_validity_of_apply_norm_to_code(self, apply_norm_to: int) -> None:
        """
        Check of validity of the `apply_norm_to` code. Raises ValueError if code is invalid.
        @param apply_norm_to: int: `apply_norm_to` code to check
        """
        if apply_norm_to in list(self.norm_to_mode_codes.values()):
            return
        raise ValueError("Invalid value of `apply_norm_to` code was received!")

    @staticmethod
    def check_validity_of_action_space(action_space: Union[int, List[int, Union[List[int], None]]]) -> None:
        """
        Checks the validity of action space for agents.
        @param action_space: Union[int, List[int, Union[List[int], None]]]: The action space to check
        """
        if isinstance(action_space, int):
            return
        if isinstance(action_space, list):
            if len(action_space) == 2:
                if isinstance(action_space[0], int) and isinstance(action_space[-1], (list, type(None))):
                    return
        raise ValueError(
            "`action_space` must be either an int for discrete actions or a list for continuous actions. "
            "Please refer to rlpack.actor_critic.a2c.A2c.__init__ for more details."
        )

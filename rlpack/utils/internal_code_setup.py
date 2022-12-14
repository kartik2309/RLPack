from typing import List

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
        apply_norm_to = tuple(apply_norm_to)
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

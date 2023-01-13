"""!
@package rlpack.utils.base.registers
@brief This package implements the base classes for registers to be used across rlpack


Currently following base classes have been implemented:
    - `Register`: Register of information of all in-built models and agents implemented as
        rlpack.utils.base.registers.register.Register.
    - `InternalCodeRegister`: Register for information on codes to be used internally in RLPack; implemented as
        rlpack.utils.base.registers.internal_code_register.InternalCodeRegister
"""


from rlpack import pytorch


class InternalCodeRegister:
    def __init__(self):
        ## The mapping between given keyword and normalisation method codes. @I{# noqa: E266}
        self.norm_mode_codes = {"none": -1, "min_max": 0, "standardize": 1, "p_norm": 2}
        ## The mapping between given keyword and normalisation quantity (`apply_norm_to`) codes. @I{# noqa: E266}
        self.norm_to_mode_codes = {
            "none": -1,
            "states": 0,
            "state_values": 1,
            "rewards": 2,
            "returns": 3,
            "td": 4,
            "advantages": 5,
            "action_log_probabilities": 6,
            "entropies": 7,
        }
        ## The mapping between prioritization strategy keywords and prioritization strategy codes. @I{# noqa: E266}
        self.prioritization_strategy_codes = {
            "uniform": 0,
            "proportional": 1,
            "rank-based": 2,
        }
        ## The mapping between strings for datatypes to pytorch datatypes. @I{# noqa: E266}
        self.pytorch_dtype_map = {
            "float32": pytorch.float32,
            "float64": pytorch.float64,
        }

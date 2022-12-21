"""!
@package rlpack.utils.base
@brief This package implements the base classes to be used across rlpack


Currently following base classes have been implemented:
    - `Agent`: Base class for all agents, implemented as rlpack.utils.base.agent.Agent.
    - `Register`: Register of information of all in-built models and agents implemented as
        rlpack.utils.base.register.Register.
    - `InternalCodeRegister`: Register for information on codes to be used internally in RLPack; implemented as
        rlpack.utils.base.internal_code_register.InternalCodeRegister
"""


from rlpack import pytorch


class InternalCodeRegister:
    def __init__(self):
        ## The mapping between given keyword and normalisation method codes. @I{# noqa: E266}
        self.norm_mode_codes = {"none": -1, "min_max": 0, "standardize": 1, "p_norm": 2}
        ## The mapping between given keyword and normalisation quantity (`apply_norm_to`) codes. @I{# noqa: E266}
        self.norm_to_mode_codes = {
            ("none",): -1,
            ("states",): 0,
            ("rewards",): 1,
            ("td",): 2,
            ("advantage",): 2,
            ("rewards", "states"): 3,
            ("states", "td"): 4,
            ("advantage", "states"): 4,
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

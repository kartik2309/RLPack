
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
            ("states", "rewards"): 3,
            ("states", "td"): 4,
            ("states", "advantage"): 4,
        }
        ## The mapping between prioritization strategy keywords and prioritization strategy codes. @I{# noqa: E266}
        self.prioritization_strategy_codes = {
            "uniform": 0,
            "proportional": 1,
            "rank-based": 2,
        }

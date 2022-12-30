"""!
@package rlpack.dqn.utils
@brief This package implements the utilities DQN methods.

Currently following classes have been implemented:
    - `DqnAgent`: This is the base class for all DQN agents. This implements common methods and functions for all
        DQN variants. Implemented as rlpack.dqn.utils.dqn_agent.DqnAgent
    - `ProcessPrioritizationParams`: This is a helper class that implements methods to process prioritization
        parameters. Implemented as rlpack.dqn.utils.process_prioritization_params.ProcessPrioritizationParams
"""


from typing import Any, Dict, Union

from rlpack.utils.internal_code_setup import InternalCodeSetup


class ProcessPrioritizationParams(object):
    """
    Class to process prioritization parameters from input for DQN. This will also set default values when
    required. This class is for internal use and in general should not be used by the end user.
    """

    def __init__(self, prioritization_params: Union[Dict[str, Any], None] = None):
        """
        Initialization method for ProcessPrioritizationParams.
        @param prioritization_params: Dict[str, Any]: The prioritization parameters for when
            we use prioritized memory.
        """
        if prioritization_params is None:
            prioritization_params = dict()
        ## The input prioritization parameters from `prioritization_params`. @I{# noqa: E266}
        self._prioritization_params = prioritization_params
        ## The retrieved prioritization strategy. @I{# noqa: E266}
        self._prioritization_strategy = prioritization_params.get(
            "prioritization_strategy", "uniform"
        )
        setup = InternalCodeSetup()
        ## The retrieved prioritization strategy code. @I{# noqa: E266}
        self._prioritization_strategy_code = setup.get_prioritization_code(
            prioritization_strategy=self._prioritization_strategy
        )

    def __call__(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Call method to call internal method for processing prioritization.
        @param batch_size: int: The requested batch size; used in rank-based prioritization to determine the number of
            segments. For others, it is used to determine the number of samples to be sampled.
        @return Dict[str, Any]: The processed prioritization parameters with necessary parameters loaded.
        """
        return self._process_prioritization_params(batch_size)

    def get_prioritization_strategy(self) -> str:
        """
        Gets the prioritization strategy name corresponding to the given prioritization parameters.
        @return: str: The prioritization strategy name.
        """
        return self._prioritization_strategy

    def get_prioritization_strategy_code(self) -> int:
        """
        Gets the prioritization strategy code corresponding to the given prioritization parameters.
        @return: int: The prioritization code.
        """
        return self._prioritization_strategy_code

    def _process_prioritization_params(
        self,
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Method to process the prioritization parameters. This includes sanity check and loading of default
            values of mandatory parameters.
        @param batch_size: int: The requested batch size; used in rank-based prioritization to determine the number of
            segments.
        @return Dict[str, Any]: The processed prioritization parameters with necessary parameters loaded.
        """
        to_anneal_alpha = False
        to_anneal_beta = False
        if (
            self._prioritization_params is not None
            and self._prioritization_strategy_code > 0
        ):
            assert (
                "alpha" in self._prioritization_params.keys()
            ), "`alpha` must be passed when passing self.prioritization_params"
            assert (
                "beta" in self._prioritization_params.keys()
            ), "`beta` must be passed when passing self.prioritization_params"
        else:
            self._prioritization_params = dict()
        alpha = float(self._prioritization_params.get("alpha", -1))
        beta = float(self._prioritization_params.get("beta", -1))
        min_alpha = float(self._prioritization_params.get("min_alpha", 1.0))
        max_beta = float(self._prioritization_params.get("max_beta", 1.0))
        alpha_annealing_frequency = int(
            self._prioritization_params.get("alpha_annealing_frequency", -1)
        )
        beta_annealing_frequency = int(
            self._prioritization_params.get("beta_annealing_frequency", -1)
        )
        alpha_annealing_fn = self._prioritization_params.get(
            "alpha_annealing_fn", self._anneal_alpha_default_fn
        )
        beta_annealing_fn = self._prioritization_params.get(
            "beta_annealing_fn", self._anneal_beta_default_fn
        )
        # Check if to anneal alpha based on input parameters.
        if alpha_annealing_frequency != -1:
            to_anneal_alpha = True
        # Get args and kwargs for to pass to alpha_annealing_fn.
        alpha_annealing_fn_args = self._prioritization_params.get(
            "alpha_annealing_fn_args", tuple()
        )
        alpha_annealing_fn_kwargs = self._prioritization_params.get(
            "alpha_annealing_fn_kwargs", dict()
        )
        # Check if to anneal beta based on input parameters.
        if beta_annealing_frequency != -1:
            to_anneal_beta = True
        # Get args and kwargs for to pass to beta_annealing_fn.
        beta_annealing_fn_args = self._prioritization_params.get(
            "beta_annealing_fn_args", tuple()
        )
        beta_annealing_fn_kwargs = self._prioritization_params.get(
            "beta_annealing_fn_kwargs", dict()
        )
        # Error for proportional based prioritized memory.
        error = float(self._prioritization_params.get("error", 5e-3))
        # Number of segments for rank-based prioritized memory.
        num_segments = self._prioritization_params.get("num_segments", batch_size)
        # Creation of final process dictionary for prioritization_params
        prioritization_params_processed = {
            "prioritization_strategy_code": self._prioritization_strategy_code,
            "to_anneal_alpha": to_anneal_alpha,
            "to_anneal_beta": to_anneal_beta,
            "alpha": alpha,
            "beta": beta,
            "min_alpha": min_alpha,
            "max_beta": max_beta,
            "alpha_annealing_frequency": alpha_annealing_frequency,
            "beta_annealing_frequency": beta_annealing_frequency,
            "alpha_annealing_fn": alpha_annealing_fn,
            "alpha_annealing_fn_args": alpha_annealing_fn_args,
            "alpha_annealing_fn_kwargs": alpha_annealing_fn_kwargs,
            "beta_annealing_fn": beta_annealing_fn,
            "beta_annealing_fn_args": beta_annealing_fn_args,
            "beta_annealing_fn_kwargs": beta_annealing_fn_kwargs,
            "error": error,
            "num_segments": num_segments,
        }
        return prioritization_params_processed

    @staticmethod
    def _anneal_alpha_default_fn(alpha: float, alpha_annealing_factor: float) -> float:
        """
        Protected method to anneal alpha parameter for important sampling weights. This will be called
            every `alpha_annealing_frequency` times. `alpha_annealing_frequency` is a key to be passed in dictionary
            `prioritization_params` argument in the DqnAgent class' constructor. This method is called by default
            to anneal alpha.

        If `alpha_annealing_frequency` is not passed in `prioritization_params`, the annealing of alpha will not take
            place. This method uses another value `alpha_annealing_factor` that must also be passed in
            `prioritization_params`. `alpha_annealing_factor` is typically below 1 to slowly annealed it to
            0 or `min_alpha`.

        @param alpha: float: The input alpha value to anneal.
        @param alpha_annealing_factor: float: The annealing factor to be used to anneal alpha.
        @return float: Annealed alpha.
        """
        alpha *= alpha_annealing_factor
        return alpha

    @staticmethod
    def _anneal_beta_default_fn(beta: float, beta_annealing_factor: float) -> float:
        """
        Protected method to anneal beta parameter for important sampling weights. This will be called
            every `beta_annealing_frequency` times. `beta_annealing_frequency` is a key to be passed in dictionary
            `prioritization_params` argument in the DqnAgent class' constructor.

        If `beta_annealing_frequency` is not passed in `prioritization_params`, the annealing of beta will not take
            place. This method uses another value `beta_annealing_factor` that must also be passed in
            `prioritization_params`. `beta_annealing_factor` is typically above 1 to slowly annealed it to
            1 or `max_beta`

        @param beta: float: The input beta value to anneal.
        @param beta_annealing_factor: float: The annealing factor to be used to anneal beta.
        @return float: Annealed beta.
        """
        beta *= beta_annealing_factor
        return beta

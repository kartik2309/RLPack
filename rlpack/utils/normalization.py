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


from typing import Any, Dict, Optional, Tuple, Union

from rlpack import pytorch
from rlpack.utils.internal_code_setup import InternalCodeSetup


class Normalization:
    """
    Normalization class providing methods for normalization techniques.
    """

    def __init__(
        self,
        apply_norm: Union[int, str],
        custom_min_max: Optional[Tuple[int, int]] = None,
        eps: float = 5e-12,
        p: int = 2,
        dim: int = 0,
    ):
        """
        Initialize Normalization class.
        @param apply_norm: Union[int, str]: apply_norm code for normalization. (Refer rlpack.utils.setup.Setup
            for more information).
        @param custom_min_max: Optional[Tuple[int, int]]: Tuple of custom min and max value
            for min-max normalization. Default: None.
        @param eps: float: The epsilon value for normalization (small value for numerical stability). Default: 5e-12.
        @param p: int: The p-value for p-normalization. Default: 2.
        @param dim: int: The dimension along which normalization is to be applied. Default: 0.
        """
        setup = InternalCodeSetup()
        ## The input `apply_norm` argument; indicating the normalisation to be used. @I{# noqa: E266}
        if isinstance(apply_norm, str):
            apply_norm = setup.get_apply_norm_mode_code(apply_norm=apply_norm)
        setup.check_validity_of_apply_norm_code(apply_norm=apply_norm)
        self.apply_norm = apply_norm
        ## The input `custom_min_max` argument. ## Indicating the custom min-max values for min-max normalisation to be used. @I{# noqa: E266}
        self.custom_min_max = custom_min_max
        ## The input `eps` argument; indicating epsilon to be used for normalisation. @I{# noqa: E266}
        self.eps = eps
        ## The input `p` argument; indicating p-value for p-normalisation. @I{# noqa: E266}
        self.p = p
        ## The input `dim` argument; indicating dimension along which we wish to normalise. @I{# noqa: E266}
        self.dim = dim
        ## The statistics dictionary to store useful statistics for each quantity. @I{# noqa: E266}
        self.statistics = dict()
        return

    def apply_normalization(self, tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        All encompassing function to perform normalization depending on the instance's apply_norm code.
        @param tensor: pytorch.Tensor: The tensor to apply normalization on.
        @return pytorch.Tensor: The normalized tensor.
        """
        if self.apply_norm == -1:
            return tensor
        elif self.apply_norm == 0:
            return self.min_max_normalization(tensor)
        elif self.apply_norm == 1:
            return self.standardization(tensor)
        elif self.apply_norm == 2:
            return self.p_normalization(tensor)
        else:
            raise ValueError("Invalid value of normalization received!")

    def apply_normalization_pre(
        self, tensor: pytorch.Tensor, quantity: str
    ) -> pytorch.Tensor:
        """
        All encompassing function to perform normalization depending on the instance's apply_norm code. Applies
        normalization as per pre-computed statistics. Note that this method cannot be applied to p-normalization.
        @param tensor: pytorch.Tensor: The tensor to apply normalization on.
        @param quantity: str: The quantity for which is being normalized. This must be present in `statistics_dict`
            with corresponding statistics.
        @return pytorch.Tensor: The normalized tensor.
        """
        if quantity not in self.statistics.keys():
            raise ValueError(
                f"The given quantity `{quantity}` does not have pre-computed statistics associated with it! "
                f"Perform exploration or remove `{quantity}` from `apply_norm_to`."
            )
        if self.apply_norm == -1:
            return tensor
        elif self.apply_norm == 0:
            return self.min_max_normalization_pre(tensor, quantity)
        elif self.apply_norm == 1:
            return self.standardization_pre(tensor, quantity)
        elif self.apply_norm == 2:
            raise ValueError(
                "`p-normalization` cannot be used in pre methods for pre-computed statistics!"
            )
        else:
            raise ValueError("Invalid value of normalization received!")

    def apply_normalization_pre_silent(
        self, tensor: pytorch.Tensor, quantity: str
    ) -> pytorch.Tensor:
        """
        All encompassing function to perform normalization depending on the instance's apply_norm code. This method
        is silent version of `apply_normalization_pre` and performs no normalization if pre-computed statistics
        are not present in `statistics_dict` at the moment. Note that this will still raise an error if p-normalization
        is used.
        @param tensor: pytorch.Tensor: The tensor to apply normalization on.
        @param quantity: str: The quantity for which is being normalized. This must be present in `statistics_dict`
            with corresponding statistics.
        @return pytorch.Tensor: The normalized tensor.
        """
        if quantity not in self.statistics.keys():
            return tensor
        return self.apply_normalization_pre(tensor, quantity)

    def update_statistics(self, quantity: str, statistics: Dict[str, pytorch.Tensor]):
        if quantity in self.statistics.keys():
            self.statistics[quantity].update(statistics)
        else:
            self.statistics[quantity] = statistics

    def min_max_normalization(self, tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        Method to apply min-max normalization.
        @param tensor: pytorch.Tensor: The input tensor to be min-max normalized.
        @return (pytorch.Tensor): The normalized tensor.
        """
        if self.custom_min_max is not None:
            _min = self.custom_min_max[0]
            _max = self.custom_min_max[1]
        else:
            _min = pytorch.min(tensor, dim=self.dim)
            _max = pytorch.max(tensor, dim=self.dim)
        tensor = (tensor - _min) / (_max - _min + self.eps)
        return tensor

    def min_max_normalization_pre(
        self, tensor: pytorch.Tensor, quantity: str
    ) -> pytorch.Tensor:
        """
        Method to apply min-max normalization with pre-computed statistics. If `custom_min_max` was passed, it will
        be used for normalization.
        @param tensor: pytorch.Tensor: The input tensor to be min-max normalized.
        @param quantity: str: The quantity for which is being normalized. This must be present in `statistics_dict`
            with corresponding min and max value.
        @return (pytorch.Tensor): The normalized tensor.
        """
        if self.custom_min_max is not None:
            _min = self.custom_min_max[0]
            _max = self.custom_min_max[1]
        else:
            _min = self.statistics[quantity]["min"]
            _max = self.statistics[quantity]["max"]
        tensor = (tensor - _min) / (_max - _min + self.eps)
        return tensor

    def standardization(self, tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        Method to standardize the input tensor.
        @param tensor: pytorch.Tensor: he input tensor to be standardized.
        @return pytorch.Tensor: The standardized tensor.
        """
        _mean = pytorch.mean(tensor, dim=self.dim)
        _std = pytorch.std(tensor, dim=self.dim)
        tensor = (tensor - _mean) / (_std + self.eps)
        return tensor

    def standardization_pre(
        self, tensor: pytorch.Tensor, quantity: str
    ) -> pytorch.Tensor:
        """
        Method to standardize the input tensor with pre-computed statistics.
        @param tensor: pytorch.Tensor: he input tensor to be standardized.
        @param quantity: str: The quantity for which is being normalized. This must be present in `statistics_dict`
            with corresponding mean and std values.
        @return pytorch.Tensor: The standardized tensor.
        """
        _mean = self.statistics[quantity]["mean"]
        _std = self.statistics[quantity]["std"]
        tensor = (tensor - _mean) / (_std + self.eps)
        return tensor

    def p_normalization(self, tensor: pytorch.Tensor) -> pytorch.Tensor:
        """
        The p-normalization method.
        @param tensor: pytorch.Tensor: The input tensor to be standardized.
        @return pytorch.Tensor: The p-normalized tensor.
        """
        tensor = pytorch.linalg.norm(tensor, dim=self.dim, ord=self.p)
        return tensor

    def get_state_dict(self) -> Dict[str, Any]:
        """
        The getstate method to get the dictionary of attributes.
        @return: Dict[str, Any]: The dictionary of current attributes.
        """
        return self.__dict__

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        setstate method for setting attributes for the Normalization class
        @param state: Dict[str, Any]: The dictionary of attributes and their corresponding values.
        """
        self.__dict__ = state

from typing import List, Union

from rlpack import pytorch
from rlpack.exploration.utils.exploration import Exploration
from rlpack.exploration.utils.variance_estimator import VarianceEstimator
from rlpack.models.utils.mlp_feature_extractor import MlpFeatureExtractor
from rlpack.utils import Activation


class StateDependentExploration(Exploration):
    """
    StateDependentExploration performs exploration based on state values (or features). For more information
    on the algorithm refer [here](https://arxiv.org/abs/2005.05719).


    This implementation uses a "Variance Estimator" (rlpack.exploration.utils.variance_estimator.VarianceEstimator),
    which samples the exploration matrix when `reset`is called. Note that since std is learned, parameters
    must be included in the optimizer and computational graph must be attached for proper gradient flow.
    """

    def __init__(
        self,
        initial_std: pytorch.Tensor,
        features_dim: int,
        hidden_sizes: Union[List[int], None] = None,
        activation: Union[Activation, List[Activation]] = pytorch.nn.ReLU(),
        is_log: bool = True,
    ):
        """
        Initialization for StateDependentExploration class.
        @param initial_std: pytorch.Tensor: The initial standard deviation to be used. This will be used as parameter
            for Variance Estimator and will be learned. It must be of shape of the output action.
        @param features_dim: int: The input feature dimension (feature of state).
        @param hidden_sizes: Union[List[int], None]: The hidden sizes for the additional feature extraction
            Linear Layers to be used. By default, no layers are created. Default: None.
        @param activation: Union[Activation, List[Activation]]: The PyTorch activations. Activations will be used
            in feature extractor (if `hidden_size` is not None). Activation is also applied to the input of
            Variance Estimator. If list is passed, first activation is used for feature extractor (if valid) and last
            activation is used on the input before passing through Variance Estimator. Default: pytorch.nn.ReLU()
        @param is_log: Boolean flag indicating whether the `input_std` is log(std). Default: True
        """
        super(StateDependentExploration, self).__init__()
        self.feature_extractor = None
        self._use_feature_extractor = False
        if not isinstance(activation, list):
            self.activation = [activation]
        else:
            self.activation = activation
        if hidden_sizes is not None:
            self.feature_extractor = MlpFeatureExtractor(
                hidden_sizes=hidden_sizes, activation=self.activation[0]
            )
            self._use_feature_extractor = True
        self.variance_estimator = VarianceEstimator(
            initial_std=initial_std, features_dim=features_dim, is_log=is_log
        )

    def sample(self, *args, **kwargs) -> pytorch.Tensor:
        """
        This method is overriden here to raise a NotImplementedError. Since this exploration tool contains learnable
        parameters, `rsample` is to be used.
        @param args: Arbitrary positional arguments.
        @param kwargs: Arbitrary keyword arguments.
        @return: pytorch.Tensor: Sampled tensor. Note that this just shows the signature of the method and
            is not valid if called.
        """
        raise NotImplementedError(
            "`sample` method has not been implemented for StateDependentExploration! "
            "It should be called in an Agent!"
        )

    def rsample(self, features: pytorch.Tensor, *args, **kwargs) -> pytorch.Tensor:
        """
        This method is overriden to provide implementation of Generalized State Dependent Estimation Noise. This
        method ensures that operations are attached to computation graph.
        @param features: pytorch.Tensor: The input Tensor of features.
        @param args: Arbitrary positional arguments.
        @param kwargs: Arbitrary keyword arguments.
        @return: pytorch.Tensor: The tensor of noise to be added to action.
        """
        x = features.detach()
        if self._use_feature_extractor:
            x = self.feature_extractor(x)
        x = self.activation[-1](x)
        x = self.variance_estimator(x)
        return x

    def reset(self) -> None:
        """
        Resets the SDE exploration tool. This calls reset in Variance Estimator which resamples the exploration
        matrix.
        """
        self.variance_estimator.reset()

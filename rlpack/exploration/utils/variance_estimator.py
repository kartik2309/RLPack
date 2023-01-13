from rlpack import pytorch, pytorch_distributions


class VarianceEstimator(pytorch.nn.Module):
    """
    A utility pytorch.nn.Module which performs variance estimation by learning the standard deviation. This is used
    in StateDependentExploration (rlpack.exploration.state_dependent_exploration.StateDependentExploration).
    """

    def __init__(
        self, initial_std: pytorch.Tensor, features_dim: int, is_log: bool = True
    ):
        """
        The initialization method for VarianceEstimator.
        @param initial_std: pytorch.Tensor: The initial standard deviation. This tensor is used as module parameter
            and is learned. It must have same dimension as the expected action.
        @param features_dim: int: The input feature dimension (feature of state).
        @param is_log: Boolean flag indicating whether the `input_std` is log(std). Default: True
        """
        super(VarianceEstimator, self).__init__()
        self.std = pytorch.nn.Parameter(initial_std)
        self.features_dim = features_dim
        self.is_log = is_log
        self.exploration_matrix = self._create_exploration_matrix()

    def forward(self, features: pytorch.Tensor) -> pytorch.Tensor:
        """
        Forward method for VarianceEstimator for forward pass.
        @param features: pytorch.Tensor: The input features. This is multiplied with the sampled exploration matrix.
        @return: pytorch.Tensor: The final noise tensor.
        """
        return features @ self.exploration_matrix

    def reset(self) -> None:
        """
        Resets the exploration matrix. The exploration matrix is resampled from the new zero centered normal
        distribution with new std.
        """
        self.exploration_matrix = self._create_exploration_matrix()

    def get_variance(self) -> pytorch.Tensor:
        """
        Gets the current estimated variance.
        @return: pytorch.Tensor: The variance tensor.
        """
        if self.is_log:
            _std = pytorch.exp(self.std)
        else:
            _std = self.std
        variance = pytorch.square(_std)
        return variance

    def _create_exploration_matrix(self) -> pytorch.Tensor:
        """
        Creates the exploration matrix by sampling from the zero centered normal distribution with current
        std.
        @return: pytorch.Tensor: New exploration matrix.
        """
        distribution = self._create_distribution()
        exploration_matrix = distribution.rsample(sample_shape=(self.features_dim,))
        return exploration_matrix

    def _create_distribution(self) -> pytorch_distributions.Distribution:
        """
        Creates the zero centered normal distribution with current std.
        @return: pytorch_distributions.Distribution: The normal distribution object.
        """
        if self.is_log:
            _std = pytorch.exp(self.std)
        else:
            _std = self.std
        distribution = pytorch_distributions.Normal(pytorch.zeros_like(_std), _std)
        return distribution

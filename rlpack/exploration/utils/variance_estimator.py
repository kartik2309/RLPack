from rlpack import pytorch, pytorch_distributions


class VarianceEstimator(pytorch.nn.Module):
    def __init__(
        self, initial_std: pytorch.Tensor, features_dim: int, is_log: bool = True
    ):
        super(VarianceEstimator, self).__init__()
        self.std = pytorch.nn.Parameter(initial_std)
        self.features_dim = features_dim
        self.is_log = is_log
        self.exploration_matrix = self._create_exploration_matrix()

    def forward(self, features: pytorch.Tensor) -> pytorch.Tensor:
        return features @ self.exploration_matrix

    def reset(self) -> None:
        self.exploration_matrix = self._create_exploration_matrix()

    def _create_exploration_matrix(self) -> pytorch.Tensor:
        distribution = self._create_distribution()
        exploration_matrix = distribution.rsample(sample_shape=(self.features_dim,))
        return exploration_matrix

    def _create_distribution(self) -> pytorch_distributions.Distribution:
        if self.is_log:
            _std = pytorch.exp(self.std)
        else:
            _std = self.std
        distribution = pytorch_distributions.Normal(pytorch.zeros_like(_std), _std)
        return distribution

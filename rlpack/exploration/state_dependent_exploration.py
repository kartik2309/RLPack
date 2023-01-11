from typing import List, Union

from rlpack import pytorch
from rlpack.exploration.utils.exploration import Exploration
from rlpack.exploration.utils.variance_estimator import VarianceEstimator
from rlpack.models.utils.mlp_feature_extractor import MlpFeatureExtractor


class StateDependentExploration(Exploration):
    def __init__(
        self,
        initial_std: pytorch.Tensor,
        features_dim: int,
        hidden_sizes: Union[List[int], None] = None,
        is_log: bool = True,
    ):
        super(StateDependentExploration, self).__init__()
        self.feature_extractor = None
        self._use_feature_extractor = False
        if hidden_sizes is not None:
            self.feature_extractor = MlpFeatureExtractor(hidden_sizes=hidden_sizes)
            self._use_feature_extractor = True
        self.variance_estimator = VarianceEstimator(
            initial_std=initial_std, features_dim=features_dim, is_log=is_log
        )

    def sample(self, *args, **kwargs) -> pytorch.Tensor:
        raise NotImplementedError(
            "`sample` method has not been implemented for StateDependentExploration! "
            "It should be called in an Agent!"
        )

    def rsample(self, features: pytorch.Tensor, *args, **kwargs) -> pytorch.Tensor:
        x = features.detach()
        if self._use_feature_extractor:
            x = self.feature_extractor(x)
        x = self.variance_estimator(x)
        return x

    def reset(self) -> None:
        self.variance_estimator.reset()

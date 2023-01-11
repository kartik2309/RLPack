"""!
@package rlpack.models
@brief This package implements the [in-built](@ref models/index.md) models.


Currently following models have been implemented:
    - `Mlp`: MLP model implemented as rlpack.models.mlp.Mlp. More information can be found
        [here](@ref models/in_built/mlp.md).
    - `ActorCriticMlpPolicy`: MLP based Policy model for Actor-Critic Methods implemented as
        rlpack.models.actor_critic_mlp_policy.ActorCriticMlpPolicy. More information can be found
        [here](@ref models/in_built/actor_critic_mlp_policy.md).

Following packages are part of models:
    - `utils`: A package utilities for models package.
"""


from typing import List

from rlpack import pytorch
from rlpack.models.utils.mlp_feature_extractor import MlpFeatureExtractor
from rlpack.utils import Activation
from rlpack.utils.base.model import Model


class Mlp(Model):
    """
    This class is a PyTorch Model implementing the MLP model for 1-D or 2-D state values.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        num_actions: int,
        sequence_length: int = 1,
        activation: Activation = pytorch.nn.ReLU(),
        dropout: float = 0.5,
    ):
        """
        Initialize Mlp model.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param num_actions: int: The number of actions for the environment.
        @param sequence_length: int: The sequence length of the expected tensor. Default: 1
        @param activation: Activation: The activation function class for the model. Must be an initialized
            activation object from PyTorch's nn (torch.nn) module.
        @param dropout: float: The dropout to be used in the final Linear (FC) layer.
        """
        super(Mlp, self).__init__()
        ## The feature extractor instance of rlpack.models._mlp_feature_extractor.MlpFeatureExtractor. @I{# noqa: E266}
        self.mlp_feature_extractor = MlpFeatureExtractor(
            sequence_length=sequence_length,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        ## The input dropout probability. @I{# noqa: E266}
        self.dropout = pytorch.nn.Dropout(dropout)
        ## The object to flatten the output fo feature extractor. @I{# noqa: E266}
        self.flatten = pytorch.nn.Flatten(start_dim=1, end_dim=-1)
        ## The final head to produce logits for given action. @I{# noqa: E266}
        self.final_head = pytorch.nn.Linear(
            in_features=hidden_sizes[-1], out_features=num_actions
        )

    def forward(self, x: pytorch.Tensor) -> pytorch.Tensor:
        """
        The forwards method of the nn.Module.
        @param x: pytorch.Tensor: The model input.
        @return pytorch.Tensor: The model output (logits).
        """
        x = self.mlp_feature_extractor(x)
        x = self.dropout(x)
        x = self.final_head(x)
        x = self.flatten(x)
        return x

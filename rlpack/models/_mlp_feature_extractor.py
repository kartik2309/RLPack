"""!
@package rlpack.models
@brief This package implements the [in-built](@ref models/index.md) models.


Currently following models have been implemented:
    - Mlp: MLP model implemented as rlpack.models.mlp.Mlp. More information can be found
        [here](@ref models/in_built/mlp.md).
    - ActorCriticMlpPolicy: MLP based Policy model for Actor-Critic Methods implemented as
        rlpack.models.actor_critic_mlp_policy.ActorCriticMlpPolicy. More information can be found
        [here](@ref models/in_built/actor_critic_mlp_policy.md).
    - _MlpFeatureExtractor: MLP based feature extraction model implemented as
        rlpack.models._mlp_feature_extractor._MlpFeatureExtractor. Only to be used internally.
"""


from typing import List

from rlpack import pytorch
from rlpack.utils import Activation


class _MlpFeatureExtractor(pytorch.nn.Module):
    """
    This class is a PyTorch Model implementing the MLP based feature extractor for 1-D or 2-D state values.
    """

    def __init__(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        activation: Activation = pytorch.nn.ReLU(),
        dropout: float = 0.5,
    ):
        """
        Initialize MlpFeatureExtractor model.
        @param sequence_length: int: The sequence length of the expected tensor.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param activation: Activation: The activation function class for the model. Must be an initialized
            activation object from PyTorch's nn (torch.nn) module.
        @param dropout: float: The dropout to be used in the final Linear (FC) layer.
        """
        super(_MlpFeatureExtractor, self).__init__()
        ## The input sequence length of expected tensor. @I{# noqa: E266}
        self.sequence_length = sequence_length
        ## The input hidden sizes for each layer. @I{# noqa: E266}
        self.hidden_sizes = hidden_sizes.copy()
        ## The input activation function. @I{# noqa: E266}
        self.activation = activation
        ## The input dropout probability. @I{# noqa: E266}
        self.dropout = pytorch.nn.Dropout(dropout)
        ## The number of layers/blocks of MLP. @I{# noqa: E266}
        self.num_blocks = len(self.hidden_sizes) - 1
        ## The ModuleDict of Linear Layers. @I{# noqa: E266}
        self.linear_module_dict = pytorch.nn.ModuleDict(
            {
                f"layer_{idx}": pytorch.nn.Linear(
                    in_features=self.hidden_sizes[idx],
                    out_features=self.hidden_sizes[idx + 1],
                )
                for idx in range(self.num_blocks)
            }
        )

    def forward(self, x: pytorch.Tensor) -> pytorch.Tensor:
        """
        The forwards method of the nn.Module.
        @param x: pytorch.Tensor: The model input.
        @return pytorch.Tensor: The model output (logits).
        """
        for layer_idx, layer_info in enumerate(self.linear_module_dict.items()):
            name, layer = layer_info
            x = layer(x)
            x = self.activation(x)
        x = self.dropout(x)
        return x

"""!
@package rlpack.models.utils
@brief This package implements the utilities for [in-built](@ref models/index.md) models.


Currently following classes have been implemented for utilities:
    - `_MlpFeatureExtractor`: MLP based feature extraction model implemented as
        rlpack.models._mlp_feature_extractor._MlpFeatureExtractor. Only to be used internally.
"""


from typing import List

from rlpack import pytorch
from rlpack.utils import Activation
from rlpack.utils.base.model import Model


class MlpFeatureExtractor(Model):
    """
    This class is a PyTorch Model implementing the MLP based feature extractor for 1-D or 2-D state values.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        sequence_length: int = 1,
        activation: Activation = pytorch.nn.ReLU(),
    ):
        """
        Initialize MlpFeatureExtractor model.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param sequence_length: int: The sequence length of the expected tensor. Default: 1
        @param activation: Activation: The activation function class for the model. Must be an initialized
            activation object from PyTorch's nn (torch.nn) module.
        """
        super(MlpFeatureExtractor, self).__init__()
        ## The input sequence length of expected tensor. @I{# noqa: E266}
        self.sequence_length = sequence_length
        ## The input hidden sizes for each layer. @I{# noqa: E266}
        self.hidden_sizes = hidden_sizes.copy()
        ## The input activation function. @I{# noqa: E266}
        self.activation = activation
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
        return x

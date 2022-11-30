from typing import List

from rlpack import pytorch
from rlpack.utils import Activation


class MlpFeatureExtractor(pytorch.nn.Module):
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
        Initialize MlpFeatureExtractor model
        :param sequence_length: int: The sequence length of the expected tensor
        :param hidden_sizes: List[int]: The list of hidden sizes for each layer
        :param activation: Activation: The activation function class for the model. Must be an initialized
            activation object from PyTorch's nn (torch.nn) module
        :param dropout: float: The dropout to be used in the final Linear (FC) layer
        """
        super(MlpFeatureExtractor, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_sizes = hidden_sizes.copy()
        self.activation = activation
        self.dropout = pytorch.nn.Dropout(dropout)
        self.num_blocks = len(self.hidden_sizes) - 1
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
        :param x: pytorch.Tensor: The model input.
        :return: pytorch.Tensor: The model output (logits).
        """
        for layer_idx, layer_info in enumerate(self.linear_module_dict.items()):
            name, layer = layer_info
            x = layer(x)
            x = self.activation(x)
        x = self.dropout(x)
        return x

from typing import List

from rlpack import pytorch
from rlpack.models.mlp_feature_extractor import MlpFeatureExtractor
from rlpack.utils import Activation


class ActorCriticMlpPolicy(pytorch.nn.Module):
    """
    This class is a PyTorch Model implementing the MLP based Actor-Critic Policy.
    """

    def __init__(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        num_actions: int,
        activation: Activation = pytorch.nn.ReLU(),
        dropout: float = 0.5,
    ):
        """
        Initialize ActorCriticMlpPolicy model
        :param sequence_length: int: The sequence length of the expected tensor
        :param hidden_sizes: List[int]: The list of hidden sizes for each layer
        :param num_actions: int: The number of actions for the environment
        :param activation: Activation: The activation function class for the model. Must be an initialized
            activation object from PyTorch's nn (torch.nn) module
        :param dropout: float: The dropout to be used in the final Linear (FC) layer
        """
        super(ActorCriticMlpPolicy, self).__init__()
        self.mlp_feature_extractor = MlpFeatureExtractor(
            sequence_length=sequence_length,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
        )
        self.flatten = pytorch.nn.Flatten(start_dim=1, end_dim=-1)
        self.actor_head = pytorch.nn.Linear(
            in_features=hidden_sizes[-1], out_features=num_actions
        )
        self.critic_head = pytorch.nn.Linear(
            in_features=hidden_sizes[-1], out_features=1
        )

    def forward(self, x):
        """
        The forwards method of the nn.Module.
        :param x: pytorch.Tensor: The model input.
        :return: pytorch.Tensor: The model output (logits).
        """
        x = self.mlp_feature_extractor(x)
        x = self.flatten(x)
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        return action_logits, state_value

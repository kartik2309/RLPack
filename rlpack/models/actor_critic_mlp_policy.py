"""!
@package rlpack.models
@brief This package implements the [in-built](@ref models/index.md) models.


Currently following models have been implemented:
    - `Mlp`: MLP model implemented as rlpack.models.mlp.Mlp. More information can be found
        [here](@ref models/in_built/mlp.md).
    - `ActorCriticMlpPolicy`: MLP based Policy model for Actor-Critic Methods implemented as
        rlpack.models.actor_critic_mlp_policy.ActorCriticMlpPolicy. More information can be found
        [here](@ref models/in_built/actor_critic_mlp_policy.md).
    - `_MlpFeatureExtractor`: MLP based feature extraction model implemented as
        rlpack.models._mlp_feature_extractor._MlpFeatureExtractor. Only to be used internally.
"""


from typing import List, Tuple, Union

from rlpack import pytorch
from rlpack.models._mlp_feature_extractor import _MlpFeatureExtractor
from rlpack.utils import Activation


class ActorCriticMlpPolicy(pytorch.nn.Module):
    """
    This class is a PyTorch Model implementing the MLP based Actor-Critic Policy.
    """

    def __init__(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        action_space: Union[int, List[Union[int, List[int]]]],
        activation: Union[Activation, List[Activation]] = pytorch.nn.ReLU(),
        dropout: float = 0.5,
        share_network: bool = False,
    ):
        """
        Initialize ActorCriticMlpPolicy model.
        @param sequence_length: int: The sequence length of the expected tensor.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param action_space: Union[int, List[Union[int, List[int]]]]: The action space of the environment. If
            discrete action set is used, number of actions can be passed. If continuous action space is used,
            a list must be passed with first element representing the output features from model, second
            representing the shape of action to be sampled.
        @param activation: Union[Activation, List[Activation]]: The activation function class(es) for the model.
            Must be an initialized activation object from PyTorch's nn (torch.nn) module. If a list is passed, List
            must be of length [1, 3], first activation for feature extractor, second for actor head and third for
            critic head.
        @param dropout: float: The dropout to be used in the final Linear (FC) layer.
        @param share_network: bool: Flag indicating whether to use the shared network for actor and critic or
            separate networks. Default: False
        """
        super(ActorCriticMlpPolicy, self).__init__()
        ## FLag indicating whether to apply activation to output of actor head or not. @I{# noqa: E266}
        self._apply_actor_activation = False
        ## FLag indicating whether to apply activation to output of critic head or not. @I{# noqa: E266}
        self._apply_critic_activation = False
        # Process `activation`
        activation = self._process_activation(activation)
        # Process `action_space`
        out_features = self._process_action_space(action_space)
        self.share_network = share_network
        ## The feature extractor instance of rlpack.models._mlp_feature_extractor._MlpFeatureExtractor. @I{# noqa: E266}
        ## This will be None if network is not shared. @I{# noqa: E266}
        self.mlp_feature_extractor = None
        ## The feature extractor for actor model. This will always be None if network is shared. @I{# noqa: E266}
        self.actor_feature_extractor = None
        ## The feature extractor for critic model. This will always be None if network is shared. @I{# noqa: E266}
        self.critic_feature_extractor = None
        if not share_network:
            self._set_non_shared_network_attributes(
                sequence_length, hidden_sizes, activation, dropout
            )
        else:
            self._set_shared_network_attributes(
                sequence_length, hidden_sizes, activation, dropout
            )
        ## The final head for actor; creates logits/parameters for actions @I{# noqa: E266}
        self.actor_head = pytorch.nn.Linear(
            in_features=hidden_sizes[-1],
            out_features=out_features,
        )
        ## The final head for critic; creates the state value. @I{# noqa: E266}
        self.critic_head = pytorch.nn.Linear(
            in_features=hidden_sizes[-1], out_features=1
        )
        if len(activation) > 1:
            self.actor_activation = activation[1]
            self._apply_actor_activation = True
        if len(activation) > 2:
            self.value_activation = activation[2]
            self._apply_critic_activation = True
        ## The object to flatten the output fo feature extractor. @I{# noqa: E266}
        self.flatten = pytorch.nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x: pytorch.Tensor) -> Tuple[pytorch.Tensor, pytorch.Tensor]:
        """
        The forwards method of the nn.Module.
        @param x: pytorch.Tensor: The model input.
        @return Tuple[pytorch.Tensor, pytorch.Tensor]: The tuple of actor and critic outputs.
        """
        if not self.share_network:
            action_logits, state_value = self._run_non_shared_forward(x)
        else:
            action_logits, state_value = self._run_shared_forward(x)
        if self._apply_actor_activation:
            action_logits = self.actor_activation(action_logits)
        if self._apply_critic_activation:
            state_value = self.value_activation(state_value)
        return action_logits, state_value

    def _run_shared_forward(
        self, x: pytorch.Tensor
    ) -> Tuple[pytorch.Tensor, pytorch.Tensor]:
        """
        The forwards method of the nn.Module when actor and critic share network
        @param x: pytorch.Tensor: The model input.
        @return Tuple[pytorch.Tensor, pytorch.Tensor]: The tuple of actor and critic outputs.
        """
        x = self.mlp_feature_extractor(x)
        x = self.flatten(x)
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        return action_logits, state_value

    def _run_non_shared_forward(
        self, x: pytorch.Tensor
    ) -> Tuple[pytorch.Tensor, pytorch.Tensor]:
        """
        The forwards method of the nn.Module when actor and critic do not share network
        @param x: pytorch.Tensor: The model input.
        @return Tuple[pytorch.Tensor, pytorch.Tensor]: The tuple of actor and critic outputs.
        """
        action_logits = self.actor_feature_extractor(x)
        state_value = self.critic_feature_extractor(x)
        action_logits = self.flatten(action_logits)
        state_value = self.flatten(state_value)
        action_logits = self.actor_head(action_logits)
        state_value = self.critic_head(state_value)
        return action_logits, state_value

    def _set_shared_network_attributes(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        activation: List[Activation],
        dropout: float = 0.5,
    ) -> None:
        """
        Sets appropriate attributes to create shared network.
        @param sequence_length: int: The sequence length of the expected tensor.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param activation: List[Activation]: List of activations to be used.
        @param dropout: float: The dropout to be used in the final Linear (FC) layer.
        """
        self.mlp_feature_extractor = _MlpFeatureExtractor(
            sequence_length=sequence_length,
            hidden_sizes=hidden_sizes,
            activation=activation[0],
            dropout=dropout,
        )

    def _set_non_shared_network_attributes(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        activation: List[Activation],
        dropout: float = 0.5,
    ) -> None:
        """
        Sets appropriate attributes to create non-shared network.
        @param sequence_length: int: The sequence length of the expected tensor.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param activation: List[Activation]: List of activations to be used.
        @param dropout: float: The dropout to be used in the final Linear (FC) layer.
        """
        self.actor_feature_extractor = _MlpFeatureExtractor(
            sequence_length=sequence_length,
            hidden_sizes=hidden_sizes,
            activation=activation[0],
            dropout=dropout,
        )
        self.critic_feature_extractor = _MlpFeatureExtractor(
            sequence_length=sequence_length,
            hidden_sizes=hidden_sizes,
            activation=activation[0],
            dropout=dropout,
        )

    @staticmethod
    def _process_action_space(
        action_space: Union[int, List[Union[int, List[int]]]],
    ) -> int:
        """
        Processes `action_space` for use by the model. If checks are passed, returns the output features for
        actor head.
        @param action_space: Union[int, List[Union[int, List[int]]]]: The action space of the environment. If
            discrete action set is used, number of actions can be passed. If continuous action space is used,
            a list must be passed with first element representing the output features from model, second
            representing the shape of action to be sampled.

        @return: Tuple[Union[int, List[Union[int, List[int]]]], Union[Activation, List[Activation]]]: The corrected
            values for action_space and activation if required.
        """
        if isinstance(action_space, list):
            if len(action_space) != 2:
                raise ValueError(
                    f"length of `action_space` should be 2; "
                    f"first element representing the output features from model, "
                    f"second representing the shape of action to be sampled."
                    f"Received {action_space}"
                )
            if not isinstance(action_space[1], list):
                raise TypeError(
                    f"The second element of `action_space must"
                    f"represent the shape of action to be sampled and must a list. "
                    f"Received {action_space}"
                )
            out_features = action_space[0]
        elif isinstance(action_space, int):
            out_features = action_space
        else:
            raise TypeError(
                f"Expected argument `activation` to be of type {int} or {list} but received {type(action_space)}"
            )
        return out_features

    @staticmethod
    def _process_activation(
        activation: Union[Activation, List[Activation]]
    ) -> List[Activation]:
        """
        Processes `activation` for use by the model.
        @param activation: Union[Activation, List[Activation]]: The activation function class(es) for the model.
            Must be an initialized activation object from PyTorch's nn (torch.nn) module. If a list is passed, List
            must be of length [1, 3], first activation for feature extractor, second for actor head and third for
            critic head.
        """
        if isinstance(activation, list):
            if not 0 < len(activation) <= 3:
                raise ValueError(
                    "Activation must be a list of either one, two or three activation; "
                    "first for feature extractor; second for actor head; third for critic head"
                )
        elif isinstance(activation, pytorch.nn.Module):
            activation = [activation]
        else:
            raise TypeError(
                f"Expected argument `activation` to be of type {Activation} or {list} but received {type(activation)}"
            )
        return activation

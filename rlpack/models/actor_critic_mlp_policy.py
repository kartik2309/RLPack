"""!
@package rlpack.models
@brief This package implements the [in-built](@ref models/index.md) models.


Currently following models have been implemented:
    - `Mlp`: MLP model implemented as rlpack.models.mlp.Mlp. More information can be found
        [here](@ref models/in_built/mlp.md).
    - `ActorCriticMlpPolicy`: MLP based Policy model for Actor-Critic Methods implemented as
        rlpack.models.actor_critic_mlp_policy.ActorCriticMlpPolicy. More information can be found
        [here](@ref models/in_built/actor_critic_mlp_policy.md).
"""


from typing import List, Tuple, Union

from rlpack import pytorch
from rlpack.models.utils._mlp_feature_extractor import _MlpFeatureExtractor
from rlpack.utils import Activation


class ActorCriticMlpPolicy(pytorch.nn.Module):
    """
    This class is a PyTorch Model implementing the MLP based Actor-Critic Policy.
    """

    def __init__(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        action_space: Union[int, Tuple[int, Union[List[int], None]]],
        activation: Union[Activation, List[Activation]] = pytorch.nn.ReLU(),
        dropout: float = 0.5,
        share_network: bool = False,
        use_actor_projection: bool = False,
    ):
        """
        Initialize ActorCriticMlpPolicy model.
        @param sequence_length: int: The sequence length of the expected tensor.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param action_space: Union[int, Tuple[int, Union[List[int], None]]]: The action space of the environment.
            - If discrete action set is used, number of actions can be passed.
            - If continuous action space is used, a list must be passed with first element representing
            the output features from model, second element representing the shape of action to be sampled. Second
            element can be an empty list or None, if you wish to sample the default no. of samples.
        @param activation: Union[Activation, List[Activation]]: The activation function class(es) for the model.
            Must be an initialized activation object from PyTorch's nn (torch.nn) module. If a list is passed, List
            must be of length [1, 3], first activation for feature extractor, second for actor head and third for
            critic head.
        @param dropout: float: The dropout to be used in the final Linear (FC) layer.
        @param share_network: bool: Flag indicating whether to use the shared network for actor and critic or
            separate networks. Default: False
        @param use_actor_projection: bool: Flag indicating whether to use projection for actor. Projection is applied
            to output of feature extractor of actor model.
        """
        super(ActorCriticMlpPolicy, self).__init__()
        ## The input `share_network`. @I{# noqa: E266}
        self.share_network = share_network
        ## The input `use_actor_projection`. @I{# noqa: E266}
        self.use_actor_projection = use_actor_projection
        ## Flag indicating whether to apply activation to output of actor head or not. @I{# noqa: E266}
        self._apply_actor_activation = False
        ## Flag indicating whether to apply activation to output of critic head or not. @I{# noqa: E266}
        self._apply_critic_activation = False
        ## The feature extractor instance of rlpack.models._mlp_feature_extractor._MlpFeatureExtractor. @I{# noqa: E266}
        ## This will be None if network is not shared. @I{# noqa: E266}
        self.mlp_feature_extractor = None
        ## The feature extractor for actor model. This will always be None if network is shared. @I{# noqa: E266}
        self.actor_feature_extractor = None
        ## The feature extractor for critic model. This will always be None if network is shared. @I{# noqa: E266}
        self.critic_feature_extractor = None
        ## The projection vector for actor. This will be None if `use_actor_projection` is False. @I{# noqa: E266}
        self.actor_projector = None
        ## The core activation function to be used. This will be used in feature extractor @I{# noqa: E266}
        ## and between feature extractor and heads. @I{# noqa: E266}
        self.activation_core = activation[0]
        ## The activation function for the actor's output. @I{# noqa: E266}
        self.actor_activation = None
        ## The activation function for the critic's output. @I{# noqa: E266}
        self.value_activation = None
        if not share_network:
            self._set_non_shared_network_attributes(
                sequence_length, hidden_sizes, activation
            )
        else:
            self._set_shared_network_attributes(
                sequence_length, hidden_sizes, activation
            )
        # Process `activation`
        activation = self._process_activation(activation, use_actor_projection)
        # Process `action_space`
        out_features = self._process_action_space(action_space)
        ## The final head for actor; creates logits/parameters for actions. @I{# noqa: E266}
        self.actor_head = pytorch.nn.Linear(
            in_features=hidden_sizes[-1],
            out_features=out_features,
        )
        if use_actor_projection:
            self.actor_projector = pytorch.nn.Linear(
                in_features=hidden_sizes[-1], out_features=out_features
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
        ## The input dropout probability. @I{# noqa: E266}
        self.dropout = pytorch.nn.Dropout(dropout)

    def forward(
        self, x: pytorch.Tensor
    ) -> Union[
        Tuple[List[pytorch.Tensor], pytorch.Tensor],
        Tuple[pytorch.Tensor, pytorch.Tensor],
    ]:
        """
        The forwards method of the nn.Module.
        @param x: pytorch.Tensor: The model input.
        @return  Union[
            Tuple[List[pytorch.Tensor], pytorch.Tensor],
            Tuple[pytorch.Tensor, pytorch.Tensor],
        ]: The tuple of actor and critic outputs.
        """
        if not self.share_network:
            action_outputs, state_value = self._run_non_shared_forward(x)
        else:
            action_outputs, state_value = self._run_shared_forward(x)
        if self._apply_actor_activation:
            if self.use_actor_projection:
                action_outputs = [
                    actor_activation(action_output)
                    for action_output, actor_activation in zip(
                        action_outputs, self.actor_activation
                    )
                ]
                action_outputs[1] = pytorch.diag_embed(action_outputs[1])
            else:
                action_outputs = self.actor_activation[0](action_outputs)
        if self._apply_critic_activation:
            state_value = self.value_activation(state_value)
        return action_outputs, state_value

    def _run_shared_forward(
        self, x: pytorch.Tensor
    ) -> Union[
        Tuple[List[pytorch.Tensor], pytorch.Tensor],
        Tuple[pytorch.Tensor, pytorch.Tensor],
    ]:
        """
        The forwards method of the nn.Module when actor and critic share network
        @param x: pytorch.Tensor: The model input.
        @return  Union[
            Tuple[List[pytorch.Tensor], pytorch.Tensor],
            Tuple[pytorch.Tensor, pytorch.Tensor],
        ]: The tuple of actor and critic outputs.
        """
        # Extract features from input.
        features = self.mlp_feature_extractor(x)
        features = self.activation_core(features)
        # Apply dropout on features.
        features = self.dropout(features)
        # Pass the features through the respective heads.
        action_outputs = self.actor_head(features)
        state_value = self.critic_head(features)
        # Flatten action and state outputs
        action_outputs = self.flatten(action_outputs)
        state_value = self.flatten(state_value)
        if self.use_actor_projection:
            # When actor projection is to be used, pass through actor_projector.
            action_projection = self.actor_projector(features)
            # Flatten action and action projections.
            action_projection = self.flatten(action_projection)
            action_outputs = [action_outputs, action_projection]
        return action_outputs, state_value

    def _run_non_shared_forward(
        self, x: pytorch.Tensor
    ) -> Union[
        Tuple[List[pytorch.Tensor], pytorch.Tensor],
        Tuple[pytorch.Tensor, pytorch.Tensor],
    ]:
        """
        The forwards method of the nn.Module when actor and critic do not share network
        @param x: pytorch.Tensor: The model input.
        @return  Union[
            Tuple[List[pytorch.Tensor], pytorch.Tensor],
            Tuple[pytorch.Tensor, pytorch.Tensor],
        ]: The tuple of actor and critic outputs.
        """
        # Extract features from input.
        action_features = self.actor_feature_extractor(x)
        state_value_features = self.critic_feature_extractor(x)
        # Apply dropout on features.
        action_features = self.dropout(action_features)
        state_value_features = self.dropout(state_value_features)
        # Pass the features through the respective heads.
        action_outputs = self.actor_head(action_features)
        state_value = self.critic_head(state_value_features)
        # Flatten action and state outputs.
        action_outputs = self.flatten(action_outputs)
        state_value = self.flatten(state_value)
        if self.use_actor_projection:
            # When actor projection is to be used, pass through actor_projector.
            action_projection = self.actor_projector(action_features)
            # Flatten action and action projections.
            action_projection = self.flatten(action_projection)
            action_outputs = [action_outputs, action_projection]
        return action_outputs, state_value

    def _set_shared_network_attributes(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        activation: List[Activation],
    ) -> None:
        """
        Sets appropriate attributes to create shared network.
        @param sequence_length: int: The sequence length of the expected tensor.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param activation: List[Activation]: List of activations to be used.
        """
        self.mlp_feature_extractor = _MlpFeatureExtractor(
            sequence_length=sequence_length,
            hidden_sizes=hidden_sizes,
            activation=activation[0],
        )

    def _set_non_shared_network_attributes(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        activation: List[Activation],
    ) -> None:
        """
        Sets appropriate attributes to create non-shared network.
        @param sequence_length: int: The sequence length of the expected tensor.
        @param hidden_sizes: List[int]: The list of hidden sizes for each layer.
        @param activation: List[Activation]: List of activations to be used.
        """
        self.actor_feature_extractor = _MlpFeatureExtractor(
            sequence_length=sequence_length,
            hidden_sizes=hidden_sizes,
            activation=activation[0],
        )
        self.critic_feature_extractor = _MlpFeatureExtractor(
            sequence_length=sequence_length,
            hidden_sizes=hidden_sizes,
            activation=activation[0],
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
            if not isinstance(action_space[1], (list, type(None))):
                raise TypeError(
                    f"The second element of `action_space must"
                    f"represent the shape of action to be sampled and must a list or None "
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
        activation: Union[Activation, List[Activation]], use_actor_projection: bool
    ) -> List[Union[Activation, List[Activation]]]:
        """
        Processes `activation` for use by the model.
        @param activation: Union[Activation, List[Activation]]: The activation function class(es) for the model.
            Must be an initialized activation object from PyTorch's nn (torch.nn) module. If a list is passed, List
            must be of length [1, 3], first activation for feature extractor, second for actor head and third for
            critic head.
        @param use_actor_projection: bool: Flag indicating whether to use projection for actor. Projection is applied
            to output of feature extractor of actor model.
        """
        if isinstance(activation, list):
            if not 0 < len(activation) <= 3 and not use_actor_projection:
                raise ValueError(
                    "Activation must be a list of either one, two or three activation; "
                    "first for feature extractor; second for actor head; third for critic head"
                )
            if not 0 < len(activation) <= 4 and use_actor_projection:
                raise ValueError(
                    "Activation must be a list of either one, two three or four activation; "
                    "first for feature extractor; second for actor head's first projection; "
                    "third for actor's second projection; fourth for critic head"
                )
            if len(activation) > 1:
                if use_actor_projection:
                    if len(activation) == 2:
                        activation[1] = [activation[1], pytorch.nn.Identity()]
                    elif len(activation) == 3:
                        activation[1] = [activation[1], activation[2]]
                        del activation[2]
                    elif len(activation) == 4:
                        activation[1] = [activation[1], activation[2]]
                else:
                    activation[1] = [activation[1]]
        elif isinstance(activation, pytorch.nn.Module):
            activation = [activation]
        else:
            raise TypeError(
                f"Expected argument `activation` to be of type {Activation} or {list} but received {type(activation)}"
            )
        return activation

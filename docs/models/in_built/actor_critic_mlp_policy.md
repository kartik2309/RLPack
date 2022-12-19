# Actor Critic Multi-Layered Perceptron Policy {#a2c_mlp}

RLPack implements the Actor Critic MLP Policy model for model-based learning. This model is implemented 
as `rlpack.models.actor_critic_mlp_policy.ActorCriticMlpPolicy`. In the implementation provided 
by RLPack, the model uses an MLP based feature extractor which is coupled with two Linear heads for action
logits and state value (*V(s)*). The model returns a tuple of PyTorch tensors with first value being action 
logits and second being the state value. 

<h4> Keyword: <kbd> model_name: "actor_critic_mlp_policy" </kbd> </h4>


With RLPack's implementation, it is possible to easily tweak the network architecture on the fly without any
intervention in code. `model_name: "actor_critic_mlp_policy"` is set, we have to pass to relevant `model_args` 
to the config dict.
An example of `model_args` is:
```python
{
  ...
  model_name: "actor_critic_mlp_policy",
  model_args = {
    "num_actions": 4,
    "sequence_length": 1,
    "hidden_sizes": [ 8, 64, 128, 256, 512 ],
    "dropout": 0.1
  }
  ... 
}
```
here
- `action_space` represents the action space for the agent. For more information you can refer 
[here](@ref agents/actor_critic/index.md)
- `sequence_length` is the length of sequence of input data. Depending on how states are reshaped, this can be 1 or
  above, but has to be at least 1. In cases where states are vectors, it can be reshaped manually before calling the
  agent's train function, or better yet, pass `new_shape` in the config dict to reshape the input under the hood. For
  example if states are vectors of shape `(4, )`, we can pass `"new_shape": [1, 4]` in the config dict. This reshapes
  the input accordingly. Consequently, we can see that sequence length is 1, hence we can set `sequence_length` to 1.
- `hidden_sizes`: The hidden features of each layer of the MLP. The first element of the list passed here must be
  corresponding to the number of features in states. Following the previous example provided in explanation of
  `sequence_length`, the first layer must have the size of 4, hence `"hidden_sizes": [4, ...]`. Depending on `new_shape`
  passed in config dict, this can be adjusted accordingly.
- `dropout`: The dropout probability to be applied on the last layer.

Since this model contains two heads (for actor and critic), you can choose to apply activation on either or both of 
them. This can be achieved by passing `activation` as a list. When using config with simulator, you can pass a list of 
strings (keywords) for activations. If you simply pass a string (keyword) for activation, activation is only applied 
to MLP based feature extractor (between hidden layers). To pass the activation args, you can pass the corresponding 
arguments as a list of keyword dictionary.If you pass a list, following will be followed:
- First element of list would be activation applied to MLP based feature extractor (between hidden layers) 
- Second element of list would be the activation applied to output of actor head. 
- Third element of list would be the activation applied to the output of critic head. 

Corresponding keyword arguments are used to initialize the activation objects.
 

You can pass a list of any size between one and three and activations will be applied as per above. For example if 
you pass a list as `["relu", "softplus"]`, only MLP based feature extractor and actor head get activations applied and 
output of critic head is passed as it is. 

Note that if you are directly using the class and not the simulator, make sure to read the documentation for
`rlpack.models.actor_critic_mlp_policy.ActorCriticMlpPolicy`

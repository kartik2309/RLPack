# Multi-Layered Perceptron

RLPack implements the MLP model for model-based learning. MLP model is implemented as `rlpack.models.mlp.Mlp`.

#### model_name: "mlp" 

With RLPack's implementation, it is possible to easily tweak the network architecture on the fly without any 
intervention in code. `model_name: "mlp"` is set, we have to pass to relevant `model_args` to the config dict. 
An example of `model_args` is: 
```python
model_args = {
  "num_actions": 4,
  "sequence_length": 1,
  "hidden_sizes": [ 8, 64, 128, 256, 512 ],
  "dropout": 0.1
}
```
here 
- `num_actions` is the number of actions of the agent (for discrete action space).
- `sequence_length` is the length of sequence of input data. Depending on how states are reshaped, this can be 1 or 
above, but has to be at least 1. In cases where states are vectors, it can be reshaped manually before calling the 
agent's train function, or better yet, pass `new_shape` in the config dict to reshape the input under the hood. For 
example if states are vectors of shape `[4, ]`, we can pass `"new_shape": [1, 4]` in the config dict. This reshapes 
the input accordingly. Consequently, we can see that sequence length is 1, hence we can set `sequence_length` to 1.
- `hidden_sizes`: The hidden features of each layer of the MLP. The first element of the list passed here must be 
corresponding to the number of features in states. Following the previous example provided in explanation of 
`sequence_length`, the first layer must have the size of 4, hence `"hidden_sizes": [4, ...]`. Depending on `new_shape` 
passed in config dict, this can be adjusted accordingly.
- `dropout`: The dropout probability to be applied on the last layer. 

The activation (keyed `activation_args`) is applied at the output of each layer except the final layer. 

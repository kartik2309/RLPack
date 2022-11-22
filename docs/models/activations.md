# Activations

Since RLPack is built on top of PyTorch, it by default uses PyTorch implementations for activations functions. These 
activation functions are applied to in-built networks. The application of activation varies network-to-network 
depending on the implementations. More details on a specific model can be found in [in-built-models](index.md).

To select an activation to be used for an in-built function, we can pass activation name by keywords in config 
dictionary. Keywords for each activation is given below. In config dictionary we can pass `"activation_name": <keyword>`. 
For example to select ReLU activation function, we can pass `"activation_name": "relu"` in the config dict. Any additional 
arguments for each activation's initialization can be passed via `activation_args` as dictionary of keyword arguments. 
Further details for each activation provided in RLPack is given below. 

## Implementations

Currently, RLPack implements three activations that can be used out of the box. 
- `ReLU`: The Rectified Linear-Unit Activation function. Since ReLU doesn't 
require additional arguments as per PyTorch's implementation, we can pass `"activation_args"=None`, or not pass it 
at all. **activation_name: "relu"**
- `LeakyReLU`: The Leaky Rectified Linear-Unit Activation function. If you wish to pass additional parameters for 
LeakyReLU as per PyTorch's arguments, you can pass `activation_args` with keyword arguments in the config dict. If 
not passed, default values are loaded as per 
[PyTorch's default](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html). **activation_name: "leaky_relu"**
- `TanH`: Hyperbolic Tangent Activation function. No arguments are required and hence we can pass 
`"activation_args"=None`, or not pass it at all. **activation_name: "tanh"**
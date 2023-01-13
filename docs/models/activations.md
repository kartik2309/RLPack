# Activations {#activations}

Since RLPack is built on top of PyTorch, it by default uses PyTorch implementations for activations functions. These 
activation functions are applied to in-built networks. The application of activation varies network-to-network 
depending on the implementations. More details on a specific model can be found in [in-built-models](@ref models/index.md).

To select an activation to be used for an in-built function, we can pass activation name by keywords in config 
dictionary. Keywords for each activation is given below. In config dictionary we can pass `"activation_name": <keyword>`. 
For example to select ReLU activation function, we can pass `"activation_name": "relu"` in the config dict. Any additional 
arguments for each activation's initialization can be passed via `activation_args` as dictionary of keyword arguments. 
Further details for each activation provided in RLPack is given below. 

Currently, the following activations have been implemented in RLPack, i.e. they can be used with keywords. 

| Activation   | Description                                                                                                                                                                                                               | Keyword        |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| `ReLU`       | The Rectified Linear-Unit Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html).                                             | `"relu"`       |
| `LeakyReLU`  | The Leaky Rectified Linear-Unit Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html).                                  | `"leaky_relu"` |
| `TanH`       | Hyperbolic Tangent Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.tanH.html)                                                     | `"tanh"`       |
| `SoftPlus`   | The Softplus activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)                                                       | `"softplus"`   |
| `Softmax`    | The Softmax activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)                                                         | `"softmax"`    |
| `Sigmoid`    | The Sigmoid activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)                                                         | `"sigmoid"`    |
| `ELU`        | The Exponential Linear Unit Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU)                                | `"elu"`        |
| `CELU`       | The Continuously Differentiable Exponential Linear Units Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html#torch.nn.CELU) | `"celu"`       |
| `GELU`       | The Gaussian Error Linear Units Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#torch.nn.GELU)                          | `"gelu"`       |
| `GLU`        | The Gated Linear Unit Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.GLU.html#torch.nn.GLU)                                      | `"glu"`        |
| `Hardtanh`   | The Hard-hyperbolic Tangent Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html#torch.nn.Hardtanh)                      | `"hardtanh"`   |
| `Mish`       | The Mish Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html#torch.nn.Mish)                                                 | `"mish"`       |
| `SiLU`       | The Sigmoid Linear Unit  Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU)                                 | `"silu"`       |
| `Softsign`   | The Softsign Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SoftSign)                                         | `"softsign"`   |
| `Softshrink` | The Softshrink Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html#torch.nn.Softshrink)                               | `"softshink"`  |
| `Tanhshrink` | The Tanhshrink Activation function. Optional arguments and more information can be looked up [here](https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html#torch.nn.Tanhshrink)                               | `"tanhshrink`  |
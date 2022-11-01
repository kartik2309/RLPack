# Optimizers

RLPack provides set of optimizers directly from PyTorch to be used to train the models in our agents. Just like 
everything in RLPack, optimizers are accessible via keywords too. Optimizer can be set via 
`optimizer_name: <keyword>` methodology as usual. For example `optimizer_name: "adam"` will initialize the Adam 
Optimizer from PyTorch. Further arguments is passed via the key `optimizer_args` which is a dictionary of keyword 
arguments for the optimizer (except model parameters). For keyword arguments of an optimizer, you can refer to 
PyTorch's official documentation corresponding that optimizer.

## Implementations

- `Adam`: The Adam Optimizer. Mandatory arguments can be looked up 
[here](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). To further understand Adam Optimization 
algorithm, you can refer [here](https://arxiv.org/abs/1412.6980). 
    #### optimizer_name: "adam"
- `AdamW`: The AdamW Optimizer. Mandatory arguments can be looked up 
[here](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html). To further understand Adam Optimization
algorithm, you can refer [here](https://arxiv.org/abs/1711.05101). 
    #### optimizer_name: "adamw"
- `RMSProp`: The Root Mean Squared Propagation optimizer. Mandatory arguments can be looked up 
[here](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html). For further details, 
[lecture notes](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) by G. Hinton can be referred. 
    #### optimizer_name: "rmsprop"
- `SGD`: The Stochastic Gradient Descend optimizer. The mandatory arguments for SGD Optimizer can be looked up 
[here](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html). For further understanding of SGD algorithm, 
refer [here](https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf).
    #### optimizer_name: "sgd"
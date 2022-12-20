# Loss Functions {#loss_functions}

Since RLPack is built on top of PyTorch, it uses loss functions provided by PyTorch. Loss functions is typically a 
mandatory argument. `loss_function_name` selects a loss function based on keyword from currently implemented loss 
functions when passed in config dictionary. For example `loss_function_name: "mse"` selects MSE loss function Further 
arguments can be passed to Loss Function initialization via `loss_function_args` key in config dictionary. Details 
pertaining arguments for each loss function can be referred to in PyTorch's official documentation.

Currently, the following loss functions have been implemented in RLPack, i.e. they can be used with keywords.

| Loss Function  | Description                                                                                                                                                                                                                                                                                          | Keyword            |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `HuberLoss`    | An improvement over MSE loss, making it less sensitive to outliers. Additional arguments can be passed via `loss_function_args`. For exact arguments and further details on this loss function, please refer [here](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html).              | `"huber_loss"`     |
| `MSE`          | Mean Squared Error is a simple loss function computing squared errors. Additional arguments can be passed via `loss_function_args`. For exact arguments and further details on this loss function, please refer [here](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html).             | `"mse"`            |
| `SmoothL1Loss` | The smooth L1 Loss, a variant of MSE, this is also less sensitive to outliers. Additional arguments can be passed via `loss_function_args`. For exact arguments and further details on this loss function, please refer [here](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html) | `"smooth_l1_loss"` |

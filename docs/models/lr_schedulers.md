# Learning Rate Schedulers {#lr_schedulers}

RLPack provides easy-to-use interface to integrate LR Schedulers in your training. All the LR Schedulers use 
PyTorch implementation under-the-hood. To use Learning Rate Schedulers easily, you can pass 
`lr_scheduler_name: <keyword>` in the config dictionary. For example, if we set `lr_scheduler_name: "step_lr"` 
we will select `StepLR` scheduler in our training. To pass additional arguments to the desired scheduler, we must 
pass `lr_scheduler_args` as a dictionary containing keyword arguments to the config dict. Further, you can limit the 
influence of LR Scheduler by setting `lr_threshold` in `agent_args`. Once this value is reached, the LR Scheduler is 
not called further. 

| LR Scheduler | Description                                                                                                                                                                         | Keyword       |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `StepLR`     | Step Learning Rate Scheduler. For mandatory arguments and further details, please refer [here](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html).     | `"step_lr"`   |
| `LinearLR`   | Linear Learning Rate Scheduler. For mandatory arguments and further details, please refer [here](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html). | `"linear_lr"` |
| `CyclicLR`   | Cyclic Learning Rate Scheduler. For mandatory arguments and further details, please refer [here](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html). | `"cyclic_lr"` |

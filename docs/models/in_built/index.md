# In-Built Models {#in_build_models}

RLPack provides in-built models that can be used on the fly with simply keywords from 
config. This help you set up experiments and train your agent on the fly. All in-built models 
inherit from Models class in `rlpack.utils.base.models.Model`. 

To include in-built models, you need to pass the key `model_name` in the config. An example 
to use 

Currently, the following models are in-built into RLPack: 

| In-Built model                                             | Keyword                     |
|------------------------------------------------------------|-----------------------------|
| [MLP](@ref models/in_built/mlp.md)                         | `"mlp"`                     |
| [A2C MLP](@ref models/in_built/actor_critic_mlp_policy.md) | `"actor_critic_mlp_policy"` |

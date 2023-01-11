"""!
@package rlpack.utils.base.registers
@brief This package implements the base classes for registers to be used across rlpack


Currently following base classes have been implemented:
    - `Register`: Register of information of all in-built models and agents implemented as
        rlpack.utils.base.registers.register.Register.
    - `InternalCodeRegister`: Register for information on codes to be used internally in RLPack; implemented as
        rlpack.utils.base.registers.internal_code_register.InternalCodeRegister
"""


from rlpack import pytorch, pytorch_distributions
from rlpack.actor_critic.a2c import A2C
from rlpack.actor_critic.a3c import A3C
from rlpack.actor_critic.ac import AC
from rlpack.distributions import (
    GaussianMixture,
    GaussianMixtureLogStd,
    MultivariateNormalLogStd,
    NormalLogStd,
)
from rlpack.dqn import Dqn
from rlpack.exploration import GaussianNoiseExploration, StateDependentExploration
from rlpack.models.actor_critic_mlp_policy import ActorCriticMlpPolicy
from rlpack.models.mlp import Mlp


class Register:
    """
    This abstract class contains all the necessary information about agents and models for setting them up.
    """

    def __init__(self):
        ## The tuple for mandatory keys (or keyword arguments) always expected. @I{# noqa: E266}
        self.mandatory_keys = (
            "mode",
            "env",
            "agent",
            "num_episodes",
            "agent_args",
        )
        ## The mandatory agent initialisation arguments @I{# noqa: E266}
        self.agent_init_args = ("agent", "agent_args")
        ## The mapping between given keyword and PyTorch optimizer class. @I{# noqa: E266}
        self.optimizer_map = {
            "adam": pytorch.optim.Adam,
            "adamw": pytorch.optim.AdamW,
            "rms_prop": pytorch.optim.RMSprop,
            "sgd": pytorch.optim.SGD,
        }
        ## The mapping between given keyword and PyTorch loss function class. @I{# noqa: E266}
        self.loss_function_map = {
            "huber_loss": pytorch.nn.HuberLoss,
            "mse": pytorch.nn.MSELoss,
            "smooth_l1_loss": pytorch.nn.SmoothL1Loss,
            "gaussian_nll_loss": pytorch.nn.GaussianNLLLoss,
        }
        ## The mapping between given keyword and PyTorch activation function class. @I{# noqa: E266}
        self.activation_map = {
            "celu": pytorch.nn.CELU,
            "elu": pytorch.nn.ELU,
            "gelu": pytorch.nn.GELU,
            "glu": pytorch.nn.GLU,
            "hardtanh": pytorch.nn.Hardtanh,
            "identity": pytorch.nn.Identity,
            "leaky_relu": pytorch.nn.LeakyReLU,
            "mish": pytorch.nn.Mish,
            "relu": pytorch.nn.ReLU,
            "selu": pytorch.nn.SELU,
            "sigmoid": pytorch.nn.Sigmoid,
            "silu": pytorch.nn.SiLU,
            "softplus": pytorch.nn.Softplus,
            "softmax": pytorch.nn.Softmax,
            "softsign": pytorch.nn.Softsign,
            "softshrink": pytorch.nn.Softshrink,
            "tanh": pytorch.nn.Tanh,
            "tanhshrink": pytorch.nn.Tanhshrink,
        }
        ## The mapping between given keyword and PyTorch LR Scheduler class. @I{# noqa: E266}
        self.lr_scheduler_map = {
            "step_lr": pytorch.optim.lr_scheduler.StepLR,
            "linear_lr": pytorch.optim.lr_scheduler.LinearLR,
            "cyclic_lr": pytorch.optim.lr_scheduler.CyclicLR,
        }
        ## The mapping between given keyword and Distribution class. @I{# noqa: E266}
        self.distributions_map = {
            "categorical": pytorch_distributions.Categorical,
            "bernoulli": pytorch_distributions.Bernoulli,
            "binomial": pytorch_distributions.Binomial,
            "normal": pytorch_distributions.Normal,
            "log_normal": pytorch_distributions.LogNormal,
            "multivariate_normal": pytorch_distributions.MultivariateNormal,
            "normal_log_std": NormalLogStd,
            "multivariate_normal_log_std": MultivariateNormalLogStd,
            "gaussian_mixture": GaussianMixture,
            "gaussian_mixture_log_std": GaussianMixtureLogStd,
        }
        ## The mapping between given keyword and Exploration class. @I{# noqa: E266}
        self.explorations_map = {
            "gaussian_noise": GaussianNoiseExploration,
            "state_dependent": StateDependentExploration,
        }
        ## The mapping between given keyword and [in-built](@ref models/index.md) models. @I{# noqa: E266}
        self.models = {"mlp": Mlp, "actor_critic_mlp_policy": ActorCriticMlpPolicy}
        ## The mapping between given keyword and [agents](@ref agents/index.md) models. @I{# noqa: E266}
        self.agents = {"dqn": Dqn, "ac": AC, "a2c": A2C, "a3c": A3C}
        ## The mapping between given keyword and [in-built](@ref models/index.md) model's default arguments. @I{# noqa: E266}
        self.model_args_default = {
            "mlp": ("sequence_length", "activation", "dropout"),
            "actor_critic_mlp_policy": (
                "sequence_length",
                "activation",
                "dropout",
                "share_network",
                "use_actor_projection",
                "exploration_tool",
                "use_diagonal_embedding_on_projection",
            ),
        }
        ## The mapping between given keyword and [in-built](@ref models/index.md) models' arguments. @I{# noqa: E266}
        self.model_args = {
            "mlp": (
                "hidden_sizes",
                "num_actions",
                *self.model_args_default["mlp"],
            ),
            "actor_critic_mlp_policy": (
                "hidden_sizes",
                "action_space",
                *self.model_args_default["actor_critic_mlp_policy"],
            ),
        }
        ## The mapping between given keyword and [agent](@ref agents/index.md) agent's default arguments. @I{# noqa: E266}
        self.agent_args_default = {
            "dqn": (
                "bootstrap_rounds",
                "device",
                "prioritization_params",
                "force_terminal_state_selection_prob",
                "tau",
                "normalization_tool",
                "apply_norm_to",
                "max_grad_norm",
                "grad_norm_p",
                "clip_grad_value",
            ),
            "ac": (
                "gae_lambda",
                "exploration_tool",
                "exploration_steps",
                "grad_accumulation_rounds",
                "training_frequency",
                "device",
                "dtype",
                "apply_norm",
                "normalization_tool",
                "apply_norm_to",
                "max_grad_norm",
                "grad_norm_p",
                "clip_grad_value",
                "timeout",
            ),
            "a2c": (
                "gae_lambda",
                "exploration_tool",
                "exploration_steps",
                "grad_accumulation_rounds",
                "training_frequency",
                "device",
                "dtype",
                "apply_norm",
                "normalization_tool",
                "apply_norm_to",
                "max_grad_norm",
                "grad_norm_p",
                "clip_grad_value",
                "timeout",
            ),
            "a3c": (
                "gae_lambda",
                "exploration_tool",
                "exploration_steps",
                "grad_accumulation_rounds",
                "training_frequency",
                "device",
                "dtype",
                "apply_norm",
                "normalization_tool",
                "apply_norm_to",
                "max_grad_norm",
                "grad_norm_p",
                "clip_grad_value",
                "timeout",
            ),
        }
        ## The mapping between given keyword and [agent](@ref agents/index.md) agents' arguments. @I{# noqa: E266}
        self.agent_args = {
            "dqn": (
                "target_model",
                "policy_model",
                "optimizer",
                "lr_scheduler",
                "loss_function",
                "gamma",
                "epsilon",
                "min_epsilon",
                "epsilon_decay_rate",
                "epsilon_decay_frequency",
                "memory_buffer_size",
                "target_model_update_rate",
                "policy_model_update_rate",
                "backup_frequency",
                "lr_threshold",
                "batch_size",
                "num_actions",
                "save_path",
                *self.agent_args_default["dqn"],
            ),
            "ac": (
                "policy_model",
                "optimizer",
                "lr_scheduler",
                "loss_function",
                "distribution",
                "gamma",
                "entropy_coefficient",
                "state_value_coefficient",
                "backup_frequency",
                "lr_threshold",
                "action_space",
                "save_path",
                *self.agent_args_default["ac"],
            ),
            "a2c": (
                "policy_model",
                "optimizer",
                "lr_scheduler",
                "loss_function",
                "distribution",
                "gamma",
                "entropy_coefficient",
                "state_value_coefficient",
                "backup_frequency",
                "lr_threshold",
                "action_space",
                "save_path",
                *self.agent_args_default["a2c"],
            ),
            "a3c": (
                "policy_model",
                "optimizer",
                "lr_scheduler",
                "loss_function",
                "distribution",
                "gamma",
                "entropy_coefficient",
                "state_value_coefficient",
                "backup_frequency",
                "lr_threshold",
                "action_space",
                "save_path",
                *self.agent_args_default["a3c"],
            ),
        }
        ## The mapping between keyword and agents' model arguments to wrap optimizer with. @I{# noqa: E266}
        self.model_args_to_optimize = {
            "dqn": {"target_model": False, "policy_model": True},
            "ac": {"policy_model": True},
            "a2c": {"policy_model": True},
            "a3c": {"policy_model": True},
        }
        ## The tuple of agent names, mandatory to be launched in distributed setting. @I{# noqa: E266}
        self.mandatory_distributed_agents = ("a2c", "a3c")
        ## The tuple of agent names, that require `distribution` parameter. @I{# noqa: E266}
        self.mandatory_distribution_required_agents = ("ac", "a2c", "a3c")

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
            "mlp": Mlp.__init__.__code__.co_varnames[-len(Mlp.__init__.__defaults__) :],
            "actor_critic_mlp_policy": ActorCriticMlpPolicy.__init__.__code__.co_varnames[
                -len(ActorCriticMlpPolicy.__init__.__defaults__) :
            ],
        }
        ## The mapping between given keyword and [in-built](@ref models/index.md) models' arguments. @I{# noqa: E266}
        self.model_args = {
            "mlp": Mlp.__init__.__code__.co_varnames[1:],
            "actor_critic_mlp_policy": ActorCriticMlpPolicy.__init__.__code__.co_varnames[
                1:
            ],
        }
        ## The mapping between given keyword and [agent](@ref agents/index.md) agent's default arguments. @I{# noqa: E266}
        self.agent_args_default = {
            "dqn": Dqn.__new__.__code__.co_varnames[-len(Dqn.__new__.__defaults__) :],
            "ac": AC.__init__.__code__.co_varnames[-len(AC.__init__.__defaults__) :],
            "a2c": A2C.__init__.__code__.co_varnames[-len(A2C.__init__.__defaults__) :],
            "a3c": A3C.__init__.__code__.co_varnames[-len(A3C.__init__.__defaults__) :],
        }
        ## The mapping between given keyword and [agent](@ref agents/index.md) agents' arguments. @I{# noqa: E266}
        self.agent_args = {
            "dqn": Dqn.__new__.__code__.co_varnames[1:],
            "ac": AC.__init__.__code__.co_varnames[1:],
            "a2c": A2C.__init__.__code__.co_varnames[1:],
            "a3c": A3C.__init__.__code__.co_varnames[1:],
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

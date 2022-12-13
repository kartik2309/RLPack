"""!
@package rlpack.utils.base
@brief This package implements the base classes to be used across rlpack


Currently following base classes have been implemented:
    - `Agent`: Base class for all agents, implemented as rlpack.utils.base.agent.Agent.
    - `Register`: Register of information of all in-built models and agents implemented as
        rlpack.utils.base.register.Register.
"""


from site import getsitepackages

from rlpack import pytorch
from rlpack.actor_critic.a2c import A2C
from rlpack.dqn import Dqn
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
            "env_name",
            "agent_name",
            "num_episodes",
            "max_timesteps",
            "reward_logging_frequency",
            "agent_args",
        )
        ## The model initialization arguments when using [in-built](@ref models/index.md) models. @I{# noqa: E266}
        self.model_init_args = ("model_name", "model_args")
        ## The mandatory agent initialisation arguments @I{# noqa: E266}
        self.agent_init_args = ("agent_name", "agent_args")
        ## The activation initialization arguments when using [in-built](@ref models/index.md) models. @I{# noqa: E266}
        self.activation_init_args = ("activation_name", "activation_args")
        ## The optimizer initialization arguments for given models. @I{# noqa: E266}
        self.optimizer_init_args = ("optimizer_name", "optimizer_args")
        ## The LR Scheduler initialization arguments. @I{# noqa: E266}
        self.lr_scheduler_init_args = ("lr_scheduler_name", "lr_scheduler_args")
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
        }
        ## The mapping between given keyword and PyTorch activation function class. @I{# noqa: E266}
        self.activation_map = {
            "relu": pytorch.nn.ReLU,
            "leaky_relu": pytorch.nn.LeakyReLU,
            "tanh": pytorch.nn.Tanh,
        }
        ## The mapping between given keyword and PyTorch LR Scheduler class. @I{# noqa: E266}
        self.lr_scheduler_map = {
            "step_lr": pytorch.optim.lr_scheduler.StepLR,
            "linear_lr": pytorch.optim.lr_scheduler.LinearLR,
            "cyclic_lr": pytorch.optim.lr_scheduler.CyclicLR,
        }

        ## The mapping between given keyword and [in-built](@ref models/index.md) models. @I{# noqa: E266}
        self.models = {"mlp": Mlp, "actor_critic_mlp_policy": ActorCriticMlpPolicy}
        ## The mapping between given keyword and [agents](@ref agents/index.md) models. @I{# noqa: E266}
        self.agents = {
            "dqn": Dqn,
            "a2c": A2C,
        }
        ## The mapping between given keyword and [in-built](@ref models/index.md) model's default arguments. @I{# noqa: E266}
        self.model_args_default = {
            "mlp": ("activation", "dropout"),
            "actor_critic_mlp_policy": ("activation", "dropout"),
        }
        ## The mapping between given keyword and [in-built](@ref models/index.md) models' arguments. @I{# noqa: E266}
        self.model_args = {
            "mlp": (
                "sequence_length",
                "hidden_sizes",
                "num_actions",
                *self.model_args_default["mlp"],
            ),
            "actor_critic_mlp_policy": (
                "sequence_length",
                "hidden_sizes",
                "num_actions",
                *self.model_args_default["actor_critic_mlp_policy"],
            ),
        }
        ## The mapping between given keyword and [agent](@ref agents/index.md) agent's default arguments. @I{# noqa: E266}
        self.agent_args_default = {
            "dqn": (
                "device",
                "prioritization_params",
                "force_terminal_state_selection_prob",
                "tau",
                "apply_norm",
                "apply_norm_to",
                "eps_for_norm",
                "p_for_norm",
                "dim_for_norm",
                "max_grad_norm",
                "grad_norm_p",
            ),
            "a2c": (
                "bootstrap_rounds",
                "device",
                "apply_norm",
                "apply_norm_to",
                "eps_for_norm",
                "p_for_norm",
                "dim_for_norm",
                "max_grad_norm",
                "grad_norm_p",
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
            "a2c": (
                "policy_model",
                "optimizer",
                "lr_scheduler",
                "loss_function",
                "gamma",
                "entropy_coefficient",
                "state_value_coefficient",
                "backup_frequency",
                "lr_threshold",
                "num_actions",
                "save_path",
                *self.agent_args_default["a2c"],
            ),
        }
        ## The mapping between keyword and agents' model arguments to wrap optimizer with. @I{# noqa: E266}
        self.model_args_to_optimize = {
            "dqn": {"target_model": False, "policy_model": True},
            "a2c": {"policy_model": True},
        }

    @staticmethod
    def get_prefix_path():
        """
        Gets prefix path for rlpack package, from python installation.
        @return: str: The prefix path to rlpack.
        """
        return f"{getsitepackages()[0]}/rlpack"

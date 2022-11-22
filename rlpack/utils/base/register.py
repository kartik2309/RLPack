from site import getsitepackages

from rlpack import pytorch
from rlpack.dqn.dqn_agent import DqnAgent
from rlpack.models.mlp import Mlp


class Register:
    def __init__(self):
        self.mandatory_keys = (
            "mode",
            "env_name",
            "agent_name",
            "num_episodes",
            "max_timesteps",
            "reward_logging_frequency",
            "agent_args",
        )
        self.model_init_args = ("model_name", "model_args")
        self.agent_init_args = ("agent_name", "agent_args")
        self.activation_init_args = ("activation_name", "activation_args")
        self.optimizer_init_args = ("optimizer_name", "optimizer_args")
        self.lr_scheduler_init_args = ("lr_scheduler_name", "lr_scheduler_args")
        self.optimizer_map = {
            "adam": pytorch.optim.Adam,
            "adamw": pytorch.optim.AdamW,
            "rms_prop": pytorch.optim.RMSprop,
            "sgd": pytorch.optim.SGD,
        }
        self.loss_function_map = {
            "huber_loss": pytorch.nn.HuberLoss,
            "mse": pytorch.nn.MSELoss,
        }
        self.activation_map = {
            "relu": pytorch.nn.ReLU,
            "leaky_relu": pytorch.nn.LeakyReLU,
            "tanh": pytorch.nn.Tanh,
        }
        self.lr_scheduler_map = {
            "step_lr": pytorch.optim.lr_scheduler.StepLR,
            "linear_lr": pytorch.optim.lr_scheduler.LinearLR,
            "cyclic_lr": pytorch.optim.lr_scheduler.CyclicLR,
        }
        self.norm_mode_codes = {"none": -1, "min_max": 0, "standardize": 1, "p_norm": 2}
        self.norm_to_mode_codes = {
            ("none",): -1,
            ("states",): 0,
            ("rewards",): 1,
            ("td",): 2,
            ("states", "rewards"): 3,
            ("states", "td"): 4,
        }
        self.models = {"mlp": Mlp}
        self.agents = {"dqn": DqnAgent}
        self.model_args = {
            "mlp": (
                "sequence_length",
                "hidden_sizes",
                "num_actions",
                "activation",
                "dropout",
            )
        }
        self.model_args_default = {"mlp": ("activation", "dropout")}
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
                "model_backup_frequency",
                "lr_threshold",
                "batch_size",
                "num_actions",
                "save_path",
                "device",
                "prioritization_params",
                "force_terminal_state_selection_prob",
                "tau",
                "apply_norm",
                "apply_norm_to",
                "eps_for_norm",
                "p_for_norm",
                "dim_for_norm",
            )
        }
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
            )
        }
        self.model_args_for_agents = {
            "dqn": {"target_model": False, "policy_model": True},
        }

        self.prioritization_strategy_codes = {
            "uniform": 0,
            "proportional": 1,
            "rank-based": 2,
        }
        self.agents_with_prioritized_memory = ("dqn",)

    @staticmethod
    def get_prefix_path():
        """
        Gets prefix path for rlpack package, from python installation.
        :return (str): The prefix path to rlpack.
        """
        return f"{getsitepackages()[0]}/rlpack"

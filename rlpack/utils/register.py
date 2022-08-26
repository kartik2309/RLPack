from typing import Any, Dict, List, TypeVar
from site import getsitepackages
import yaml

from rlpack import pytorch
from rlpack.dqn.dqn_agent import DqnAgent
from rlpack.dqn.models.dlqn1d import Dlqn1d
from rlpack.utils.base.agent import Agent

LRScheduler = TypeVar("LRScheduler")
LossFunction = TypeVar("LossFunction")
Activation = TypeVar("Activation")


class Register(object):
    def __init__(self):
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
        self.models = {"dlqn1d": Dlqn1d}
        self.agents = {"dlqn1d": DqnAgent}
        self.model_args = {
            "dlqn1d": (
                "sequence_length",
                "hidden_sizes",
                "num_actions",
                "activation",
                "dropout",
            )
        }
        self.agent_args = {
            "dlqn1d": (
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
                "min_lr",
                "batch_size",
                "num_actions",
                "save_path",
                "device",
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
            "dlqn1d": {"target_model": False, "policy_model": True},
        }
        self.default_configs = {"dlqn1d": f"{self.get_prefix_path()}/environments/configs/dlqn1d.yaml"}

    def get_model_args(self, model_name: str) -> List[str]:
        return self.model_args[model_name]

    def get_models(self, model_name: str, *args, **kwargs) -> List[pytorch.nn.Module]:
        return [
            self.models[model_name](*args, **kwargs)
            for _ in range(len(self.model_args_for_agents[model_name].keys()))
        ]

    def get_agent(self, model_name: str, *args, **kwargs) -> Agent:
        return self.agents[model_name](*args, **kwargs)

    def get_optimizer(
        self, params: List[pytorch.Tensor], optimizer_args: Dict[str, Any]
    ) -> pytorch.optim.Optimizer:
        name = optimizer_args["optimizer"]
        optimizer_args.pop("optimizer")
        optimizer = self.optimizer_map[name](params=params, **optimizer_args)
        return optimizer

    def get_activation(self, activation_args: Dict[str, Any]) -> Activation:
        activation_args_ = activation_args.copy()
        name = activation_args_["activation"]
        activation_args_.pop("activation")
        activation = self.activation_map[name](**activation_args_)
        return activation

    def get_lr_scheduler(
        self, optimizer: pytorch.optim.Optimizer, lr_scheduler_args: Dict[str, Any]
    ) -> LRScheduler:
        if lr_scheduler_args is None:
            return

        name = lr_scheduler_args["scheduler"]
        lr_scheduler_args.pop("scheduler")
        lr_scheduler = self.lr_scheduler_map[name](
            optimizer=optimizer, **lr_scheduler_args
        )
        return lr_scheduler

    def get_loss_function(self, loss_function_args) -> LossFunction:
        name = loss_function_args["loss_function"]
        loss_function_args.pop("loss_function")
        loss_function = self.loss_function_map[name](**loss_function_args)
        return loss_function

    def get_apply_norm_mode_code(self, apply_norm: str) -> int:
        if apply_norm not in self.norm_mode_codes.keys():
            raise ValueError("Invalid or unsupported value for `apply_norm` passed")
        return self.norm_mode_codes[apply_norm]

    def get_apply_norm_to_mode_code(self, apply_norm_to: str) -> int:
        apply_norm_to = tuple(apply_norm_to)
        if apply_norm_to not in self.norm_to_mode_codes.keys():
            raise ValueError("Invalid or unsupported value for `apply_norm_to` passed")
        return self.norm_to_mode_codes[apply_norm_to]

    def get_default_config(self, model_name: str) -> Dict[str, Any]:
        with open(self.default_configs[model_name], "rb") as f:
            config = yaml.load(f, yaml.Loader)

        return config

    @staticmethod
    def get_prefix_path():
        return f"{getsitepackages()[0]}/rlpack"

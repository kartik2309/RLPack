from site import getsitepackages
from typing import Any, Dict, List, TypeVar

import yaml

from rlpack import pytorch
from rlpack.dqn.dqn_agent import DqnAgent
from rlpack.models.mlp import Mlp
from rlpack.utils.base.agent import Agent

LRScheduler = TypeVar("LRScheduler")
LossFunction = TypeVar("LossFunction")
Activation = TypeVar("Activation")


class Register(object):
    """
    This class registers all the necessary objects that are required to run any configuration.
    """

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
        self.model_args_for_agents = {
            "dqn": {"target_model": False, "policy_model": True},
        }
        self.default_configs = {
            "dqn": f"{self.get_prefix_path()}/environments/configs/dlqn1d.yaml"
        }
        self.prioritization_strategy_codes = {
            "uniform": 0,
            "proportional": 1,
            "rank-based": 2,
        }
        self.agents_with_prioritized_memory = ("dqn",)

    def get_model_args(self, model_name: str) -> List[str]:
        """
        @:param model_name (str): The model name for which we want to obtain the args.
        @:return (List[str]): The list of model arguments.
        """
        return self.model_args[model_name]

    def get_models(
        self, model_name: str, agent_name: str, *args, **kwargs
    ) -> List[pytorch.nn.Module]:
        """
        This method automatically retrieves the given model(s) required by the agent.

        @:param model_name (str): The initialized model for the supplied model_name.
        @:param agent_name (str): The agent name for which models are requested.
        @:param args: Additional positional arguments for the model.
        @:param kwargs: Additional keyword arguments for the model.
        @:return List[pytorch.nn.Module]: The list of models required by the supplied agent.
        """
        return [
            self.models[model_name](*args, **kwargs)
            for _ in range(len(self.model_args_for_agents[agent_name].keys()))
        ]

    def get_agent(self, agent_name: str, *args, **kwargs) -> Agent:
        """
        This method retrieves the agent given the agent name.

        @:param agent_name (str): The agent to retrieve.
        @:param args: The additional positional arguments for the model.
        @:param kwargs: The additional keyword arguments required by the model.
        :return (Agent): The initialized agent.
        """
        if agent_name in self.agents_with_prioritized_memory:
            prioritization_params = kwargs.get("prioritization_params")
            if prioritization_params is not None:
                prioritization_params[
                    "prioritization_strategy"
                ] = self.get_prioritization_code(
                    prioritization_strategy=prioritization_params.get(
                        "prioritization_strategy", "uniform"
                    )
                )
                kwargs["prioritization_params"] = prioritization_params
        return self.agents[agent_name](*args, **kwargs)

    def get_optimizer(
        self, params: List[pytorch.Tensor], optimizer_args: Dict[str, Any]
    ) -> pytorch.optim.Optimizer:
        """
        This method retrieves the optimizer given by the "optimizer" key in the argument optimizer_args.

        @:param params List[pytorch.Tensor]: The model parameters to wrap the optimizer.
        @:param optimizer_args Dict[str, Any]: The optimizer arguments with mandatory "optimizer" key in the
            dictionary specifying the optimizer name.
        :return (pytorch.optim.Optimizer): The initialized optimizer.
        """
        name = optimizer_args["optimizer"]
        optimizer_args.pop("optimizer")
        optimizer = self.optimizer_map[name](params=params, **optimizer_args)
        return optimizer

    def get_activation(self, activation_args: Dict[str, Any]) -> Activation:
        """
        This method retrieves the activation to be supplied for the models.

        @:param activation_args: Dict[str, Any]: Activation arguments with mandatory "activation" key in the
            dictionary to specify the activation name.
        @:return (Activation): The initialized activated function
        """
        activation_args_ = activation_args.copy()
        name = activation_args_["activation"]
        activation_args_.pop("activation")
        activation = self.activation_map[name](**activation_args_)
        return activation

    def get_lr_scheduler(
        self, optimizer: pytorch.optim.Optimizer, lr_scheduler_args: Dict[str, Any]
    ) -> LRScheduler:
        """
        This method retrieves the lr_scheduler to be supplied for the models.
        @:param optimizer (pytorch.optim.Optimizer): The optimizer to wrap the lr scheduler around.
        @:param lr_scheduler_args: Dict[str, Any]: LR scheduler arguments with mandatory "lr_scheduler" key in the
            dictionary to specify the lr_scheduler name.
        @:return (Activation): The initialized lr_scheduler.
        """
        if lr_scheduler_args is None:
            return

        name = lr_scheduler_args["scheduler"]
        lr_scheduler_args.pop("scheduler")
        lr_scheduler = self.lr_scheduler_map[name](
            optimizer=optimizer, **lr_scheduler_args
        )
        return lr_scheduler

    def get_loss_function(self, loss_function_args: Dict[str, Any]) -> LossFunction:
        """
        This method retrieves the Loss Function to be supplied for the models.

        @:param loss_function_args (Dict[str, Any]): The loss function args with mandatory key "lr_scheduler"
            specifying the name of the scheduler to be used.
        @:return (LossFunction): The initialized loss function.
        """
        name = loss_function_args["loss_function"]
        loss_function_args.pop("loss_function")
        loss_function = self.loss_function_map[name](**loss_function_args)
        return loss_function

    def get_apply_norm_mode_code(self, apply_norm: str) -> int:
        """
        This method retrieves the apply_norm code from the given string. This code is to be supplied to agents.
        @:param apply_norm (str): The apply_norm string, specifying the normalization techniques to be used.
            *See the notes below to see the accepted values.
        :return (int): The code corresponding to the supplied valid apply_norm.

        * NOTE
        The value accepted for `apply_norm` are:
            - "none": No normalization
            - "min_max": Min-Max normalization
            - "standardize": Standardization.
            - "p_norm": P-Normalization
        """
        if apply_norm not in self.norm_mode_codes.keys():
            raise ValueError("Invalid or unsupported value for `apply_norm` passed")
        return self.norm_mode_codes[apply_norm]

    def get_apply_norm_to_mode_code(self, apply_norm_to: List[str]) -> int:
        """
        This method retrieves the apply_norm code_to from the given string. This code is to be supplied to agents.
        @:param apply_norm_to (List[str]): The apply_norm_to list, specifying the quantities on which we wish to
            apply normalization specified by `apply_norm`.
            *See the notes below to see the accepted values.
        @:return (int): The code corresponding to the supplied valid apply_norm_to.

        *NOTE
        The value accepted for `apply_norm_to` are:
            - ["none"]: Don't apply normalization to any quantity.
            - ["states"]: Apply normalization to states.
            - ["rewards"]: Apply normalization to rewards.
            - ["td"]: Apply normalization for TD values.
            - ["states", "rewards"]: Apply normalization to states and rewards.
            - ["states", "td"]: Apply normalization to states and TD values.
        """
        apply_norm_to = tuple(apply_norm_to)
        if apply_norm_to not in self.norm_to_mode_codes.keys():
            raise ValueError("Invalid or unsupported value for `apply_norm_to` passed")
        return self.norm_to_mode_codes[apply_norm_to]

    def get_prioritization_code(self, prioritization_strategy: str) -> int:
        """
        This method retrieves the prioritization code for corresponding strategy passed as string
            in prioritized parameters.
        @:param prioritization_strategy (str): A dictionary containing memory prioritization parameters for
            agents that may use it.
            *See the notes below to see the accepted values.
        :return: int: The prioritization code for corresponding string value.

        *NOTE:
        The accepted values for `prioritization_strategy` are as follows:
            - "uniform": No prioritization is done, i.e., uniform sampling takes place.
            - "proportional": Proportional prioritization takes place when sampling transition.
            - "rank-based": Rank based prioritization takes place when sampling transitions.
        """
        if prioritization_strategy not in self.prioritization_strategy_codes.keys():
            raise NotImplementedError(
                f"The provided prioritization strategy {prioritization_strategy} is not supported or is invalid!"
            )
        code = self.prioritization_strategy_codes[prioritization_strategy]
        return code

    def get_default_config(self, model_name: str) -> Dict[str, Any]:
        """
        Loads default config for the supplied model_name.

        @:param model_name (str): The model name to retrieve default config for.
        :return (Dict[str, Any]): Loaded configuration dictionary.
        """
        with open(self.default_configs[model_name], "rb") as f:
            config = yaml.load(f, yaml.Loader)

        return config

    @staticmethod
    def get_prefix_path():
        """
        Gets prefix path for rlpack package, from python installation.
        :return (str): The prefix path to rlpack.
        """
        return f"{getsitepackages()[0]}/rlpack"

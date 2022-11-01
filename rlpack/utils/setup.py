from typing import Any, Dict, List, Optional, TypeVar

from rlpack import pytorch
from rlpack.utils.base.agent import Agent
from rlpack.utils.base.register import Register

LRScheduler = TypeVar("LRScheduler")
LossFunction = TypeVar("LossFunction")
Activation = TypeVar("Activation")


class Setup(Register):
    """
    This class registers all the necessary objects that are required to run any configuration.
    """

    def __init__(self):
        super(Setup, self).__init__()

    def get_model_args(self, model_name: str) -> List[str]:
        """
        :param model_name: str: The model name for which we want to obtain the args.
        :return: List[str]: The list of model arguments.
        """
        return self.model_args[model_name]

    def get_models(
        self, model_name: str, agent_name: str, *args, **kwargs
    ) -> List[pytorch.nn.Module]:
        """
        This method automatically retrieves the given model(s) required by the agent.
        :param model_name: str: The initialized model for the supplied model_name.
        :param agent_name: str: The agent name for which models are requested.
        :param args: Additional positional arguments for the model.
        :param kwargs: Additional keyword arguments for the model.
        :return: List[pytorch.nn.Module]: The list of models required by the supplied agent.
        """
        return [
            self.models[model_name](*args, **kwargs)
            for _ in range(len(self.model_args_for_agents[agent_name].keys()))
        ]

    def get_agent(self, agent_name: str, *args, **kwargs) -> Agent:
        """
        This method retrieves the agent given the agent name.
        :param agent_name: str: The agent to retrieve.
        :param args: The additional positional arguments for the model.
        :param kwargs: The additional keyword arguments required by the model.
        :return: Agent: The initialized agent.
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
        self,
        params: List[pytorch.Tensor],
        optimizer_name: str,
        optimizer_args: Dict[str, Any],
    ) -> pytorch.optim.Optimizer:
        """
        This method retrieves the optimizer given by the "optimizer" key in the argument optimizer_args.
        :param params: List[pytorch.Tensor]: The model parameters to wrap the optimizer.
        :param optimizer_name: str: The optimizer name to be used.
        :param optimizer_args: Dict[str, Any]: A dictionary with keyword arguments for to-be initialized
            optimizer.
        :return: pytorch.optim.Optimizer: The initialized optimizer.
        """
        optimizer = self.optimizer_map[optimizer_name](params=params, **optimizer_args)
        return optimizer

    def get_activation(
        self, activation_name: str, activation_args: Dict[str, Any]
    ) -> Activation:
        """
        This method retrieves the activation to be supplied for the models.
        @:param activation_name: str: The activation name to be used.
        @:param activation_args: Dict[str, Any]: A dictionary with keyword arguments for to-be initialized
            activation function.
        @:return: Activation: The initialized activated function
        """
        activation = self.activation_map[activation_name](**activation_args)
        return activation

    def get_lr_scheduler(
        self,
        optimizer: pytorch.optim.Optimizer,
        lr_scheduler_name: Optional[str],
        lr_scheduler_args: Optional[Dict[str, Any]],
    ) -> LRScheduler:
        """
        This method retrieves the lr_scheduler to be supplied for the models.
        :param optimizer: pytorch.optim.Optimizer: The optimizer to wrap the lr scheduler around.
        :param lr_scheduler_name: str: The LR Scheduler's name to be used.
        :param lr_scheduler_args: Dict[str, Any]: A dictionary with keyword arguments for to-be initialized
            LR Scheduler.
        :return: Activation: The initialized lr_scheduler.
        """
        if lr_scheduler_name is None or lr_scheduler_args is None:
            return
        lr_scheduler = self.lr_scheduler_map[lr_scheduler_name](
            optimizer=optimizer, **lr_scheduler_args
        )
        return lr_scheduler

    def get_loss_function(
        self, loss_function_name: str, loss_function_args: Dict[str, Any]
    ) -> LossFunction:
        """
        This method retrieves the Loss Function to be supplied for the models.
        :param loss_function_name: str: The loss function's name to be used.
        :param loss_function_args: Dict[str, Any]: A dictionary with keyword arguments for to-be initialized
            loss function.
        :return (LossFunction): The initialized loss function.
        """
        loss_function = self.loss_function_map[loss_function_name](**loss_function_args)
        return loss_function

    def get_apply_norm_mode_code(self, apply_norm: str) -> int:
        """
        This method retrieves the apply_norm code from the given string. This code is to be supplied to agents.
        :param apply_norm: str: The apply_norm string, specifying the normalization techniques to be used.
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
        :param apply_norm_to: List[str]: The apply_norm_to list, specifying the quantities on which we wish to
            apply normalization specified by `apply_norm`.
            *See the notes below to see the accepted values.
        :return: int: The code corresponding to the supplied valid apply_norm_to.

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
        :param prioritization_strategy: str: A dictionary containing memory prioritization parameters for
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

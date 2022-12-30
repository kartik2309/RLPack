"""!
@package rlpack.utils
@brief This package implements the basic utilities to be used across rlpack.


Currently following classes have been implemented:
    - `Normalization`: Normalization tool implemented as rlpack.utils.normalization.Normalization with
        support for regular normalization methods.
    - `SanityCheck`: Sanity check for arguments when using Simulator from rlpack.simulator.Simulator. Class is
        implemented as rlpack.utils.sanity_check.SanityCheck.
    - `Setup`: Sets up the simulator to run the agent with environment. Implemented as rlpack.utils.setup.Setup.
    - `InternalCodeSetup`: For internal use to check/validate arguments and to retrieve codes for internal use.
        Implemented as rlpack.utils.internal_code_setup.InternalCodeSetup.

Following packages are part of utils:
    - `base`: A package for base class, implemented as rlpack.utils.base

Following exceptions have been defined:
    - `AgentError`: For error happening in Agent's initialization. Implemented as rlpack.utils.exceptions.AgentError.

Following typing hints have been defined:
    - `LRScheduler`: The Typing variable for LR Schedulers.
    - `LossFunction`: Typing hint for loss functions for RLPack. Implemented as
        rlpack.utils.typing_hints.LossFunction.
    - `Activation`: Typing hint for activation functions for RLPack. Implemented as
        rlpack.utils.typing_hints.Activation.
    - `RunFuncSignature`: Typing hint for function signatures to be launched in
        rlpack.simulator_distributed.SimulatedDistributed in distributed mode. Implemented as
        rlpack.utils.typing_hints.RunFuncSignature.
    - `GenericFuncSignature`: Typing hint for generic void function signatures. Implemented as
        rlpack.utils.typing_hints.GenericFuncSignature.
"""


import re
from typing import Any, Dict, List, Optional, Union

from rlpack import pytorch, pytorch_distributions
from rlpack.utils import Activation, LossFunction, LRScheduler
from rlpack.utils.base.agent import Agent
from rlpack.utils.base.registers.internal_code_register import InternalCodeRegister
from rlpack.utils.base.registers.register import Register


class Setup(Register, InternalCodeRegister):
    """
    This class sets up all the necessary objects that are required to run any configuration.
    """

    def __init__(self):
        super(Setup, self).__init__()

    def get_model_args(self, model_name: str) -> List[str]:
        """
        @param model_name: str: The model name for which we want to obtain the args.
        @return List[str]: The list of model arguments.
        """
        return self.model_args[model_name]

    def get_models(
        self, model_name: str, agent_name: str, *args, **kwargs
    ) -> List[pytorch.nn.Module]:
        """
        This method automatically retrieves the given model(s) required by the agent.
        @param model_name: str: The initialized model for the supplied model_name.
        @param agent_name: str: The agent name for which models are requested.
        @param args: Additional positional arguments for the model.
        @param kwargs: Additional keyword arguments for the model.
        @return List[pytorch.nn.Module]: The list of models required by the supplied agent.
        """
        return [
            self.models[model_name](*args, **kwargs)
            for _ in range(len(self.model_args_to_optimize[agent_name].keys()))
        ]

    def get_agent_model_args(self, agent_name: str):
        agent_model_args = [
            agent_arg
            for agent_arg in self.agent_args[agent_name]
            if re.match(r".*model$", agent_arg) is not None
        ]
        return agent_model_args

    def get_agent(self, agent_name: str, *args, **kwargs) -> Agent:
        """
        This method retrieves the agent given the agent name.
        @param agent_name: str: The agent to retrieve.
        @param args: The additional positional arguments for the model.
        @param kwargs: The additional keyword arguments required by the model.
        @return Agent: The initialized agent.
        """
        return self.agents[agent_name](*args, **kwargs)

    def get_optimizer(
        self,
        params: List[pytorch.Tensor],
        optimizer_name: str,
        optimizer_args: Dict[str, Any],
    ) -> pytorch.optim.Optimizer:
        """
        This method retrieves the optimizer given by the "optimizer" key in the argument optimizer_args.
        @param params: List[pytorch.Tensor]: The model parameters to wrap the optimizer.
        @param optimizer_name: str: The optimizer name to be used.
        @param optimizer_args: Dict[str, Any]: A dictionary with keyword arguments for to-be initialized
            optimizer.
        @return pytorch.optim.Optimizer: The initialized optimizer.
        """
        optimizer = self.optimizer_map[optimizer_name](params=params, **optimizer_args)
        return optimizer

    def get_activation(
        self,
        activation_name: Union[str, List[str]],
        activation_args: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Union[Activation, List[Activation]]:
        """
        This method retrieves the activation to be supplied for the models. If list is passed, list of initialized
        activation objects are retrieved.
        @param activation_name: Union[str, List[str]]: The activation name to be used.
        @param activation_args: Union[Dict[str, Any], List[Dict[str, Any]], None]: A dictionary with keyword arguments
            for to-be initialized activation function.
        @return Union[Activation, List[Activation]]: The initialized activated function(s).
        """
        if isinstance(activation_name, str):
            if activation_args is None:
                activation_args = dict()
            if not isinstance(activation_args, dict):
                raise TypeError(
                    f"Expected `activation_args` to be of type {dict} or {type(None)}, "
                    f"but got {type(activation_args)}"
                )
            activation = self.activation_map[activation_name](**activation_args)
        elif isinstance(activation_name, (list, tuple)):
            if activation_args is None:
                activation_args = [dict()] * len(activation_name)
            if not isinstance(activation_args, (list, tuple)):
                raise TypeError(
                    f"Expected `activation_args` to be of type {list}, {tuple} or {type(None)}, "
                    f"but got {type(activation_args)}"
                )
            activation = [
                self.activation_map[activation_name_](**activation_args_)
                for activation_name_, activation_args_ in zip(
                    activation_name, activation_args
                )
            ]
        else:
            raise TypeError(
                f"Expected `activation_name` to be of type {list}, {tuple} or {Activation}, "
                f"but got {type(activation_args)}"
            )
        return activation

    def get_lr_scheduler(
        self,
        optimizer: pytorch.optim.Optimizer,
        lr_scheduler_name: Optional[str] = None,
        lr_scheduler_args: Optional[Dict[str, Any]] = None,
    ) -> Union[LRScheduler, None]:
        """
        This method retrieves the lr_scheduler to be supplied for the models if LR Scheduler is requested.
        @param optimizer: pytorch.optim.Optimizer: The optimizer to wrap the lr scheduler around.
        @param lr_scheduler_name: str: The LR Scheduler's name to be used.
        @param lr_scheduler_args: Dict[str, Any]: A dictionary with keyword arguments for to-be initialized
            LR Scheduler.
        @return Union[LRScheduler, None]: The initialized lr_scheduler if passed else None.
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
        @param loss_function_name: str: The loss function's name to be used.
        @param loss_function_args: Dict[str, Any]: A dictionary with keyword arguments for to-be initialized
            loss function.
        :return (LossFunction): The initialized loss function.
        """
        loss_function = self.loss_function_map[loss_function_name](**loss_function_args)
        return loss_function

    def get_distribution_class(
        self, distribution: str
    ) -> pytorch_distributions.Distribution:
        return self.distributions_map[distribution]

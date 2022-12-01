"""!
@package utils
@brief This package implements the basic utilities to be used across rlpack.


Currently following classes have been implemented:
    - Normalization: Normalization tool implemented as rlpack.utils.normalization.Normalization with
        support for regular normalization methods.
    - SanityCheck: Sanity check for arguments when using Simulator from rlpack.simulator.Simulator. Class is
        implemented as rlpack.utils.sanity_check.SanityCheck.
    - Setup: Sets up the simulator to run the agent with environment. Implemented as rlpack.utils.setup.Setup.

Following TypeVars have been defined:
    - LRScheduler: The Typing variable for LR Schedulers.
    - LossFunction: The Typing variable for Loss Functions.
    - Activation: The Typing variable for Activations.
"""


import logging
import re
from typing import Any, Dict, List

from rlpack.utils.base.register import Register


class SanityCheck(Register):
    """
    This class does the basic sanity check of input_config.
    """

    def __init__(self, input_config: Dict[str, Any]):
        """
        :param input_config: (Dict[str, Any]): The input config that is to be used for training.
        """
        super(SanityCheck, self).__init__()
        self.args = list(input_config.keys())
        self.input_config = input_config

    def check_mandatory_params_sanity(self) -> None:
        """
        Checks the sanity of mandatory parameters for RLPack Simulator to run.
        """
        present_mandatory_args = [k in self.args for k in self.mandatory_keys]
        if not all(present_mandatory_args):
            raise ValueError(
                self.__error_message("mandatory_argument", present_mandatory_args)
            )

    def check_model_init_sanity(self) -> bool:
        """
        Checks the sanity of model arguments, either custom or in-built. When in-built is to be used, both
        `model_name` and `model_args` must be passed. If both are not passed, it will try to check if there is a
        custom model that is passed. Custom models' names must correspond to their target agent's keyword argument
        name.
        :return: A flag indicating if we use a custom model or an in-built model.
        """
        present_model_init_args = [k in self.args for k in self.model_init_args]
        custom_model = False
        # Checks if we are passing custom model or not.
        if not all(present_model_init_args):
            custom_model = True
            logging.debug("Incomplete `model_init_args``")
            logging.info("Trying to find custom model if being passed ...")
        # If not a custom model, check with `model_args` to see if all arguments are received for in-built model.
        # If a custom model, check with `agent_args` to see if all the model related arguments are received for agent.
        if not custom_model:
            model_name = self.input_config.get("model_name")
            if not isinstance(model_name, str):
                raise TypeError(
                    f"Expected `model_name` to be of type {str} but received type {type(model_name)}"
                )
            if model_name not in self.models.keys():
                raise NotImplementedError(
                    f"The requested model in-built model "
                    f"{model_name} is not supported."
                )
            model_args = [
                agent_arg
                for agent_arg in self.model_args[model_name]
                if agent_arg not in self.model_args_default[model_name]
            ]
            model_args_from_input_config = list(
                self.input_config.get("model_args", list())
            )
            present_model_init_args = [
                k in model_args_from_input_config for k in model_args
            ]
            if not all(present_model_init_args):
                raise ValueError(
                    f"Cannot initialize requested model; "
                    f"{self.__error_message('model_args', present_model_init_args)}"
                )
            self.check_activation_init_sanity()
        else:
            self.check_agent_init_sanity(only_model_arg_check=True)
        return custom_model

    def check_agent_init_sanity(self, only_model_arg_check: bool = False) -> None:
        """
        Check the sanity of agent input. This function will check the arguments received for agents to verify if
        all necessary arguments are received.
        :param only_model_arg_check: bool: Indicating weather to check only the model related arguments or all of the
            mandatory arguments.
        """
        present_agent_init_args = [k in self.args for k in self.agent_init_args]
        if not all(present_agent_init_args):
            raise ValueError(
                f"Cannot Initialize requested Agent; "
                f"{self.__error_message('agent_init_args', present_agent_init_args)}"
            )
        agent_name = self.input_config.get("agent_name")
        if not isinstance(agent_name, str):
            raise TypeError(
                f"Expected `agent_name` to be of type {str} but received type {type(agent_name)}"
            )
        if agent_name not in self.agents.keys():
            raise NotImplementedError(
                f"The requested agent {agent_name} is not supported."
            )
        if only_model_arg_check:
            agent_args_from_input_config = list(
                self.input_config.get("agent_args", list())
            )
            agent_args = [
                agent_arg
                for agent_arg in self.agent_args[agent_name]
                if agent_arg not in self.agent_args_default[agent_name]
                and re.match(r".*model$", agent_arg) is not None
            ]
            present_agent_init_args = [
                k in agent_args_from_input_config for k in agent_args
            ]
            if not all(present_agent_init_args):
                raise ValueError(
                    f"Cannot initialize agent with custom model for given Agent; "
                    f"{self.__error_message('agent_args', present_agent_init_args)}"
                )

    def check_activation_init_sanity(self) -> None:
        """
        Checks the basic sanity of activation related arguments in config. Note that this will not check sanity of
        the given activation even if Activation is valid.
        If invalid arguments are passed, error will be raised by PyTorch.
        """
        # If only activation name is passed but not the activation_args, will default to an empty dictionary.
        if (
            self.activation_init_args[0] in self.args
            and self.activation_init_args[1] not in self.args
        ):
            self.args.append(self.activation_init_args[1])
        present_activation_init_args = [
            k in self.args for k in self.activation_init_args
        ]
        if (
            not all(present_activation_init_args)
            and not present_activation_init_args[1]
        ):
            raise ValueError(
                f"Cannot Initialize requested Activation for the given Agent; "
                f"{self.__error_message('activation_init_args', present_activation_init_args)}"
            )
        activation_name = self.input_config["activation_name"]
        if not isinstance(activation_name, str):
            raise TypeError(
                f"Expected `activation_name` to be of type {str} but received type {type(activation_name)}"
            )
        if activation_name not in self.activation_map.keys():
            raise NotImplementedError(
                f"The requested activation {activation_name} is not supported."
            )

    def check_optimizer_init_sanity(self) -> None:
        """
        Checks the basic sanity of optimizer related arguments in config. Note that this will not check sanity of
        the given optimizer args even if optimizer is valid.
        If invalid arguments are passed, error will be raised by PyTorch.
        """
        present_optimizer_init_args = [k in self.args for k in self.optimizer_init_args]
        if not all(present_optimizer_init_args):
            raise ValueError(
                f"Cannot Initialize requested Optimizer for the given Agent; "
                f"{self.__error_message('optimizer_init_args', present_optimizer_init_args)}"
            )
        optimizer_name = self.input_config["optimizer_name"]
        if not isinstance(optimizer_name, str):
            raise TypeError(
                f"Expected `optimizer_name` to be of type {str} but received type {type(optimizer_name)}"
            )
        if optimizer_name not in self.optimizer_map.keys():
            raise NotImplementedError(
                f"The requested optimizer {optimizer_name} is not supported."
            )

    def check_lr_scheduler_init_sanity(self) -> None:
        """
        Checks the basic sanity of lr_scheduler related arguments in config. Note that this will not check sanity of
        the given lr_scheduler's args even if LR Scheduler valid.
        If invalid arguments are passed, error will be raised by PyTorch.
        """
        # If not LR Scheduler is requested, no sanity check is to be done.
        if (
            self.lr_scheduler_init_args[0] not in self.args
            and self.lr_scheduler_init_args[1] not in self.args
        ):
            return
        present_lr_scheduler_init_args = [
            k in self.args for k in self.lr_scheduler_init_args
        ]
        if (
            not all(present_lr_scheduler_init_args)
            and not present_lr_scheduler_init_args[1]
        ):
            raise ValueError(
                f"Cannot Initialize requested LR Scheduler for the given Agent; "
                f"{self.__error_message('lr_scheduler_init', present_lr_scheduler_init_args)}"
            )
        if all(present_lr_scheduler_init_args):
            lr_scheduler_name = self.input_config["lr_scheduler_name"]
            if not isinstance(lr_scheduler_name, str):
                raise TypeError(
                    f"Expected `lr_scheduler_name` to be of type {str} but received type {type(lr_scheduler_name)}"
                )
            if lr_scheduler_name not in self.lr_scheduler_map.keys():
                raise NotImplementedError(
                    f"The requested lr_scheduler {lr_scheduler_name} is not supported."
                )

    def __error_message(self, param_of_arg: str, boolean_args: List[bool]) -> str:
        """
        Private method to craft the error message indicating parameters that are missing.
        :param param_of_arg: str: The parameter for which argument is missing.
        :param boolean_args: Boolean list of indicating if the corresponding argument was received or not
        :return: str: The error message.
        """
        message = (
            f"The following `{param_of_arg}` arguments were not received: "
            f" {[self.args[idx] for idx, arg in enumerate(boolean_args) if arg is False]}"
        )
        return message

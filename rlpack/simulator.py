"""!
@package rlpack
@brief Implementation of RL Algorithms built on top of PyTorch. Heavy workloads have
been optimized with C++ backend.
"""


import os
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import yaml

from rlpack import SummaryWriter, pytorch
from rlpack.trainer.trainer import Trainer
from rlpack.utils.base.agent import Agent
from rlpack.utils.sanity_check import SanityCheck
from rlpack.utils.setup import Setup


class Simulator:
    """
    Simulator class simulates the environments and runs the agent through the environment. It also sets up
    the models and agents for training and/or evaluation.
    """

    def __init__(self, config: Dict[str, Any], is_child_process: bool = False):
        """
        @param config: Dict[str, Any]: The configuration dictionary for setup.
        @param is_child_process: bool: Indicates if Simulator is being run as a child process. Default: False
        """
        ## The object of rlpack.utils.setup.Setup to set-up models. @I{# noqa: E266}
        self.setup = Setup()
        # Perform sanity check before starting.
        ## The object of rlpack.utils.sanity_check.SanityCheck to perform sanity checks on arguments. @I{# noqa: E266}
        self.sanity_check = SanityCheck(config)
        # Check sanity of agent depending on if Simulator is being launched as a child process.
        if is_child_process:
            self.sanity_check.check_if_valid_agent_for_simulator_distributed()
        else:
            self.sanity_check.check_if_valid_agent_for_simulator()
        ## The input config. @I{# noqa: E266}
        self.config = config
        # Check if mandatory arguments are received from config.
        self.sanity_check.check_mandatory_params_sanity()
        # Get save path from environment variable or config.
        save_path = (
            self.config["agent_args"].get("save_path")
            if self.config["agent_args"].get("save_path") is not None
            else os.getenv("SAVE_PATH")
        )
        # Check validity of passed save_path
        if save_path is None:
            raise ValueError(
                "The argument `save_path` was not set. "
                "Either pass it in config dictionary or set the environment variable `SAVE_PATH`"
            )
        # Set save path in config.
        self.config["agent_args"]["save_path"] = save_path
        ## The agent object requested via config. @I{# noqa: E266}
        self.agent = self.setup_agent()
        env = self.setup_environment()
        summary_writer = self.setup_summary_writer(save_path)
        ## The environment object requested via config or passed via config. @I{# noqa: E266}
        self.trainer = Trainer(
            mode=self.config["mode"],
            agent=self.agent,
            env=env,
            save_path=save_path,
            num_episodes=self.config["num_episodes"],
            max_timesteps=self.config.get("max_timesteps"),
            custom_suffix=self.config.get("custom_suffix"),
            reshape_func=self.config.get("reshape_func"),
            new_shape=self.config.get("new_shape"),
            summary_writer=summary_writer,
            is_distributed=is_child_process,
        )
        ## The flag indicating if the model is custom or not, i.e. does not use [in-built](@ref models/index.md) models. @I{# noqa: E266}
        self.is_custom_model = False
        self.agent_model_args = list()
        return

    def setup_agent(self) -> Agent:
        """
        This method sets up agent by loading all the necessary arguments.
        @return Agent: The loaded and initialized agent.
        """
        models = self.setup_models()
        self.sanity_check.check_agent_init_sanity()
        self.sanity_check.check_optimizer_init_sanity()
        self.sanity_check.check_lr_scheduler_init_sanity()
        self.sanity_check.check_distribution_sanity()
        self.sanity_check.check_exploration_tool_sanity()
        agent_args_for_models = [
            arg
            for arg in self.setup.agent_args[self.config["agent"]]
            if arg in self.setup.model_args_to_optimize[self.config["agent"]].keys()
        ]
        agent_model_kwargs = {
            arg: models[idx] for idx, arg in enumerate(agent_args_for_models)
        }
        trainable_models = [
            model
            for model_arg_name, model in agent_model_kwargs.items()
            if self.setup.model_args_to_optimize[self.config["agent"]][model_arg_name]
        ]
        optimizers = [
            self.setup.get_optimizer(
                params=model.parameters(),
                optimizer_name=self.config["agent_args"]["optimizer"],
                optimizer_args=self.config["optimizer_args"],
            )
            for model in trainable_models
        ]
        lr_schedulers = [
            self.setup.get_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler_name=self.config["agent_args"].get("lr_scheduler"),
                lr_scheduler_args=self.config.get("lr_scheduler_args"),
            )
            for optimizer in optimizers
        ]
        default_model_args = {
            arg: self.config["agent_args"].get(arg)
            for arg in self.setup.agent_args_default[self.config["agent"]]
            if self.config["agent_args"].get(arg) is not None
        }
        processed_agent_args = dict(
            **agent_model_kwargs,
            optimizer=optimizers[0] if len(optimizers) == 1 else optimizers,
            lr_scheduler=lr_schedulers[0] if len(lr_schedulers) == 1 else lr_schedulers,
            loss_function=self.setup.get_loss_function(
                loss_function_name=self.config["agent_args"]["loss_function"],
                loss_function_args=self.config["loss_function_args"],
            ),
            save_path=self.config["agent_args"]["save_path"],
        )
        # Add distribution object to arguments if required by agent.
        if "distribution" in self.config["agent_args"]:
            distribution_kwargs = dict(
                distribution=self.setup.get_distribution_class(
                    self.config["agent_args"]["distribution"]
                )
            )
            processed_agent_args.update(distribution_kwargs)
        if "exploration_tool" in self.config["agent_args"]:
            exploration_tool_kwargs = dict(
                exploration_tool=self.setup.get_exploration_tool(
                    self.config["agent_args"]["exploration_tool"],
                    self.config.get("exploration_tool_args", dict()),
                )
            )
            processed_agent_args.update(exploration_tool_kwargs)
        if "normalization_tool" in self.config["agent_args"]:
            normalization_tool = dict(
                normalization_tool=self.setup.get_normalization_tool(
                    self.config["agent_args"]["normalization_tool"],
                    self.config.get("normalization_tool_args", dict()),
                )
            )
            processed_agent_args.update(normalization_tool)
        agent_args_from_config = {
            k: v
            for k, v in self.config["agent_args"].items()
            if k not in processed_agent_args.keys()
            and k in self.setup.agent_args[self.config["agent"]]
        }
        agent_kwargs = {
            **default_model_args,
            **processed_agent_args,
            **agent_args_from_config,
        }
        agent = self.setup.get_agent(agent_name=self.config["agent"], **agent_kwargs)
        config = self.prepare_config_to_save(self.config.copy())
        with open(
            os.path.join(self.config["agent_args"]["save_path"], "config.yaml"), "w"
        ) as conf:
            yaml.dump(config, conf)
        return agent

    def setup_models(self) -> List[pytorch.nn.Module]:
        """
        The method sets up the models. Depending on the requirement of the agent, returns a list of
        models, all of which are loaded and initialized.
        @return List[pytorch.nn.Module]: List of models.
        """
        self.is_custom_model = self.sanity_check.check_model_init_sanity()
        if not self.is_custom_model:
            model_kwargs = {
                key: value
                for key, value in self.config["model_args"].items()
                if key in self.setup.get_model_args(self.config["model"])
            }
            if "activation" in model_kwargs.keys():
                activation = self.setup.get_activation(
                    activation_name=self.config["model_args"].get(
                        "activation", pytorch.nn.ReLU()
                    ),
                    activation_args=self.config.get("activation_args"),
                )
                model_kwargs["activation"] = activation
            if "exploration_tool" in model_kwargs:
                self.sanity_check.check_exploration_tool_sanity()
                exploration_tool = self.setup.get_exploration_tool(
                    exploration_tool_name=self.config["model_args"].get(
                        "exploration_tool"
                    ),
                    exploration_tool_args=self.config["exploration_tool_args"],
                )
                model_kwargs["exploration_tool"] = exploration_tool
            models = self.setup.get_models(
                model_name=self.config["model"],
                agent_name=self.config["agent"],
                **model_kwargs,
            )
        else:
            self.agent_model_args = self.setup.get_agent_model_args(
                agent_name=self.config["agent"]
            )
            models = [
                self.config["agent_args"][agent_model_arg]
                for agent_model_arg in self.agent_model_args
            ]
        return models

    def setup_environment(self):
        env = self.config["env"]
        if isinstance(env, str):
            env_args = self.config.get("env_args", dict())
            env = gym.make(env, **env_args)
        return env

    @staticmethod
    def setup_summary_writer(save_path: str) -> SummaryWriter:
        """
        Sets up summary writer for Tensorboard logger.
        @param save_path: str: The save path for agents. A new directory "tensorboard_logs" is created is save_path
            is a directory. If path for agent file is passed, "tensorboard_logs" directory is created in the parent
            directory.
        @return SummaryWriter: The instance of summary writer for Tensorboard logging.
        """
        path = Path(save_path)
        if not path.exists():
            raise NotADirectoryError(
                "Given path is not a valid directory or a valid path to a file name"
            )
        if path.is_dir():
            save_path = os.path.join(save_path, "tensorboard_logs")
            os.makedirs(save_path, exist_ok=True)
            summary_writer = SummaryWriter(log_dir=save_path)
        else:
            save_path = os.path.join(str(path.parent), "tensorboard_logs")
            os.makedirs(save_path, exist_ok=True)
            summary_writer = SummaryWriter(log_dir=save_path)
        return summary_writer

    @staticmethod
    def prepare_config_to_save(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares config for saving as a YAML file. This removes all values except
        str, int, bool, list, tuple, dict from the dictionary.
        @param config: Dict[str, Any]: The config dictionary.
        @return: Dict[str, Any]: The prepared dictionary
        """
        to_save_config = dict()
        # Iterate through dictionary.
        for key, value in config.items():
            # Check if of valid datatypes.
            if not isinstance(value, (str, bool, int, list, tuple, dict)):
                continue
            # If list or tuple, call the function recursively to check for each item.
            if isinstance(value, (list, tuple)):
                if len(value) > 0:
                    values_from_recursion = list()
                    for item in value:
                        rec = Simulator.prepare_config_to_save({"rec": item})
                        if len(rec) != 0:
                            values_from_recursion.append(item)
                    if len(values_from_recursion) == 0:
                        continue
            # If dictionary, call function recursively to check for each item.
            if isinstance(value, dict):
                value = Simulator.prepare_config_to_save(value)
                if len(value) == 0:
                    continue
            # Save accepted values to dictionary
            to_save_config[key] = value
        return to_save_config

    def run(self, metrics_logging_frequency: int, **kwargs) -> None:
        """
        This method runs the simulator.
        @param metrics_logging_frequency: int: The logging frequency for rewards.
        @param kwargs: Additional keyword arguments for the training run.
        """
        if self.trainer.is_train():
            self.trainer.train_agent(
                metrics_logging_frequency=metrics_logging_frequency,
                render=kwargs.get("render", self.config.get("render", False)),
                load=kwargs.get("load", self.config.get("load", False)),
            )
        elif self.trainer.is_eval():
            self.trainer.evaluate_agent(
                metrics_logging_frequency=metrics_logging_frequency,
                render=kwargs.get("render", self.config.get("render", False)),
            )
        else:
            raise ValueError("Invalid mode passed! Must be in `train` or `eval`")
        return

import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from numpy import ndarray

from rlpack import pytorch
from rlpack.environments.environments import Environments
from rlpack.utils.base import Agent
from rlpack.utils.register import Register


class Simulator:
    """
    Simulator class simulates the environments and runs the agent through the environment. It also sets up
    the models and agents for training and/or evaluation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        algorithm: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        """
        @:param config (Optional[Dict[str, Any]]): The configuration dictionary for setup. Default: None
        @:param algorithm (Optional[str]): The algorithm to be used. Default: None
        @:param environment (Optional[str]): The environment to be used. Default: None

        Note that either `config` or both `algorithm` and `environment` must be passed.
        """
        self.register = Register()
        # Check arguments and select config.
        if config is None and algorithm is None and environment is None:
            raise ValueError(
                "At least one of the arguments, `config`, `algorithm` or `environments` must be passed!"
            )
        if config is None and algorithm is not None:
            config = self.register.get_default_config(algorithm)
        if environment is not None:
            config["env_name"] = environment

        self.config = config
        self.agent = self.setup_agent()
        self.env = Environments(agent=self.agent, config=self.config)
        return

    def setup_agent(self) -> Agent:
        """
        This method sets up agent by loading all the necessary arguments.
        @:return (Agent): The loaded and initialized agent.
        """
        models = self.setup_models()
        agent_args_for_models = [
            arg
            for arg in self.register.agent_args[self.config["model_name"]]
            if arg
            in self.register.model_args_for_agents[self.config["model_name"]].keys()
        ]
        agent_model_kwargs = {
            arg: models[idx] for idx, arg in enumerate(agent_args_for_models)
        }
        trainable_models = [
            model
            for model_arg_name, model in agent_model_kwargs.items()
            if self.register.model_args_for_agents[self.config["model_name"]][
                model_arg_name
            ]
        ]
        optimizers = [
            self.register.get_optimizer(
                params=model.parameters(),
                optimizer_args=self.config["optimizer_args"],
            )
            for model in trainable_models
        ]
        lr_schedulers = [
            self.register.get_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler_args=self.config.get("lr_scheduler_args"),
            )
            for optimizer in optimizers
        ]
        save_path = (
            self.config["agent_args"].get("save_path")
            if self.config["agent_args"].get("save_path") is not None
            else os.getenv("SAVE_PATH")
        )
        processed_agent_args = dict(
            **agent_model_kwargs,
            optimizer=optimizers[0] if len(optimizers) == 1 else optimizers,
            lr_scheduler=lr_schedulers[0] if len(lr_schedulers) == 1 else lr_schedulers,
            loss_function=self.register.get_loss_function(
                loss_function_args=self.config["loss_function_args"]
            ),
            save_path=save_path,
            device=self.config["device"],
            apply_norm=self.register.get_apply_norm_mode_code(
                self.config["agent_args"].get("apply_norm", "none")
            ),
            apply_norm_to=self.register.get_apply_norm_to_mode_code(
                self.config["agent_args"].get("apply_norm_to", ("none",))
            ),
            eps_for_norm=self.config["agent_args"].get("eps_for_norm", 5e-8),
            p_for_norm=self.config["agent_args"].get("p_for_norm", 2),
            dim_for_norm=self.config["agent_args"].get("dim_for_norm", 0),
        )
        agent_args_from_config = {
            k: v
            for k, v in self.config["agent_args"].items()
            if k not in processed_agent_args.keys()
        }
        processed_agent_args = {
            **processed_agent_args,
            **agent_args_from_config,
        }
        agent_kwargs = {
            k: processed_agent_args[k]
            for k in self.register.agent_args[self.config["model_name"]]
        }
        agent = self.register.get_agent(
            agent_name=self.config["agent_name"], **agent_kwargs
        )
        with open(os.path.join(save_path, "config.yaml"), "w") as conf:
            yaml.dump(self.config, conf)

        return agent

    def setup_models(self) -> List[pytorch.nn.Module]:
        """
        The method sets up the models. Depending on the requirement of the agent, returns a list of
        models, all of which are loaded and initialized.
        @:return (List[pytorch.nn.Module]): List of models.
        """
        activation = self.register.get_activation(
            activation_args=self.config["activation_args"]
        )
        model_kwargs = {
            k: self.config["model_args"][k] if k != "activation" else activation
            for k in self.register.get_model_args(self.config["model_name"])
        }
        models = self.register.get_models(
            model_name=self.config["model_name"],
            **model_kwargs,
        )

        return models

    def run(self, **kwargs) -> None:
        """
        This method runs the simulator.

        @:param kwargs: Additional keyword arguments for the training run.
        """
        if self.env.is_train():
            self.env.train_agent(**kwargs)
        elif self.env.is_eval():
            self.env.evaluate_agent()
        else:
            raise ValueError("Invalid mode passed! Must be in `train` or `eval`")
        return

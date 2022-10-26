from typing import Any, Dict


class SanityCheck:
    def __new__(cls, input_config: Dict[str, Any]):
        """
        This class does the basic sanity check of input_config.
        @:param input_config (Dict[str, Any]): The input config that is to be used for training.
        """
        mandatory_keys = (
            "mode",
            "env_name",
            "model_name",
            "agent_name",
            "num_episodes",
            "max_timesteps",
            "reward_logging_frequency",
            "render",
            "model_args",
            "optimizer_args",
        )

        args = list(input_config.keys())
        missing_args = [k in args for k in mandatory_keys]
        if not all(missing_args):
            raise ValueError(
                f"The following arguments were not received: "
                f" {[args[idx] for idx, arg in enumerate(missing_args) if arg is False]}"
            )
        return

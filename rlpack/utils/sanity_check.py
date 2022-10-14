from typing import Dict, Any


class SanityCheck:
    def __new__(cls, input_config: Dict[str, Any]):
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
            "optimizer_args"
        )

        args = list(input_config.keys())
        missing_args = [k in mandatory_keys for k in args]
        if not all(missing_args):
            raise ValueError(
                f"The following arguments were not received: "
                f" {[args[idx] for idx, arg in enumerate(missing_args) if arg is False]}"
            )
        return

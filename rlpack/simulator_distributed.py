import logging
import os
from typing import Any, Dict, Protocol

from rlpack import dist, mp
from rlpack.simulator import Simulator


class SimulatorDistributed:
    """
    Similar to rlpack.simulator.Simulator, SimulatorDistributed class sets up agents and runs simulation by
    interacting with the given environment. This class simulates the agent-environment interaction parallel, i.e.
    there will be multiple agents interacting with their local copy of environment. Agents are responsible
    for maintaining any synchronization.
    """

    class RunFuncSignature(Protocol):
        """
        Typing hint for function to be spawned in multiprocess.
        """

        def __call__(
            self, process_rank: int, world_size: int, config: Dict[str, Any], **kwargs
        ) -> None:
            """
            __call__ method to define the signature for the callable.
            @param process_rank: int: The process rank of the initialized process.
            @param world_size: int: Total number of processes launched or to be launched.
            @param config: Dict[str, Any]: The configuration to be used.
            @param kwargs: Other keyword arguments corresponding to
                rlpack.environments.environments.Environments.train method.
            """
            return

    def __init__(self, n_procs: int, config: Dict[str, Any], backend: str = "gloo"):
        """
        Initialization of class SimulatorDistributed.
        @param n_procs: int: The total number of processes to be launched.
        @param config: Dict[str, Any]: The configuration to be used.
        @param backend: str: The PyTorch multiprocessing backend to be used. Default: "gloo"; the Gloo backend. More
            information can be found [here](https://pytorch.org/docs/stable/distributed.html#module-torch.distributed)
        """
        self.n_procs = n_procs
        self.config = config
        self.backend = backend

    @staticmethod
    def init_process(
        process_rank: int,
        world_size: int,
        func: RunFuncSignature,
        config: Dict[str, Any],
        backend: str = "gloo",
        **kwargs,
    ) -> None:
        """
        Initialized the distributed environment to run the given func.
        @param process_rank: int: The process rank of the initialized process.
        @param world_size: int: Total number of processes launched or to be launched.
        @param func: RunFuncSignature: A function with given signature to be launched
            in distributed setting on processes.
        @param config: Dict[str, Any]: The configuration to be used.
        @param backend: str: The PyTorch multiprocessing backend to be used.
        @param kwargs: Other keyword arguments for `func`.
        """
        dist.init_process_group(backend, rank=process_rank, world_size=world_size)
        func(process_rank, world_size, config, **kwargs)

    @staticmethod
    def run_(
        process_rank: int, world_size: int, config: Dict[str, Any], **kwargs
    ) -> None:
        """
        Launches the rlpack.simulator.Simulator in distributed setting.
        @param process_rank: int: The process rank of the initialized process.
        @param world_size: int: Total number of processes launched or to be launched.
        @param config: Dict[str, Any]: The configuration to be used.
        @param kwargs: Other keyword arguments corresponding to
            rlpack.environments.environments.Environments.train method.
        """
        if "distributed_mode" in kwargs.keys():
            kwargs.pop("distributed_mode")
        logging.info(f"Launched process: {process_rank} out of {world_size} processes.")
        if "WORLD_SIZE" not in os.environ.keys():
            os.environ["WORLD_SIZE"] = f"{world_size}"
        if "RANK" not in os.environ.keys():
            os.environ["RANK"] = f"{process_rank}"
        simulator = Simulator(config, True)
        simulator.run(**kwargs, distributed_mode=True)

    def run(self, **kwargs):
        """
        Runs the simulation in distributed setting.
        @param kwargs: Other keyword arguments corresponding to
            rlpack.environments.environments.Environments.train method.
        """
        # List to store the Process objects.
        processes = list()
        # Check for environment variables and set them if required.
        if "MASTER_ADDR" not in os.environ.keys():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ.keys():
            os.environ["MASTER_PORT"] = "29500"
        if "WORLD_SIZE" not in os.environ.keys():
            os.environ["WORLD_SIZE"] = f"{self.n_procs}"
        # Set start method to spawn
        mp.set_start_method("spawn")
        config = self.config.copy()
        # Start each process
        for rank in range(self.n_procs):
            p = mp.Process(
                target=self.init_process,
                args=(rank, self.n_procs, self.run_, config, self.backend),
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)
        # Final process join
        for p in processes:
            p.join()

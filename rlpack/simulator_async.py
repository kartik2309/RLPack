import logging
import os

from rlpack import dist, mp
from rlpack.simulator import Simulator


class SimulatorAsync:
    @staticmethod
    def init_process(process_rank, world_size, func, config, backend="gloo", **kwargs):
        """Initialize the distributed environment."""
        dist.init_process_group(backend, rank=process_rank, world_size=world_size)
        func(process_rank, world_size, config, **kwargs)

    @staticmethod
    def run_(process_rank, world_size, config, **kwargs):
        if "async_mode" in kwargs.keys():
            kwargs.pop("async_mode")
        logging.info(
            f"Launched process: {process_rank} out of {world_size} processes.\n"
        )
        if "WORLD_SIZE" not in os.environ.keys():
            os.environ["WORLD_SIZE"] = f"{world_size}"
        if "RANK" not in os.environ.keys():
            os.environ["RANK"] = f"{process_rank}"
        simulator = Simulator(config)
        simulator.run(**kwargs, distributed_mode=True)

    def run(self, n_procs, config, backend="gloo", **kwargs):
        processes = list()
        if "MASTER_ADDR" not in os.environ.keys():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ.keys():
            os.environ["MASTER_PORT"] = "29500"
        if "WORLD_SIZE" not in os.environ.keys():
            os.environ["WORLD_SIZE"] = f"{n_procs}"
        mp.set_start_method("spawn")
        for rank in range(n_procs):
            p = mp.Process(
                target=self.init_process,
                args=(rank, n_procs, self.run_, config, backend),
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

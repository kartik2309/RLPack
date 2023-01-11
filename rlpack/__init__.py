"""!
@package rlpack
@brief Implementation of RL Algorithms built on top of PyTorch. Heavy workloads have
been optimized with C++ backend.
"""

# Import other packages
from site import getsitepackages

# Import pytorch packages
import torch as pytorch
import torch.distributed as pytorch_distributed
import torch.distributions as pytorch_distributions
import torch.multiprocessing as pytorch_multiprocessing
from torch.utils.tensorboard import SummaryWriter

# Import CPP Backend
from rlpack.lib import C_GradAccumulator, C_ReplayBuffer, C_RolloutBuffer, StlBindings


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Function Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def get_prefix_path():
    """
    Gets prefix path for rlpack package, from python installation.
    @return: str: The prefix path to rlpack.
    """
    return f"{getsitepackages()[0]}/rlpack"

"""!
@package rlpack
@brief Implementation of RL Algorithms built on top of PyTorch. Heavy workloads have
been optimized with C++ backend.
"""

# Import pytorch packages
import torch as pytorch
import torch.distributed as dist
import torch.distributions as distributions_math
import torch.multiprocessing as mp

# Import CPP Backend
from rlpack.lib import C_GradAccumulator, C_Memory

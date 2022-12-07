"""!
@package rlpack
@brief Implementation of RL Algorithms built on top of PyTorch. Heavy workloads have
been optimized with C++ backend.
"""

import torch as pytorch

# Import CPP Backend
from rlpack.lib import C_Memory

"""!
@package rlpack._C
@brief This package implements the classes to interface between C++ and Python.


Currently following classes have been implemented:
    - `ReplayBuffer`: Implemented as rlpack._C.replay_buffer.ReplayBuffer, this class is responsible for using
        Optimized C_ReplayBuffer class implemented in C++ and providing simple Python methods to access it.
    - `GradAccumulator`: Implemented as rlpack._C.grad_accumulator.GradAccumulator, this class is responsible for
        using optimized C_GradAccumulator class implemented in C++ and providing simple python methods to access it.
    - `RolloutBuffer`: Implemented as rlpack._C.rollout_buffer.RolloutBuffer, this class is responsible for using
        C_RolloutBuffer class implemented in C++ and providing simple python methods to access it.
"""

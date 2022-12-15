"""!
@package rlpack._C
@brief This package implements the classes to interface between C++ and Python.


Currently following classes have been implemented:
    - `Memory`: Implemented as rlpack._C.memory.Memory, this class is responsible for using Optimized C_Memory class
        implemented in C++ and providing simple Python methods to access it.
    - `GradAccumulator`: Implemented as rlpack._C.grad_accumulator.GradAccumulator, this class is responsible for
        using optimized GradAccumulator class implemented in C++ and providing simple python methods to access it.
"""


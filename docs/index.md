# RLPack Documentation {#intro}

RLPack provides an easy-to-use interface for Deep RL Algorithms.
RLPack is built on top [PyTorch](https://pytorch.org) and leverages accelerators available 
on your machine to train the agents' models. Heavier computational requirements outside the scope of PyTorch have been
implemented and optimized with C++ backend using OpenMP routines for CPUs and CUDA Kernels for 
CUDA enabled GPUs. OpenAI's [gym](https://www.gymlibrary.dev) is used to simulate the environment for training.

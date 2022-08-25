# RLPack

Implementation of RL Algorithms with PyTorch. Heavy workloads have
been optimized with C++ backend.

# Installation

To install RLPack, simply follow the following steps: <br>

* Clone and install
    ```zsh
    git clone https://github.com/kartik2309/RLPack.git
    cd RLPack 
    pip install .
    ```
  This will install the package in your python environment.

* To get started, you can run the following snippet:
  ```python 
  import os
  from rlpack.simulator import Simulator

  # Optional, to set SAVE_PATH environment variable. 
  os.environ["SAVE_PATH"] = 'models/'
  os.makedirs(f"{os.getenv('SAVE_PATH')}, exist_ok=True")

  simulator = Simulator(algorithm='dlqn1d', environment='LunarLander-v2')
  simulator.run()
  ```
  This will load the `dlqn1d` algorithm (which is loaded by default configuration)
  and run it in `LunarLander-v2` environment.

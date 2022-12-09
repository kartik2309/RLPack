# RLPack

# Introduction

Implementation of RL Algorithms built on top of PyTorch. Heavy workloads have
been optimized with C++ backend.

# Installation

To build and install RLPack from source, simply follow the following steps: <br>

* Clone and install
    ```zsh
    git clone https://github.com/kartik2309/RLPack.git
    cd RLPack 
    pip install .
    ```
  This will install the package in your python environment.

# Getting started 

* To get started, you can run the following snippet:
  ```python 
  import os
  import yaml
  from rlpack.simulator import Simulator

  # Optional, to set SAVE_PATH environment variable. 
  os.environ["SAVE_PATH"] = 'models/'
  os.makedirs(f"{os.getenv('SAVE_PATH')}, exist_ok=True")
  
  # Read from a yaml file with all the arguments for easy use.
  with open("config.yaml", "rb) as f:
    yaml.load(f, yaml.Loader)
  
  # Pass the config dictionary to Simulator.
  simulator = Simulator(config=config)
  simulator.run()
  ```
  
  `Simulator` accepts a config dictionary which must contain all mandatory arguments.
  
# Issues and bugs
Please raise an issue on GitHub to report a bug. Pull Requests can be raised after discussion on the raised issue.

# License
RLPack is released under [MIT LICENSE](https://github.com/kartik2309/RLPack/blob/master/LICENSE.md).  

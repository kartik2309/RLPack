"""!
@package environments
@brief This package implements the gym environment to couple it with selected environment.


This package implements Environments class as rlpack.environments.environments.Environments. This class couples the
agent we selected with the environment we pass/select. It provides basic methods for training and evaluating an agent.
This class also logs rewards and other metrics on the screen.
"""

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""!
@package rlpack.dqn
@brief This package implements the DQN methods.


Currently following classes have been implemented:
    - `Dqn`: This class is a helper class that selects the correct variant of DQN agent based on argument
        `prioritization_params`.
    - `DqnUniformAgent`: Implemented as rlpack.dqn.dqn_uniform_agent.DqnUniformAgent this class implements the basic
        DQN methodology, i.e. without prioritization.
    - `DqnProportionalAgent`: Implemented as rlpack.dqn.dqn_proportional_prioritization_agent.DqnProportionalAgent
        this class implements the DQN with proportional prioritization.
    - `DqnRankBasedAgent`: Implemented as rlpack.dqn.dqn_rank_based_agent.DqnRankBasedAgent; this class implements the
        DQN with rank prioritization.
"""


from rlpack.dqn.dqn import Dqn

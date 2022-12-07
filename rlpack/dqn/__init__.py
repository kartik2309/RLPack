"""!
@package rlpack.dqn
@brief This package implements the DQN methods.


Currently following classes have been implemented:
    - Dqn: This class is a helper class that selects the correct variant of DQN agent based on argument
        `prioritization_params`.
    - DqnAgent: Implemented as rlpack.dqn.dqn_agent.DqnAgent this class implements the basic DQN methodology, i.e.
        without prioritization. It also acts as a base class for DQN agents with prioritization strategies.
    - DqnProportionalPrioritizationAgent: Implemented as
        rlpack.dqn.dqn_proportional_prioritization_agent.DqnProportionalPrioritizationAgent this class implements the
         DQN with proportional prioritization.
    - DqnRankBasedPrioritizationAgent: Implemented as
        rlpack.dqn.dqn_rank_based_prioritization_agent.DqnRankBasedPrioritizationAgent; this class implements the
        DQN with rank prioritization.
"""


from rlpack.dqn.dqn import Dqn

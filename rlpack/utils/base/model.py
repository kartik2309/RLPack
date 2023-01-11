from abc import abstractmethod

from rlpack import pytorch


class Model(pytorch.nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.has_exploration_tool = False
        pass

    @abstractmethod
    def forward(self, *args) -> pytorch.Tensor:
        pass

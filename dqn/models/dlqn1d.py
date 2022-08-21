from typing import List, TypeVar

import torch

Activation = TypeVar("Activation")


class Dlqn1d(torch.nn.Module):
    def __init__(
        self,
        sequence_length: int,
        hidden_sizes: List[int],
        num_actions: int,
        activation: Activation = torch.nn.ReLU,
        dropout: float = 0.5,
    ):
        super(Dlqn1d, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_sizes = hidden_sizes.copy()
        self.num_action = num_actions
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout)

        self.num_blocks = len(self.hidden_sizes) - 1

        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_module_dict = torch.nn.ModuleDict(
            {
                f"layer_{idx}": torch.nn.Linear(
                    in_features=self.hidden_sizes[idx],
                    out_features=self.hidden_sizes[idx + 1],
                )
                for idx in range(self.num_blocks)
            }
        )

        self.final_fc = torch.nn.Linear(
            in_features=self.hidden_sizes[-1] * sequence_length,
            out_features=num_actions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer_idx, layer_info in enumerate(self.linear_module_dict.items()):
            name, layer = layer_info
            x = layer(x)
            x = self.activation(x)

        x = self.dropout(x)
        x = self.flatten(x)
        x = self.final_fc(x)

        return x

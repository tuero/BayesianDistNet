"""
File: distnet.py
Date: April 4, 2020
Description: Distnet FCN model definition
"""

# Pytorch
import torch
import torch.nn as nn

# Module
from config.config_handler import getModelConfig


class DistNetFCN(nn.Module):
    """
    Implementation of the DistNet model (Eggensperger et al, 2018)
    """

    def __init__(self, num_features, config: dict = getModelConfig(), name_suffix: str = ""):
        super().__init__()

        self.name_suffix = name_suffix

        # Model config
        self.n_fcdepth         = config['n_fcdepth']
        self.output_size       = config['output_size']
        self.drop_value        = config['drop_value']

        # Layer 1
        in_channel = num_features
        out_channel = self.n_fcdepth
        self.layer1 = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.Tanh(),
            nn.Dropout(self.drop_value)
        )

        # Layer 2
        in_channel = out_channel
        out_channel = self.n_fcdepth
        self.layer2 = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.Tanh(),
            nn.Dropout(self.drop_value)
        )

        # Layer 3: Final output from model
        in_channel = out_channel
        out_channel = self.output_size
        self.layer_end = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            # nn.BatchNorm1d(out_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer_end(x)
        return torch.exp(x)

    # Helper function to get the shape of the input when flattened
    def num_flat_features(self, x: torch.Tensor) -> int:
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    # Helper function for debugging
    def toStr(self) -> str:
        return "distnet" + self.name_suffix

"""
File: bayes_distnet.py
Date: April 4, 2020
Description: Bayesian version Distnet FCN model definition
"""

# PyTorch
import torch.nn as nn

# Module
from layers.misc import ModuleWrapper
from layers.BBBLinear import BBBLinear
from config.config_handler import getModelConfig


class BayesDistNetFCN(ModuleWrapper):
    """
    Implementation of the modified DistNet model (Eggensperger et al, 2018)
    Bayesian layers are used in place of traditional dense layers.
    """

    def __init__(self, num_features, config: dict = getModelConfig(), name_suffix: str = ""):
        super().__init__()
        self.name_suffix = name_suffix

        # Model config
        self.n_fcdepth         = config['n_fcdepth']
        self.output_size       = config['output_size']
        self.drop_value        = config['drop_value']

        # # Layer 1
        in_channel = num_features
        out_channel = self.n_fcdepth
        self.layer1 = nn.Sequential(
            BBBLinear(in_channel, out_channel, config, bias=True, name='fc1'),
            nn.BatchNorm1d(out_channel),
            nn.Softplus(),
        )

        # Layer 2
        in_channel = out_channel
        out_channel = self.n_fcdepth
        self.layer2 = nn.Sequential(
            BBBLinear(in_channel, out_channel, config, bias=True, name='fc2'),
            nn.BatchNorm1d(out_channel),
            nn.Softplus(),
        )

        # Layer 3: Final output from model
        in_channel = out_channel
        out_channel = self.output_size
        self.layer_end = nn.Sequential(
            BBBLinear(in_channel, out_channel, config, bias=True, name='fc3'),
            # nn.BatchNorm1d(out_channel),
            nn.Softplus()
        )


    # Helper function for debugging
    def toStr(self) -> str:
        return "Bayes_Distnet" + "_" + self.name_suffix

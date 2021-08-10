"""
File: model_util.py
Date: April 4, 2020
Description: Uility functions for model use
"""

# Pytorch
import torch
import torch.nn as nn


def getModel(net_type, model_config, num_features):
    """
    Get the model object

    Args:
        net_type (str): String name of model type
        model_config (dict): Dictionary representing model configuration

    Returns:
        Torch module object
    """
    from models.distnet import DistNetFCN
    from models.bayes_distnet import BayesDistNetFCN

    if net_type == 'distnet':
        return DistNetFCN(num_features, model_config)
    elif net_type == 'bayes_distnet':
        return BayesDistNetFCN(num_features, model_config)
    else:
        raise ValueError('Unknown net type.')


def weights_init(m):
    """
    Initializes the weights for the given module using the Xavier Uniform method

    Args:
        m (torch.nn.Module): The module to initalize the weights
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.zero_()
        m.bias.data.fill_(0.01)


def count_parameters(model):
    """
    Counts the number of parameters for the given model

    Args:
        model (torch.nn.): The model to examine
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

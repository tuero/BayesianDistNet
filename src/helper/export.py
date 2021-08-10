"""
File: export.py
Date: April 4, 2020
Description: Export the model output data and serialized model
"""

# Library
import pickle
import logging

# Numerical
import numpy as np

# Torch
import torch

# Module
from helper import device
from models.bayes_distnet import BayesDistNetFCN


def dump_res(save_path, data):
    logger = logging.getLogger()
    with open(save_path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Dumped to {}".format(save_path))


def exportData(X_data, data_path, model):
    outputs = []
    model.eval()
    for val_data in X_data:
        val_inputs = torch.tensor([val_data], dtype=torch.float).to(device.device)
        val_inputs = val_inputs.repeat(64, 1).to(device.device)
        if type(model) is BayesDistNetFCN:
            rts = []
            # for _ in range(50):
            for _ in range(16):
                val_pred = model(val_inputs)[0]
                rts = rts + val_pred.flatten().cpu().detach().tolist()
            outputs.append(rts)
        else:
            val_inputs = torch.tensor([val_data], dtype=torch.float)
            val_inputs = val_inputs.to(device.device)
            val_pred = model(val_inputs)
            outputs += val_pred.cpu().detach().tolist()

    dump_res(data_path, np.array(outputs))
    pass


def saveModel(model, model_path):
    logger = logging.getLogger()
    logger.info('Exporting model to {}'.format(model_path))
    torch.save(model, model_path)
    pass

"""
File: data_handler.py
Date: May 10, 2020
Description: Dataset for pytorch model
"""

# Numeric
import numpy as np

# Pytorch
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

# Module
from helper import preprocess


class CustomDataset(Dataset):
    """
    Since total dataset is small, we load entire dataset onto GPU so we are not loading data onto GPU during
    the training loop.
    """

    def __init__(self, features, runtimes, train_idx, validate_idx, num_train_samples, seed, lb, device_num):
        X_trn_flat, X_vld_flat, y_trn_flat, y_vld_flat, _, _ = \
            preprocess.preprocess(features, runtimes, train_idx, validate_idx, num_train_samples, lb, seed, True)

        device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")
        self.features = torch.tensor(np.concatenate((X_trn_flat, X_vld_flat)), dtype=torch.float).to(device)
        self.observations = torch.tensor(np.concatenate((y_trn_flat, y_vld_flat)), dtype=torch.float).to(device)
        self.train_idx = range(len(X_trn_flat))
        self.validate_idx = range(len(X_trn_flat), len(X_trn_flat) + len(X_vld_flat))

    def getTrainValidSubsetSampler(self):
        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.validate_idx)
        return train_sampler, valid_sampler

    def getNumFeatures(self):
        return self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.features[idx], self.observations[idx])

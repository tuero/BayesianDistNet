"""
File: eval_model.py
Date: May 10, 2020
Description: Data import
Source: https://github.com/KEggensperger/DistNet/
"""

import sys
import argparse
import os
import logging
from datetime import datetime

# Stats
import numpy as np
from sklearn.model_selection import KFold

# Torch
import torch
from torch.utils.data import DataLoader

# Module
from helper import load_data, data_source_release, preprocess
from config.config_handler import parseConfig
from data_handler.data_handler import CustomDataset
from train.train import train
from helper import device
from helper.export import exportData, saveModel


PROJ_DIR = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def runKFold(scenario, fold, num_train_samples, seed, net_type, config_section, device_num, lb, mode='kfold', ep=-1, ex=-1):
    device.device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")

    BASE_PATH = PROJ_DIR + '/export/{}/'.format(net_type)

    # Parse and get configs
    training_config, model_config = parseConfig(config_section)
    if ep > 0:
        training_config['n_epochs'] = ep
    if ex > 0:
        training_config['n_expected_epochs'] = ex

    # Load data
    sc_dict = data_source_release.get_sc_dict()
    data_dir = data_source_release.get_data_dir()
    runtimes, features, sat_ls = load_data.get_data(scenario=scenario, data_dir=data_dir, sc_dict=sc_dict, retrieve=sc_dict[scenario]['use'], log=True)

    features = np.array(features)
    runtimes = np.array(runtimes)

    # Get Kfold splits
    idx = list(range(runtimes.shape[0]))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    cntr = -1
    for train_idx, test_idx in kf.split(idx):
        # Reset seed for every instance
        np.random.seed(2)
        cntr += 1
        if cntr != fold:
            continue

        # Split training set into train/validate
        sp_r = 1.0 - training_config['split_ratio']
        train_indices = train_idx[:int(len(train_idx) * sp_r)]
        valid_indices = train_idx[int(len(train_idx) * sp_r):]

        batch_size = training_config['batch_size']
        dataset = CustomDataset(features, runtimes, train_indices, valid_indices, num_train_samples, seed, lb, device_num)
        train_sampler, valid_sampler = dataset.getTrainValidSubsetSampler()
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        model_path = BASE_PATH + '{}_{}_{}_{}_{}_{}_{}.pt'.format(net_type, scenario, config_section, num_train_samples, lb, cntr, seed)
        data_path_train = BASE_PATH + '{}_{}_{}_{}_{}_{}_{}_train.pkl'.format(net_type, scenario, config_section, num_train_samples, lb, cntr, seed)
        data_path_test = BASE_PATH + '{}_{}_{}_{}_{}_{}_{}_test.pkl'.format(net_type, scenario, config_section, num_train_samples, lb, cntr, seed)
        X_train = features[train_indices, :]
        X_valid = features[valid_indices, :]
        X_test = features[test_idx, :]
        X_train, X_valid, X_test = preprocess.preprocess_features(X_train, X_valid, X_test, scal="meanstd")
        if mode == 'kfold':
            directory = os.path.dirname(BASE_PATH)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Export model
            model = train(dataset.getNumFeatures(), net_type, train_loader, valid_loader, X_test,
                          model_path, data_path_test, training_config, model_config)
            # exportData(X_test, data_path_test, model)
            # saveModel(model, model_path)
        elif mode == 'validate':
            model = torch.load(model_path)
        else:
            raise ValueError('Unknown mode type')


if __name__ == "__main__":
    sc_dict = data_source_release.get_sc_dict()

    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", dest="scenario", required=True, choices=sc_dict.keys())
    parser.add_argument("--num_train_samples", dest="num_train_samples", default=100, type=int)
    parser.add_argument("--fold", dest="fold", required=True, type=int)
    parser.add_argument("--seed", dest="seed", required=False, default=1, type=int)
    parser.add_argument("--net_type", help="Model type", required=False, choices=["distnet", "bayes_distnet"],
                        default="distnet", type=str.lower)
    parser.add_argument("--config_section", help="Configuration to use for training", required=False,
                        default="DEFAULT", type=str.upper)
    parser.add_argument("--mode", help="Mode: Type of script to run.", required=False,
                        choices=["kfold", "validate"], type=str.lower, default="kfold")
    parser.add_argument("--device_num", dest="device_num", required=False, default=0, type=int, choices=[0, 1])
    parser.add_argument("--lb", dest="lb", required=False, default=0, type=int, choices=[0, 20, 40, 50, 60, 80, 100])
    parser.add_argument("--epochs", dest="ep", required=False, default=-1, type=int)
    parser.add_argument("--ex", dest="ex", required=False, default=-1, type=int)

    args = parser.parse_args()

    # Set logger settings
    base_path = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    str_now = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    log_path = PROJ_DIR + "/logs/{}_{}_{}_{}_{}_{}.log".format(args.config_section, args.scenario, args.fold, args.num_train_samples, args.seed, str_now)
    directory = os.path.dirname(log_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    runKFold(args.scenario, args.fold, args.num_train_samples, args.seed, args.net_type, args.config_section, args.device_num, args.lb, args.mode, args.ep, args.ex)

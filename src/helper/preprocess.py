"""
File: preprocess.py
Date: May 10, 2020
Description: Data preprocessing
Source: https://github.com/KEggensperger/DistNet/
"""

import numpy as np
import logging
import copy


def remove_timeouts(runningtimes, cutoff, features=None, sat_ls=None, log=False):
    """
    Remove all instances with more than one value >= cutoff
    """

    if features is None:
        features = [0] * runningtimes.shape[0]
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance >= cutoff):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    if log:
        print("Discarding {:d} ({:d}) instances because not stated TIMEOUTS".format(len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def remove_instances_with_status(runningtimes, features, sat_ls=None, status="CRASHED", log=False):
    if sat_ls is None:
        print("Could not remove {} instances".format(status))

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if status not in s:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)
    if log:
        print("Discarding {:d} ({:d}) instances because of {}".format(len(features) - len(new_ft), len(features), status))
    return np.array(new_rt), np.array(new_ft), new_sl


def remove_constant_instances(runningtimes, features, sat_ls=None, log=False):
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if np.std(f) > 0:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)
    if log:
        print("Discarding {:d} ({:d}) instances because of constant features".format(len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def feature_imputation(features, impute_val=-512, impute_with="median"):
    print(features.shape)
    if impute_with == "median":
        for col in range(features.shape[1]):
            med = np.median(features[:, col])
            features[:, col] = [med if i == impute_val else i for i in features[:, col]]
    return features


def remove_zeros(runningtimes, features=None, sat_ls=None, log=False):
    """
    Remove all instances with more than one value == 0
    """

    if features is None:
        features = [0] * runningtimes.shape[0]
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance <= 0):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    if log:
        print("Discarding {:d} ({:d}) instances because of ZEROS".format(len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def det_constant_features(X, log=False):
    """
    Return a list with constant features
    :param X:
    :return:
    """
    max_ = X.max(axis=0)
    min_ = X.min(axis=0)
    diff = max_ - min_

    det_idx = np.where(diff <= 10e-10)
    if log:
        print("Discarding {:d} ({:d}) features".format(det_idx[0].shape[0], X.shape[1]))
    return det_idx


def det_transformation(X):
    """
    Return min max scaling
    """
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0) - min_
    return min_, max_


def preprocess_features(tra_X, val_X, test_X=[], scal="meanstd", log=False):
    # Remove constant features and rescale

    # Remove constant features
    del_idx = det_constant_features(tra_X, log)
    tra_X = np.delete(tra_X, del_idx, axis=1)
    val_X = np.delete(val_X, del_idx, axis=1)
    if len(test_X) > 0:
        test_X = np.delete(test_X, del_idx, axis=1)

    # Min/Max Scale instance features
    if scal == "minmax":
        min_, max_ = det_transformation(tra_X)
        tra_X = (tra_X - min_) / max_
        val_X = (val_X - min_) / max_
        if len(test_X) > 0:
            test_X = (test_X - min_) / max_
    else:
        mean_ = tra_X.mean(axis=0)
        std_ = tra_X.std(axis=0)
        tra_X = (tra_X - mean_) / std_
        val_X = (val_X - mean_) / std_
        if len(test_X) > 0:
            test_X = (test_X - mean_) / std_

    return tra_X, val_X, test_X


def preprocess(features, runtimes, train_idx, validate_idx, num_train_samples, lb, seed, log=False):
    logger = None
    if logging:
        logger = logging.getLogger()
    X_train = features[train_idx, :]
    X_valid = features[validate_idx, :]
    y_train = runtimes[train_idx]
    y_valid = runtimes[validate_idx]

    X_train, X_valid, _ = preprocess_features(X_train, X_valid, scal="meanstd")

    # flatten
    X_trn_flat = np.concatenate([[x for i in range(100)] for x in X_train])
    X_vld_flat = np.concatenate([[x for i in range(100)] for x in X_valid])
    y_trn_flat = y_train.flatten().reshape([-1, 1])
    y_vld_flat = y_valid.flatten().reshape([-1, 1])

    # Unfold data
    subset_idx = list(range(100))
    if num_train_samples != 100:
        if logging:
            logger.info("Cut data down to {:d} samples with seed {:d}".format(num_train_samples, seed))
        rs = np.random.RandomState(seed)
        rs.shuffle(subset_idx)
        subset_idx = subset_idx[:num_train_samples]
        logger.info(subset_idx)

        # Only shorten data used for training
        X_trn_flat = np.concatenate([[x for i in range(num_train_samples)] for x in X_train])
        y_train = y_train[:, subset_idx]
        y_trn_flat = y_train.flatten().reshape([-1, 1])

    # Min/Max Scale runnningtimes
    y_max_ = np.max(y_trn_flat)
    y_min_ = 0
    y_trn_flat = (y_trn_flat - y_min_) / y_max_
    y_vld_flat = (y_vld_flat - y_min_) / y_max_
    y_train = (y_train - y_min_) / y_max_
    y_valid = (y_valid - y_min_) / y_max_

    print(np.mean(y_train))

    # Set solution flag
    y_trn_flat = np.c_[y_trn_flat, np.ones(len(y_trn_flat))]
    y_vld_flat = np.c_[y_vld_flat, np.ones(len(y_vld_flat))]

    # Set the lowerbound (censored) samples
    logger.info("Using lb: {}".format(lb))
    if lb != 0:
        y_trn_flat = y_train.flatten()
        ytemp = copy.deepcopy(y_trn_flat)
        ytemp.sort(axis=0)
        censored_time = ytemp[int(num_train_samples * (1 - (lb / 100))) - 1]
        idx = int(len(ytemp) * (1 - (lb / 100)) - 1)
        censored_time = ytemp[idx]
        mask = y_trn_flat > censored_time
        y_trn_flat = np.where(~mask, y_trn_flat, censored_time)
        y_trn_flat = np.c_[y_trn_flat, np.where(~mask, 1, 0).flatten()]

    return X_trn_flat, X_vld_flat, y_trn_flat, y_vld_flat, y_train, y_valid

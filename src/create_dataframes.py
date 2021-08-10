"""
File: create_dataframes.py
Date: May 10, 2020
Description: Create combined df metrics from all runs. This can take a while to complete.
"""

import numpy as np
import pandas as pd
import pickle
import math
from scipy import stats
from sklearn.model_selection import KFold
from itertools import product
import multiprocessing as mp
import argparse

# Module
from helper import load_data, data_source_release


def getScenarioData(scenario):
    sc_dict = data_source_release.get_sc_dict()
    data_dir = data_source_release.get_data_dir()
    runtimes, features, _ = load_data.get_data(scenario=scenario, data_dir=data_dir, sc_dict=sc_dict, retrieve=sc_dict[scenario]['use'])
    return np.array(runtimes), np.array(features)


def getDataPickleArray(path):
    with open(path, 'rb') as file:
        arr = pickle.load(file)
    return arr


def getDataPerFold(runtimes, features, mode):
    idx = list(range(runtimes.shape[0]))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    data_out = []
    for train_idx, validate_idx in kf.split(idx):
        y_tra_run = runtimes[train_idx]
        y_val_run = runtimes[validate_idx]
        y_max_ = np.max(y_tra_run)
        y_min_ = 0
        y_tra_run = (y_tra_run - y_min_) / y_max_
        y_val_run = (y_val_run - y_min_) / y_max_
        if mode == "train":
            data_out.append(y_tra_run)
        else:
            data_out.append(y_val_run)
    return data_out


def getDistribution(net_type, dist, params):
    if net_type == "bayes_distnet":
        dist = stats.gaussian_kde(params)
    elif dist == "EXPONENTIAL":
        dist = stats.expon(loc=0, scale=params[0])
    elif dist == "INVGAUSS":
        dist = stats.invgauss(params[0] / params[1], loc=0, scale=params[1])
    elif dist == "LOGNORMAL":
        dist = stats.lognorm(s=params[0], loc=0, scale=params[1])
    elif dist == "NORMAL":
        dist = stats.norm(loc=params[0], scale=params[1])
    else:
        raise ValueError("Unknown dist: {}".format(dist))
    return dist


def getDistKS(dist, params, runtimes, net_type):
    if net_type == "bayes_distnet":
        return stats.ks_2samp(runtimes, params)
    if dist == "EXPONENTIAL":
        return stats.kstest(runtimes, 'expon', [0, params[0]])
    elif dist == "INVGAUSS":
        return stats.kstest(runtimes, 'invgauss', [params[0] / params[1], 0, params[1]])
    elif dist == "LOGNORMAL":
        return stats.kstest(runtimes, 'lognorm', [params[0], 0, params[1]])
    elif dist == "NORMAL":
        return stats.kstest(runtimes, 'norm', [params[0], params[1]])
    else:
        raise ValueError("Unknown dist: {}".format(dist))


def getNLLH(model_params, fold_runtimes, dist, net_type):
    nllhs = []
    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)

        temp = model_dist.pdf(instance)
        temp = [np.log(i) if i > 1e-6 else np.log(1e-4) for i in temp]
        nllh_per_instance = temp + np.log(max(instance))

        # nllh_per_instance = np.log(model_dist.pdf(instance) + 1e-8) + np.log(max(instance))
        nllhs.append(-np.mean(nllh_per_instance))
    return np.mean(nllhs)


def getKSTest(model_params, fold_runtimes, dist, net_type):
    p_counter = 0.0
    distance = 0.0
    ps = []
    distances = []
    for params, instance in zip(model_params, fold_runtimes):
        d, p = getDistKS(dist, params, instance, net_type)
        distance += d
        distances.append(d)
        ps.append(1 if p < 0.01 else 0)
        if p < 0.01:
            p_counter += 1
    return np.mean(ps), np.mean(distances)
    return p_counter / len(model_params), distance / len(model_params)


def getVariances(model_params, fold_runtimes, dist, net_type):
    variances = []
    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)
        samples = model_dist.resample(100) if net_type == "bayes_distnet" else model_dist.rvs(size=100)
        variances.append(np.var(samples, ddof=1))
    return np.mean(variances)


N_STEPS = 200
def getBhattacharyyaDistance(model_params, fold_runtimes, dist, net_type):
    distance = 0.0
    xs = np.linspace(0, 1.5, N_STEPS)
    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)
        reference_kde = stats.gaussian_kde(instance)

        delta = 1.0 / N_STEPS
        BC = sum([math.sqrt(reference_kde.pdf(x) * model_dist.pdf(x)) * delta for x in xs])
        distance += -np.log(BC + 1e-8)
    return distance / len(model_params)


def getKLD(model_params, fold_runtimes, dist, net_type):
    kld = 0.0
    xs = np.linspace(0, 1.5, N_STEPS)
    EPSILON = 1e-8
    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)
        reference_kde = stats.gaussian_kde(instance)

        delta = 1.0 / N_STEPS
        kld += sum([(reference_kde.pdf(x) * (np.log(reference_kde.pdf(x) + EPSILON) - np.log(model_dist.pdf(x) + EPSILON))) * delta for x in xs])
    return kld / len(model_params)


def getMass(model_params, fold_runtimes, dist, net_type):
    # Get the probability density outside the range [0,1]
    MIN_T = 0.0
    MAX_T = 1.0
    mass = 0.0
    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)
        MAX_T = max(instance) * 1.5

        if net_type == "bayes_distnet":
            inside_mass = model_dist.integrate_box_1d(MIN_T, MAX_T)
        else:
            inside_mass = model_dist.cdf(MAX_T) - model_dist.cdf(MIN_T)

        mass += 1.0 - inside_mass
    return mass / len(model_params)


def saveRuntimeDF(file_name, df):
    df.to_pickle('../export/dfs/' + file_name + '.pkl')


DF_COLUMNS = ['Scenario', 'Num Samples', 'LB', 'Seed', 'Fold', 'Mode', 'Model', 'LLH', 'NLLH', 'P-KS', 'D-KS', 'D-B', 'KLD', 'Var', 'Mass']
def getEmptyDataFrame():
    return pd.DataFrame(columns=DF_COLUMNS)


BASE_PATH = '../export/{}/'
PATH_FMT = '{}_{}_{}_{}_{}_{}_{}_{}.pkl'
def getScenarioModeDF(scenario, mode, lb, seed, fold, sample_count, model, test_data):
    # Runner for each process, gathers df for a single configuration of parameters and is joined later
    map_llh = {'EXPONENTIAL' : 'Exponential', 'INVGAUSS': 'InverseGaussian', 'LOGNORMAL' : 'Lognormal',
               'NORMAL': 'Normal', 'BAYESIAN_INVGAUSS': 'InverseGaussian', 'BAYESIAN_LOGNORMAL': 'Lognormal',
               'BAYESIAN_NORMAL': 'Normal', 'BAYESIAN_LOGNORMAL4': 'Lognormal4'}
    map_model = {'EXPONENTIAL' : 'Distnet', 'INVGAUSS': 'Distnet', 'LOGNORMAL' : 'Distnet',
                 'NORMAL': 'Distnet', 'BAYESIAN_INVGAUSS': 'BayesianDistnet', 'BAYESIAN_LOGNORMAL': 'BayesianDistnet',
                 'BAYESIAN_NORMAL': 'BayesianDistnet', 'BAYESIAN_LOGNORMAL4': 'BayesianDistnet'}

    df = getEmptyDataFrame()
    try:
        # Data/model import
        net_type = "bayes_distnet" if "BAYESIAN" in model else "distnet"
        BASE_PATH = '../export/{}/'.format(net_type)
        data_path = BASE_PATH.format(net_type) + PATH_FMT.format(net_type, scenario, model, sample_count, lb, fold, seed, mode)
        model_params = getDataPickleArray(data_path)

        # Get metrics
        nllh = getNLLH(model_params, test_data[fold], model, net_type)
        ks_p, ks_distance = getKSTest(model_params, test_data[fold], model, net_type)
        b_distance = getBhattacharyyaDistance(model_params, test_data[fold], model, net_type)
        kld = getKLD(model_params, test_data[fold], model, net_type)
        variances = getVariances(model_params, test_data[fold], model, net_type)
        masses = getMass(model_params, test_data[fold], model, net_type)

        # Export
        metric_row = (scenario, sample_count, lb, seed, fold, mode, map_model[model], map_llh[model],
                      nllh, ks_p, ks_distance, b_distance, kld, variances, masses)
        df = df.append(pd.Series(metric_row, index=df.columns), ignore_index=True)
    except Exception as e:
        print(e)
        pass

    return df


def createRuntimeDF(scenarios, modes, lbs, seeds, folds, sample_counts, models, num_proc=1):
    df = getEmptyDataFrame()

    for scenario, mode in product(scenarios, modes):
        runtimes, features = getScenarioData(scenario)
        test_data = getDataPerFold(runtimes, features, mode)

        pool = mp.Pool(num_proc)
        results = [pool.apply_async(getScenarioModeDF, args=(scenario, mode, lb, seed, fold, sample_count, model, test_data))
                   for lb, seed, fold, sample_count, model in product(lbs, seeds, folds, sample_counts, models)
                  ]

        pool.close()
        pool.join()
        results = [r.get() for r in results]

        df = pd.concat([df, *results])

    return df


if __name__ == "__main__":

    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="File name to safe to", required=True, type=str.lower)
    parser.add_argument("--num_proc", dest="num_proc", default=1, type=int)
    args = parser.parse_args()

    scenarios = ['clasp_factoring', 'saps-CVVAR', 'lpg-zeno', 'yalsat_qcp', 'spear_qcp', 'spear_swgcp', 'yalsat_swgcp']
    models = ['EXPONENTIAL', 'INVGAUSS', 'LOGNORMAL', 'NORMAL', 'BAYESIAN_INVGAUSS', 'BAYESIAN_LOGNORMAL']
    folds = range(10)
    sample_counts = [1, 2, 4, 8, 16]
    modes = ['test']
    seeds = [0, 1, 2, 3, 4]
    lbs = [0, 20, 40, 60, 80]

    df = createRuntimeDF(scenarios, modes, lbs, seeds, folds, sample_counts, models, num_proc=args.num_proc)
    saveRuntimeDF(args.file_name, df)

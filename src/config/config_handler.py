"""
File: config_handler.py
Date: May 10, 2020
Description: Handle model and training configurations
"""

# Library
import sys
import logging
import configparser
import traceback

# Module
from train.loss_functions import LossFunctionTypes


def getTrainingConfig(n_epochs=100, n_expected_epochs=1000, n_optim='SGD', batch_size=32,
                      loss_fn=LossFunctionTypes.MSE, start_rate=1e-3, end_rate=1e-5, clip_gradient_norm=1e-2,
                      split_ration=0.8, seed=-1, n_ens=20, beta_type=0.1):
    """
    Get the config dictionary which defines the specs for training

    Args:
        n_epochs (int): The number of epochs to train for
        batch_size (int): The number of samples in each batch
        loss_fn (LossFunctionTypes): The loss function type to use
        start_rate (float): The initial learning rate
        end_rate (float): The end learning rate
        clip_gradient_norm (float): The size of gradient before clipping takes place
        split_ration (float): Percentage of data to use for training
        seed (int): Seed to use, or -1 if no seed
        n_ens (int): Number of samples for variational inference approximation

    Returns:
        Dictionary representing configuration for the training
    """
    return {
        'n_epochs'             : n_epochs,
        'n_expected_epochs'    : n_expected_epochs,
        'n_optim'              : n_optim,
        'batch_size'           : batch_size,
        'loss_fn'              : loss_fn,
        'start_rate'           : start_rate,
        'end_rate'             : end_rate,
        'clip_gradient_norm'   : clip_gradient_norm,
        'split_ration'         : split_ration,
        'seed'                 : seed,
        'n_ens'                : n_ens,
        'beta_type'            : beta_type
    }


def trainingConfigToStr(config):
    """
    Convert a given training confiruation into a readable string

    Args:
        config (dict): A dictionary representing the configuration for the training
    """
    return "Training Config:\n" + \
           "\tNumber of Epochs: {}\n".format(config['n_epochs']) + \
           "\tExpected number of Epochs: {}\n".format(config['n_expected_epochs']) + \
           "\tOptimizer: {}\n".format(config['n_optim']) + \
           "\tBatch size: {}\n".format(config['batch_size']) + \
           "\tLoss function: {}\n".format(config['loss_fn'].name) + \
           "\tStart rate: {}\n".format(config['start_rate']) + \
           "\tEnd rate: {}\n".format(config['end_rate']) + \
           "\tGradient Clipping: {}\n".format(config['clip_gradient_norm']) + \
           "\tSplit ratio: {}\n".format(config['split_ration']) + \
           "\tSeed: {}\n".format(config['seed']) + \
           "\tSamples for Variational Inference: {}\n".format(config['n_ens']) + \
           "\tVariational Inference beta type: {}\n".format(config['beta_type'])


def getModelConfig(n_fcdepth=64, output_size=1, drop_value=0.15):
    """
    Get the config dictionary which defines the specs of the modles

    Args:
        n_fcdepth (int): The number of nodes in the first fully connected layer
        output_size (int): The number of outputs nodes for the network
        drop_value (float): Probability for dropout events

    Returns:
        Dictionary representing configuration for the DistNet model
    """
    return {
        'n_fcdepth'         : n_fcdepth,
        'output_size'       : output_size,
        'drop_value'        : drop_value,
    }


def modelConfigToStr(config):
    """
    Convert a given DistNet model confiruation into a readable string

    Args:
        config (dict): A dictionary representing the configuration for the DistNet model
    """
    return "Disnet Config:\n" + \
           "\tFirst FC layer size: {}\n".format(config['n_fcdepth']) + \
           "\tDropout p = {}\n".format(config['drop_value']) + \
           "\tOutput size: {}\n".format(config['output_size']) + \
           "\tposterior_mu1: {}".format(config['posterior_mu1']) + \
           "\tposterior_mu2: {}".format(config['posterior_mu2']) + \
           "\tposterior_sigma1: {}".format(config['posterior_sigma1']) + \
           "\tposterior_sigma2: {}".format(config['posterior_sigma2']) + \
           "\tmixture pi: {}".format(config['mixture_pi']) + \
           "\tprior sigma1: {}".format(config['prior_sigma1']) + \
           "\tprior sigma2: {}".format(config['prior_sigma2'])


def parseConfig(config_section):
    # Get logger/config
    logger = logging.getLogger()
    config = configparser.ConfigParser()
    config.read('config/training_config.ini')

    if config_section not in config.sections() and config_section != "DEFAULT":
        logger.error("Training config {} section was not found. See training_config.ini".format(config_section))
        sys.exit()

    # Start with default configs
    training_config = getTrainingConfig()
    model_config = getModelConfig()

    # Get items in the config option and update our training/model config
    try:
        # Training config items
        if config.has_option(config_section, 'n_epochs'):
            training_config['n_epochs'] = int(config.get(config_section, 'n_epochs'))
        if config.has_option(config_section, 'n_expected_epochs'):
            training_config['n_expected_epochs'] = int(config.get(config_section, 'n_expected_epochs'))
        if config.has_option(config_section, 'batch_size'):
            training_config['batch_size'] = int(config.get(config_section, 'batch_size'))
        if config.has_option(config_section, 'n_optim'):
            training_config['n_optim'] = config.get(config_section, 'n_optim')
        if config.has_option(config_section, 'loss_fn'):
            training_config['loss_fn'] = LossFunctionTypes[config.get(config_section, 'loss_fn')]
        if config.has_option(config_section, 'start_rate'):
            training_config['start_rate'] = float(config.get(config_section, 'start_rate'))
        if config.has_option(config_section, 'end_rate'):
            training_config['end_rate'] = float(config.get(config_section, 'end_rate'))
        if config.has_option(config_section, 'clip_gradient_norm'):
            training_config['clip_gradient_norm'] = None if config.get(config_section, 'clip_gradient_norm') == "None" \
                else float(config.get(config_section, 'clip_gradient_norm'))
        if config.has_option(config_section, 'split_ratio'):
            training_config['split_ratio'] = float(config.get(config_section, 'split_ratio'))
        if config.has_option(config_section, 'seed'):
            training_config['seed'] = int(config.get(config_section, 'seed'))
        if config.has_option(config_section, 'n_ens'):
            training_config['n_ens'] = int(config.get(config_section, 'n_ens'))
        if config.has_option(config_section, 'early_stop'):
            training_config['early_stop'] = int(config.get(config_section, 'early_stop'))
        if config.has_option(config_section, 'beta_type'):
            beta_type = config.get(config_section, 'beta_type')
            try:
                training_config['beta_type'] = float(beta_type)
            except ValueError:
                training_config['beta_type'] = beta_type

        # Model config items
        if config.has_option(config_section, 'n_fcdepth'):
            model_config['n_fcdepth'] = int(config.get(config_section, 'n_fcdepth'))
        if config.has_option(config_section, 'output_size'):
            model_config['output_size'] = int(config.get(config_section, 'output_size'))
        if config.has_option(config_section, 'drop_value'):
            model_config['drop_value'] = float(config.get(config_section, 'drop_value'))
        if config.has_option(config_section, 'posterior_mu1'):
            model_config['posterior_mu1'] = float(config.get(config_section, 'posterior_mu1'))
        if config.has_option(config_section, 'posterior_mu2'):
            model_config['posterior_mu2'] = float(config.get(config_section, 'posterior_mu2'))
        if config.has_option(config_section, 'posterior_sigma1'):
            model_config['posterior_sigma1'] = float(config.get(config_section, 'posterior_sigma1'))
        if config.has_option(config_section, 'posterior_sigma2'):
            model_config['posterior_sigma2'] = float(config.get(config_section, 'posterior_sigma2'))
        if config.has_option(config_section, 'mixture_pi'):
            model_config['mixture_pi'] = float(config.get(config_section, 'mixture_pi'))
        if config.has_option(config_section, 'prior_sigma1'):
            model_config['prior_sigma1'] = float(config.get(config_section, 'prior_sigma1'))
        if config.has_option(config_section, 'prior_sigma2'):
            model_config['prior_sigma2'] = float(config.get(config_section, 'prior_sigma2'))

    except Exception:
        logger.error("Unknown error parsing configuration.")
        logger.error(traceback.format_exc())
        sys.exit()

    return training_config, model_config

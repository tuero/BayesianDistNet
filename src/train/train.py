"""
File: train.py
Date: May 10, 2020
Description: Main training loop
"""

# Library
import logging

# Numerical
import numpy as np

# Pytorch
import torch
import torch.optim as optim

# Module
from config.config_handler import getModelConfig, getTrainingConfig, trainingConfigToStr, modelConfigToStr
from models.model_util import weights_init, getModel, count_parameters
from models.distnet import DistNetFCN
from models.bayes_distnet import BayesDistNetFCN
from train.loss_functions import getLossFunction, getNumberOfParameters
from train.metrics import get_beta
from helper import device
from helper.export import exportData, saveModel


def train_model(model, train_loader, optimizer, training_config, total_len, epoch, num_epochs):
    training_loss = 0.0
    model.train()
    loss_fn = getLossFunction(training_config['loss_fn'])
    n_ens = training_config['n_ens']
    clip_gradient_norm = training_config['clip_gradient_norm']
    beta_type = training_config['beta_type']
    total_kl = 0.0

    for batch_idx, data in enumerate(train_loader):
        # get the inputs; data is a list of [features, runtime]
        inputs, rts = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # Propogate input to model and calculate loss
        if type(model) is BayesDistNetFCN:
            kl = 0.0
            batch_size = inputs.shape[0]
            outputs = torch.zeros(batch_size, n_ens).to(device.device)
            for j in range(n_ens):
                net_out, _kl = model(inputs)
                kl += _kl
                outputs[:,j] = net_out.flatten()
            kl /= n_ens

            beta = get_beta(batch_idx, len(train_loader), beta_type, epoch, num_epochs)
            loss = (kl * beta) / len(train_loader) + loss_fn(outputs, rts, reduce=False).sum()
            total_kl += (kl * beta) / len(train_loader)
        elif type(model) is DistNetFCN:
            outputs = model(inputs)
            loss = loss_fn(outputs, rts)
        else:
            raise ValueError('Unknown net type.')

        # Propogate loss backwards and step optimizer
        loss.backward()

        # Clip the gradients
        if clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient_norm)
        # Step optimizer
        optimizer.step()

        training_loss += loss.item()

    return training_loss / len(train_loader), total_kl


def validate_model(model, validation_loader, training_config, val_len, epoch=1, num_epochs=1):
    loss_fn = getLossFunction(training_config['loss_fn'])
    n_ens = training_config['n_ens']
    beta_type = training_config['beta_type']

    # Bayesian can't be in eval mode so that we can sample and get a distribution
    model.eval()

    validation_loss = 0.0
    log_loss = 0.0
    total_kl = 0.0
    with torch.no_grad():
        for batch_idx, val_data in enumerate(validation_loader):
            val_inputs, val_rts = val_data

            if type(model) is BayesDistNetFCN:
                kl = 0.0
                batch_size = val_inputs.shape[0]
                outputs = torch.zeros(batch_size, n_ens).to(device.device)
                for j in range(n_ens):
                    net_out, _kl = model(val_inputs)
                    kl += _kl
                    outputs[:,j] = net_out.flatten()
                kl /= n_ens

                beta = get_beta(batch_idx, len(validation_loader), beta_type, epoch, num_epochs)
                loss = loss_fn(outputs, val_rts, reduce=True).sum()
                val_loss = (kl * beta) / len(validation_loader) + loss
                total_kl += (kl * beta) / len(validation_loader)

            elif type(model) is DistNetFCN:
                outputs = model(val_inputs)
                val_loss = loss_fn(outputs, val_rts)
                loss = val_loss
            else:
                raise ValueError('Unknown net type.')

            validation_loss += val_loss.item()
            log_loss += loss.item()

            if loss != loss:
                torch.set_printoptions(edgeitems=10000)
                logger = logging.getLogger()
                logger.info(log_loss)
                logger.info(outputs)
                logger.info(kl)
                exit()

    return validation_loss / len(validation_loader), log_loss / len(validation_loader), total_kl


def train(num_features, net_type, train_loader, validation_loader, X_test, model_path, data_path_test,
          training_config=getTrainingConfig(), model_config=getModelConfig()):
    """
    Train the model using the given training configuration

    Args:
        num_features (int): Number of features (passed to create model)
        net_type (str): Model string
        train_loader (torch.DataLoader) : Dataloader for training data
        validation_loader (torch.DataLoader) : Dataloader for validation data
        training_config (dict): Configuration for training
        model_config (dict): Configuration for the model

    Returns:
        DataFrame of loss data during training with testing every epoch
    """
    logger = logging.getLogger()
    logger.info('Using device {}'.format(device.device))

    # Set output size (loss function dependent) and create model
    model_config['output_size'] = getNumberOfParameters(training_config['loss_fn'])
    model = getModel(net_type, model_config, num_features)

    # Log the current configurations
    logger.info(trainingConfigToStr(training_config))
    logger.info(modelConfigToStr(model_config))
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    # Reset model and send model to device
    model.apply(weights_init)
    model.to(device.device)

    # Training config
    start_rate = training_config['start_rate']
    end_rate = training_config['end_rate']
    n_epochs = training_config['n_epochs']
    n_expected_epochs = training_config['n_expected_epochs']
    n_optim = training_config['n_optim']
    early_stop = training_config['early_stop']

    # Optimizer and scheduler
    if n_optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=start_rate, weight_decay=0.0001, momentum=0.95)
    elif n_optim == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=start_rate, weight_decay=0.001, amsgrad=False)
    else:
        raise ValueError('Unknown optimizer type.')
    decay_rate = np.exp(np.log(end_rate / start_rate) / n_expected_epochs)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
    logger.info('Decay rate: {}'.format(decay_rate))

    total_len = len(train_loader) * training_config['batch_size']
    counter = 1
    valid_loss_min = np.Inf
    valid_loss_counter = 0
    for epoch in range(n_epochs):
        # Next epoch for training
        train_loss, train_kl = train_model(model, train_loader, optimizer, training_config, total_len, epoch, n_epochs)
        val_loss, log_loss, val_kl = validate_model(model, validation_loader, training_config, total_len, epoch, n_epochs)

        output_msg = "Epoch: {:>4d}, Training Loss {:>12,.4f} Training KL {:>12,.4f}, Validation Loss {:>12,.4f}, Validation log {:>12,.4f}, Validation KL {:>12,.4f}"
        logger.info(output_msg.format(epoch + 1, train_loss, train_kl, val_loss, log_loss, val_kl))
        counter += 1

        if epoch % 10 == 0:
            exportData(X_test, data_path_test, model)
            saveModel(model, model_path)

        # Check for early stoppage
        # Uncomment for early stoppage
        if val_loss < valid_loss_min:
            valid_loss_counter = 0
            valid_loss_min = val_loss
        else:
            valid_loss_counter += 1

        if valid_loss_counter >= early_stop and type(model) is not BayesDistNetFCN:
            logger.info('Early stoppage, breaking')
            break

        # Update learning rate
        lr_scheduler.step()

    return model

"""
File: loss_functions.py
Date: April 4, 2020
Description: Implemented custom loss functions for log-likelihood
"""

# Library
import sys
from enum import IntEnum
import logging
import unittest
import random

# Numeric
import numpy as np
from scipy import stats

# Pytorch
import torch
import torch.nn as nn

# Module
from helper import device


# Constants
EPSILON = 1e-10
TWO_PI = 2 * np.pi
LOG_2PI = 1.837877066
LOG_2PI = np.log(TWO_PI)
SQRT_TWO = np.sqrt(2.0)
HALF = 0.5
E = 2.71828182845904523536028


# Loss function types
class LossFunctionTypes(IntEnum):
    MSE = 0
    EXPONENTIAL = 1
    INVERSE_GAUSSIAN = 2
    LOG_NORMAL = 3
    NORMAL = 4
    BAYESIAN_NORMAL = 5
    BAYESIAN_INVGAUSS = 6
    BAYESIAN_LOGNORMAL = 7
    BAYESIAN_EXPONENTIAL = 8


def getNumberOfParameters(loss_function_type):
    """
    Get the number of distribution parameters for the given loss function

    Args:
        loss_function_type (LossFunctionTypes): The loss function distribution type

    Returns
        Number of parameters representing the output layer for the network
    """
    logger = logging.getLogger()
    if loss_function_type == LossFunctionTypes.MSE:
        return 1
    if loss_function_type == LossFunctionTypes.EXPONENTIAL:
        return 1
    if loss_function_type == LossFunctionTypes.INVERSE_GAUSSIAN:
        return 2
    if loss_function_type == LossFunctionTypes.LOG_NORMAL:
        return 2
    if loss_function_type == LossFunctionTypes.NORMAL:
        return 2
    if loss_function_type == LossFunctionTypes.BAYESIAN_EXPONENTIAL:
        return 1
    if loss_function_type == LossFunctionTypes.BAYESIAN_NORMAL:
        return 1
    if loss_function_type == LossFunctionTypes.BAYESIAN_INVGAUSS:
        return 1
    if loss_function_type == LossFunctionTypes.BAYESIAN_LOGNORMAL:
        return 1

    # Othewise, we have an unknown loss function
    logger.error("Unknown loss function: {}".format(loss_function_type))
    sys.exit()


def getLossFunction(loss_function_type):
    """
    Get the callable function which will be used as the loss function during training

    Args:
        loss_function_type (LossFunctionTypes): The loss function distribution type

    Returns
        The callable function for the loss
    """
    logger = logging.getLogger()
    if loss_function_type == LossFunctionTypes.MSE:
        return nn.MSELoss()
    if loss_function_type == LossFunctionTypes.EXPONENTIAL:
        return expo_loss
    if loss_function_type == LossFunctionTypes.INVERSE_GAUSSIAN:
        return invgauss_loss
    if loss_function_type == LossFunctionTypes.LOG_NORMAL:
        return lognormal_loss
    if loss_function_type == LossFunctionTypes.NORMAL:
        return normal_loss
    if loss_function_type == LossFunctionTypes.BAYESIAN_EXPONENTIAL:
        return expo_bayesian
    if loss_function_type == LossFunctionTypes.BAYESIAN_NORMAL:
        return normal_bayesian
    if loss_function_type == LossFunctionTypes.BAYESIAN_LOGNORMAL:
        return lognorm_bayesian
    if loss_function_type == LossFunctionTypes.BAYESIAN_INVGAUSS:
        return invgauss_bayesian

    # Othewise, we have an unknown loss function
    logger.error("Unknown loss function: {}".format(loss_function_type))
    sys.exit()


def _approx_erf(x):
    """
    Approximation function for the Gauss Error Function.
    This uses one of the approximations provided by Abramowitz and Stegun

    Args:
        x (torch.Tensor): The input to the distribution
    """
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    # erf(x) = -erf(-x)
    neg_mask = x < 0
    torch.abs_(x)

    t = 1 / (1 + (p * x))
    polynomial = (a1 * t) + (a2 * (t**2)) + (a3 * (t**3)) + (a4 * (t**4)) + (a5 * (t**5))
    erf = 1 - (polynomial * torch.exp(- x**2))
    erf[neg_mask] *= -1
    return erf


def _standard_gaussian_cdf(x):
    """
    Approximation function for the Standard Guassian

    Args:
        x (torch.Tensor): The input to the distribution
    """
    return HALF * (1 + _approx_erf(x / SQRT_TWO))


def expo_loss(prediction, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Exponential distribution
    for the given sample

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    scale = prediction[:, 0:1] + EPSILON
    scale = 1.0 / scale
    target = observation[:, 0:1] + EPSILON
    sol_flag = observation[:, 1] == 1

    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]
    scale_1 = scale[sol_flag]
    scale_2 = scale[~sol_flag]

    llh = torch.zeros([prediction.shape[0]], dtype=torch.float32).to(device.device)

    # Observation seen i.e. use pointwise pdf
    llh[sol_flag] = torch.flatten(torch.log(scale_1) - (scale_1 * target_1))

    # Lowerbound i.e. use survival function = 1-CDF
    cdf = 1 - torch.exp(-scale_2 * target_2)
    llh[~sol_flag] = torch.flatten(torch.log(1 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def expo_scipy_loss(prediction, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Exponential distribution
    for the given sample

    Note:
        Calculates the log-likelihood using the scipy distribution

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    with torch.no_grad():
        scale = prediction[:, 0:1] + EPSILON
        scale = 1.0 / scale
        target = observation[:, 0] + EPSILON

        nll = []
        for i, (s, t) in enumerate(zip(scale, target)):
            nll.append((stats.expon.logpdf(t.item(), loc=0, scale=1.0 / s.item())))

    llh = torch.tensor([nll], dtype=torch.float32, requires_grad=True)
    return torch.mean(-llh) if reduce else -llh


def lognormal_loss(prediction, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Lognormal distribution
    for the given sample

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not

    Returns:
        A single dim tensor representing the mean log-likelihood of the observation
    """
    sigma = prediction[:, 0:1] + EPSILON
    mu = prediction[:, 1:2] + EPSILON
    target = observation[:, 0:1] + EPSILON
    sol_flag = observation[:, 1] == 1

    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    sigma_1 = sigma[sol_flag]
    sigma_2 = sigma[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]

    llh = torch.zeros([prediction.shape[0]], dtype=torch.float32).to(device.device)

    # Observation seen i.e. use pointwise pdf
    pdf_help1 = HALF * ((torch.log(target_1) - torch.log(mu_1)) / sigma_1)**2
    llh[sol_flag] = torch.flatten((-torch.log(target_1) - torch.log(sigma_1)  - pdf_help1))

    # Lowerbound i.e. use survival function = 1-CDF
    cdf_help1 = (torch.log(target_2) - torch.log(mu_2)) / (SQRT_TWO * sigma_2)
    cdf = HALF + (HALF * _approx_erf(cdf_help1))
    cdf = torch.clamp(torch.clamp(cdf, max=1), min=0)
    llh[~sol_flag] = torch.flatten(torch.log(1 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def lognormal_scipy_loss(prediction, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Lognormal distribution
    for the given sample

    Note:
        Calculates the log-likelihood using the scipy distribution

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    with torch.no_grad():
        mu = prediction[:, 1] + EPSILON
        sigma = prediction[:, 0] + EPSILON
        target = observation[:, 0] + EPSILON

        nll = []
        for i, (m, s, t) in enumerate(zip(mu, sigma, target)):
            nll.append((stats.lognorm.logpdf(t.item(), s.item(), loc=0, scale=m.item())))

    llh = torch.tensor([nll], dtype=torch.float32, requires_grad=True)
    return torch.mean(-llh) if reduce else -llh


def _pdf_invgauss(x, mu, lambda_):
    helper = -(lambda_ * (x - mu)**2) / (2 * x * (mu**2))
    return (torch.sqrt(lambda_) / torch.sqrt(TWO_PI * (x**3))) * torch.exp(helper)


def invgauss_loss(prediction, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Inverse Gaussian distribution
    for the given sample

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    mu = prediction[:, 0] + EPSILON
    lambda_ = prediction[:, 1] + EPSILON
    target = observation[:, 0] + EPSILON
    sol_flag = observation[:, 1] == 1

    lambda_1 = lambda_[sol_flag]
    lambda_2 = lambda_[~sol_flag].unsqueeze(-1)
    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag].unsqueeze(-1)
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag].unsqueeze(-1)

    llh = torch.zeros([prediction.shape[0]], dtype=torch.float32).to(device.device)

    # Observation seen i.e. use pointwise pdf
    pdf_help1 = (lambda_1 * (target_1 - mu_1)**2) / (2 * target_1 * (mu_1**2))
    llh[sol_flag] = torch.flatten(((HALF * torch.log(lambda_1)) - (3.0 * HALF * torch.log(target_1)) - pdf_help1))

    # Lowerbound i.e. use survival function = 1-CDF
    # Approximation methods needed
    STEPS = 100
    if target_2.nelement() != 0:
        xs = torch.stack([torch.arange(1, STEPS + 1) / float(STEPS) * i.item() for i in target_2]).to(device.device)
        pdfs = _pdf_invgauss(xs, mu_2, lambda_2)
        # CDF needs to be clamped within [0,1] or NaNs get propogated
        cdfs = torch.clamp(torch.clamp(torch.trapz(pdfs, xs), max=1), min=0)
        llh[~sol_flag] = torch.flatten(torch.log(1 - cdfs + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def invgauss_scipy_loss(prediction, target, reduce=True):
    """
    Calculates the mean log-likelihood following the Inverse Gaussian distribution
    for the given sample

    Note:
        Calculates the log-likelihood using the scipy distribution

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        target (torch.Tensor): The target as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    with torch.no_grad():
        mu = prediction[:, 0] + EPSILON
        lambda_ = prediction[:, 1] + EPSILON
        target = target[:, 0] + EPSILON

        nll = []
        for i, (m, s, t) in enumerate(zip(mu, lambda_, target)):
            nll.append((stats.invgauss.logpdf(t.item(), m.item() / s.item(), loc=0, scale=s.item())))

    llh = torch.tensor([nll], dtype=torch.float32, requires_grad=True)
    return torch.mean(-llh) if reduce else -llh


def _pdf_normal(x, mu, sigma):
    helper = HALF * ((x - mu) / sigma)**2
    return 1.0 / (sigma * np.sqrt(TWO_PI)) * torch.exp(-helper)


def normal_loss(prediction, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Normal distribution
    for the given sample

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    mu = prediction[:, 0:1]
    sigma = prediction[:, 1:2] + EPSILON
    target = observation[:, 0:1]
    sol_flag = observation[:, 1] == 1

    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]
    sigma_1 = sigma[sol_flag]
    sigma_2 = sigma[~sol_flag]

    llh = torch.zeros([prediction.shape[0]], dtype=torch.float32).to(device.device)

    # Observation seen i.e. use pointwise pdf
    pdf_help = HALF * ((target_1 - mu_1) / sigma_1)**2
    llh[sol_flag] = torch.flatten(- torch.log(sigma_1) - pdf_help)

    # Lowerbound i.e. use survival function = 1-CDF
    cdf = _standard_gaussian_cdf((target_2 - mu_2) / sigma_2)
    cdf = torch.clamp(torch.clamp(cdf, max=1), min=0)
    llh[~sol_flag] = torch.flatten(torch.log(1.0 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def expo_bayesian(outputs, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Exponential distribution
    for the given sample

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    # MLE of sample for each instance
    scale = torch.reciprocal(outputs.mean(dim=1, keepdim=True))
    target = observation[:, 0:1]
    sol_flag = observation[:, 1] == 1

    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]
    scale_1 = scale[sol_flag]
    scale_2 = scale[~sol_flag]

    llh = torch.zeros([scale.shape[0]], dtype=torch.float32).to(device.device)

    # Observation seen i.e. use pointwise pdf
    llh[sol_flag] = torch.flatten(torch.log(scale_1) - (scale_1 * target_1))

    # Lowerbound i.e. use survival function = 1-CDF
    cdf = 1 - torch.exp(-scale_2 * target_2)
    llh[~sol_flag] = torch.flatten(torch.log(1 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def normal_bayesian(outputs, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Normal distribution
    for the given sample

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    # MLE of sample for each instance
    outputs += 1e-10
    mu = outputs.mean(dim=1, keepdim=True)
    sigma = outputs.std(dim=1, keepdim=True)
    target = observation[:, 0:1]
    sol_flag = observation[:, 1] == 1

    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]
    sigma_1 = sigma[sol_flag]
    sigma_2 = sigma[~sol_flag]

    llh = torch.zeros([mu.shape[0]], dtype=torch.float32).to(device.device)

    # Observation seen i.e. use pointwise pdf
    pdf_help = HALF * ((target_1 - mu_1) / sigma_1)**2
    llh[sol_flag] = torch.flatten(- torch.log(sigma_1) - pdf_help)

    # Lowerbound i.e. use survival function = 1-CDF
    cdf = _standard_gaussian_cdf((target_2 - mu_2) / sigma_2)
    cdf = torch.clamp(torch.clamp(cdf, max=1), min=0)
    llh[~sol_flag] = torch.flatten(torch.log(1.0 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def lognorm_bayesian(outputs, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Lognormal distribution
    for the given sample

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    # MLE of sample for each instance
    flag = outputs.sum(dim=1) > 1e-8
    outputs = outputs[flag]
    observation = observation[flag]
    outputs += 1e-10
    mu = torch.exp(torch.log(outputs).mean(dim=1, keepdim=True))
    sigma = torch.log(outputs).std(dim=1, keepdim=True)
    target = observation[:, 0:1] + EPSILON
    sol_flag = observation[:, 1] == 1

    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    sigma_1 = sigma[sol_flag]
    sigma_2 = sigma[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]

    llh = torch.zeros([outputs.shape[0]], dtype=torch.float32).to(device.device)

    # Observation seen i.e. use pointwise pdf
    pdf_help1 = HALF * ((torch.log(target_1) - torch.log(mu_1)) / sigma_1)**2
    llh[sol_flag] = torch.flatten((-torch.log(target_1) - torch.log(sigma_1) - pdf_help1))

    # Lowerbound i.e. use survival function = 1-CDF
    cdf_help1 = (torch.log(target_2) - torch.log(mu_2)) / (SQRT_TWO * sigma_2)
    cdf = HALF + (HALF * _approx_erf(cdf_help1))
    cdf = torch.clamp(torch.clamp(cdf, max=1), min=0)
    llh[~sol_flag] = torch.flatten(torch.log(1 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def invgauss_bayesian(outputs, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Inverse Gaussian distribution
    for the given sample

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """
    # MLE of sample for each instance
    flag = outputs.sum(dim=1) > 1e-8
    outputs = outputs[flag]
    observation = observation[flag]
    outputs += 1e-10
    
    mu = outputs.mean(dim=1, keepdim=True)
    temp_lambda = (1.0 / outputs) - (1.0 / mu)
    lambda_ = (outputs.shape[1]) / (temp_lambda.sum(dim=1, keepdim=True))
    target = observation[:, 0:1] + EPSILON
    sol_flag = observation[:, 1] == 1

    lambda_1 = lambda_[sol_flag]
    lambda_2 = lambda_[~sol_flag]
    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag].unsqueeze(-1)

    llh = torch.zeros([outputs.shape[0]], dtype=torch.float32).to(device.device)

    # Observation seen i.e. use pointwise pdf
    pdf_help1 = (lambda_1 * (target_1 - mu_1)**2) / (2 * target_1 * (mu_1**2))
    llh[sol_flag] = torch.flatten(((HALF * torch.log(lambda_1)) - (3.0 * HALF * torch.log(target_1)) - pdf_help1))

    # Lowerbound i.e. use survival function = 1-CDF
    # Approximation methods needed
    STEPS = 100
    if target_2.nelement() != 0:
        xs = torch.stack([torch.arange(1, STEPS + 1) / float(STEPS) * i.item() for i in target_2]).to(device.device)
        pdfs = _pdf_invgauss(xs, mu_2, lambda_2)
        # CDF needs to be clamped within [0,1] or NaNs get propogated
        cdfs = torch.clamp(torch.clamp(torch.trapz(pdfs, xs), max=1), min=0)
        llh[~sol_flag] = torch.flatten(torch.log(1 - cdfs + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def normal_scipy_loss(prediction, observation, reduce=True):
    """
    Calculates the mean log-likelihood following the Normal distribution
    for the given sample

    Note:
        Calculates the log-likelihood using the scipy distribution

    Args:
        prediction (torch.Tensor): The predicted distribution parameters
        observation (torch.Tensor): The observation as an input to the distribution
        reduce (bool): Whether to reduce to mean or not
    """

    with torch.no_grad():
        mu = prediction[:, 0] + EPSILON
        sigma = prediction[:, 1] + EPSILON
        target = observation[:, 0] + EPSILON

        nll = []
        for i, (m, s, t) in enumerate(zip(mu, sigma, target)):
            nll.append((stats.norm.logpdf(t.item(), loc=m.item(), scale=s.item())))

    llh = torch.tensor([nll], dtype=torch.float32, requires_grad=True)
    return torch.mean(-llh) if reduce else -llh


class TestLossFunctions(unittest.TestCase):

    def testLognormalLoss(self):
        DELTA = 1e-3
        OFFSET = -0.5 * LOG_2PI

        mus = torch.Tensor([[1.0, 1.0, 1.0, 0.75, 0.75, 0.75]])
        sigmas = torch.Tensor([[0.5, 0.5, 0.5, 3.0, 3.0, 3.0]])
        params = torch.t(torch.cat((sigmas, torch.exp(mus)), 0)).to(device.device)
        targets = torch.Tensor([[1.0, 2.0, 8.0, 0.5, 10.0, 20.0]])
        mask = torch.Tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        observations = torch.t(torch.cat((targets, mask), 0)).to(device.device)
        nllh = torch.Tensor([2.22579073241 + OFFSET, 1.10725452544 + OFFSET, 4.6356211461 + OFFSET,
                             0.378688327, 1.196024476, 1.48255859])

        results = lognormal_loss(params, observations, False)

        for n, r in zip(nllh, results):
            self.assertTrue(abs(n - r) < DELTA)

        # Check random parameters between the various implementations
        for _ in range(100):
            mu = random.uniform(2.0, 3.0)
            theta = random.uniform(1.0, 2.0)
            target = random.uniform(0.0, 20.0)
            params = torch.Tensor([[mu, theta]]).to(device.device)
            observations = torch.Tensor([[target, 1]]).to(device.device)
            l1 = lognormal_loss(params, observations).data.cpu().numpy()
            l2 = lognormal_scipy_loss(params.cpu(), observations.cpu()).data.cpu().numpy()
            self.assertTrue(abs(l1 - (l2 + OFFSET)) < DELTA)

    def testInverseGaussianLoss(self):
        DELTA = 1e-3
        OFFSET = -0.5 * LOG_2PI

        mus = torch.Tensor([[1.0, 1.0, 1.0, 0.75, 0.75, 0.75]])
        lambdas = torch.Tensor([[0.5, 0.5, 0.5, 3.0, 3.0, 3.0]])
        params = torch.t(torch.cat((mus, lambdas), 0)).to(device.device)
        targets = torch.Tensor([[0.1, 0.3, 1, 0.3, 0.5, 1.5]])
        mask = torch.Tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        observations = torch.t(torch.cat((targets, mask), 0)).to(device.device)
        nllh = torch.Tensor([-0.163368068876 + OFFSET, -0.1321153906 + OFFSET, 1.2655113853 + OFFSET,
                             0.0440765557, 0.31954294910, 3.085131954818])

        results = invgauss_loss(params, observations, False)

        for n, r in zip(nllh, results):
            self.assertTrue(abs(n - r) < DELTA)

        for _ in range(100):
            mu = random.uniform(2.0, 3.0)
            lambda_ = random.uniform(1.0, 2.0)
            target = random.uniform(0.0, 20.0)
            params = torch.Tensor([[mu, lambda_]]).to(device.device)
            observations = torch.Tensor([[target, 1]]).to(device.device)
            l1 = invgauss_loss(params, observations).data.cpu().numpy()
            l2 = invgauss_scipy_loss(params.cpu(), observations.cpu()).data.cpu().numpy()
            self.assertTrue(abs(l1 - (l2 + OFFSET)) < DELTA)

    def testNormalLoss(self):
        DELTA = 1e-3
        OFFSET = -0.5 * LOG_2PI

        mus = torch.Tensor([[1.0, 1.0, 1.0, 0.75, 0.75, 0.75]])
        sigma = torch.Tensor([[0.5, 0.5, 0.5, 3.0, 3.0, 3.0]])
        params = torch.t(torch.cat((mus, sigma), 0)).to(device.device)
        targets = torch.Tensor([[0.1, 0.3, 1, 0.3, 0.5, 1.5]])
        mask = torch.Tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        observations = torch.t(torch.cat((targets, mask), 0)).to(device.device)
        nllh = torch.Tensor([1.845793357720 + OFFSET, 1.2057911231327 + OFFSET, 0.22579080219 + OFFSET, 
                             0.58050087087, 0.628845562, 0.9130609532])

        results = normal_loss(params, observations, False)

        for n, r in zip(nllh, results):
            self.assertTrue(abs(n - r) < DELTA)

        for _ in range(100):
            mu = random.uniform(0.0, 3.0)
            sigma_squared = random.uniform(1.0, 2.0)
            target = random.uniform(0.0, 20.0)
            params = torch.Tensor([[mu, sigma_squared]]).to(device.device)
            observations = torch.Tensor([[target, 1]]).to(device.device)
            l1 = normal_loss(params, observations).data.cpu().numpy()
            l2 = normal_scipy_loss(params.cpu(), observations.cpu()).data.cpu().numpy()
            self.assertTrue(abs(l1 - (l2 + OFFSET)) < DELTA)

    def testExponentialLoss(self):
        DELTA = 1e-3

        lambda_ = torch.Tensor([[1.0, 1.0, 1.0, 0.75, 0.75, 0.75]])
        params = torch.t(1.0 / lambda_).to(device.device)
        targets = torch.Tensor([[0.1, 0.3, 1, 0.3, 0.5, 1.5]])
        mask = torch.Tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        observations = torch.t(torch.cat((targets, mask), 0)).to(device.device)
        nllh = torch.Tensor([0.100000462, 0.30000029, 1, 0.22500027, 0.375000405, 1.125001439])

        results = expo_loss(params, observations, False)

        for n, r in zip(nllh, results):
            self.assertTrue(abs(n - r) < DELTA)

        for _ in range(100):
            scale = random.uniform(0.0, 1.0)
            target = random.uniform(0.0, 20.0)
            params = torch.Tensor([[scale]]).to(device.device)
            observations = torch.Tensor([[target, 1]]).to(device.device)
            l1 = expo_loss(params, observations).data.cpu().numpy()
            l2 = expo_scipy_loss(params.cpu(), observations.cpu()).data.cpu().numpy()
            self.assertTrue(abs(l1 - l2) < DELTA)

    def testApproxErf(self):
        DELTA = 1e-5
        x = torch.Tensor([0.0, 0.1234, -0.1234, 1.0, -1.0, 2.12, -2.12])
        erf = torch.Tensor([0.0, 0.13853800, -0.13853800, 0.84270079, -0.84270079, 0.99728361, -0.99728361])
        result = _approx_erf(x)

        for e, r in zip(erf, result):
            self.assertTrue(abs(e - r) < DELTA)

    def testStandardGaussian(self):
        DELTA = 1e-5
        x = torch.Tensor([0.0, 0.1234, -0.1234, 1.0, -1.0, 2.12, -2.12])
        stg = torch.Tensor([0.5, 0.549105, 0.450895, 0.8413447461, 0.15865525393, 0.982997, 0.017003])
        result = _standard_gaussian_cdf(x)

        for s, r in zip(stg, result):
            self.assertTrue(abs(s - r) < DELTA)


if __name__ == "__main__":
    unittest.main()

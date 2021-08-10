"""
File: BBBLinear.py
Date: April 4, 2020
Description: Bayesian Linear Layer Implementation
Source: Taken from https://github.com/kumar-shridhar/PyTorch-BayesianCNN
"""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

import numpy as np

from layers.misc import ModuleWrapper
from train.metrics import calculate_kl

import sys
sys.path.append("..")

import helper.device as device


class BBBLinear(ModuleWrapper):
    
    def __init__(self, in_features, out_features, model_config, bias=False, name='BBBLinear'):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.name = name

        self.prior_mu = 0.0
        self.prior_sigma = 0.1
        self.prior_sigma1 = model_config['prior_sigma1']
        self.prior_sigma2 = model_config['prior_sigma2']
        self.pi = model_config['mixture_pi']

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))

        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        self.posterior_mu1 = model_config['posterior_mu1']
        self.posterior_mu2 = model_config['posterior_mu2']
        self.posterior_sigma1 = model_config['posterior_sigma1']
        self.posterior_sigma2 = model_config['posterior_sigma2']

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(self.posterior_mu1, self.posterior_mu2)
        self.W_rho.data.normal_(self.posterior_sigma1, self.posterior_sigma2)

        if self.use_bias:
            self.bias_mu.data.normal_(self.posterior_mu1, self.posterior_mu2)
            self.bias_rho.data.normal_(self.posterior_sigma1, self.posterior_sigma2)

    def gaussian_prior(self, x, mu, sigma):
        PI = 3.1415926535897
        scaling = 1.0 / np.sqrt(2.0 * PI * (sigma ** 2))
        bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
        return scaling * bell

    def gaussian(self, x, mu, sigma):
        PI = 3.1415926535897
        scaling = 1.0 / torch.sqrt(2.0 * PI * (sigma ** 2))
        bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
        return scaling * bell

    def log_prior_sum(self, x):
        first_gaussian = self.pi * self.gaussian_prior(x, self.prior_mu, self.prior_sigma1)
        second_gaussian = (1 - self.pi) * self.gaussian_prior(x, self.prior_mu, self.prior_sigma2)
        return torch.log(first_gaussian + second_gaussian).sum()

    def log_variational_sum(self, x):
        return torch.log(self.gaussian(x, self.W_mu, self.W_sigma)).sum()

    def forward(self, x):
        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(device.device)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        self.weights = self.W_mu + W_eps * self.W_sigma

        if self.use_bias:
            bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(device.device)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            bias = None

        return F.linear(x, self.weights, bias)

    def kl_loss(self):
        kl = self.log_variational_sum(self.weights) - self.log_prior_sum(self.weights)
        return kl

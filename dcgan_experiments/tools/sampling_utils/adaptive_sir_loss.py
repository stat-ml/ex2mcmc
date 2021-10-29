import os
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (MultivariateNormal, 
                                 Normal, 
                                 Independent, 
                                 Uniform)

from distributions import (Target, 
                           Gaussian_mixture, 
                           IndependentNormal,
                           init_independent_normal)

import pdb
import torchvision
from scipy.stats import gamma, invgamma

import pyro
from pyro.infer import MCMC, NUTS, HMC

from functools import partial
from tqdm import tqdm, trange
from easydict import EasyDict as edict
import copy

from flows import RNVP



def get_optimizer(parameters, optimizer = "Adam", lr = 1e-3, weight_decay=1e-5):
    if optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError


def get_loss(loss):
    if loss == "mix_kl":
        return mix_kl
    if loss == "forward_kl":
        return forward_kl
    if loss == "backward_kl":
        return backward_kl
    else:
        raise NotImplementedError

###Write here f divergence


#write here forward/backward KL

def forward_kl(target, proposal, flow, y):
    ##Here, y \sim \target
    ###PROPOSAL INITIAL DISTR HERE
    y_ = y.detach().requires_grad_()
    u, log_jac = flow.inverse(y_)
    est = target(y) - proposal.log_prob(u) - log_jac
    grad_est = - proposal.log_prob(u) - log_jac
    return est.mean(), grad_est.mean()



def backward_kl(target, proposal, flow, y):
    u = proposal.sample(y.shape[:-1])
    x, log_jac = flow(u)
    est = proposal.log_prob(u) - log_jac - target(x)
    grad_est = - log_jac - target(x)
    return est.mean(), grad_est.mean()

def mix_kl(target, proposal, flow, y, alpha= .99):
    est_f, grad_est_f = forward_kl(target, proposal, flow, y)
    est_b, grad_est_b = backward_kl(target, proposal, flow, y)
    return alpha * est_f + (1. - alpha) * est_b, alpha * grad_est_f + (1. - alpha) * grad_est_b 






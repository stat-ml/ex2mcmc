import torch
from torch.distributions import categorical, normal
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

class Model(ABC):
    @abstractmethod
    def sample(self, n: int):
        return NotImplementedError

    @abstractmethod
    def log_likelihood(self, ys, zs):
        return NotImplementedError

    @abstractmethod
    def log_prior(self, theta, zs):
        return NotImplementedError


class Clustering(Model):
    def __init__(self, pi, sigma_theta, sigma_n, dim):
        super().__init__()
        self.dim = dim
        self.sigma_theta = sigma_theta
        self.sigma_n = sigma_n

        if isinstance(pi, (List, np.ndarray)):
            pi = torch.FloatTensor(pi)
        if isinstance(pi, (torch.Tensor,)):
            pi = categorical.Categorical(probs=pi)
        self.k = pi.param_shape[0]
        self.pi = pi
        self.theta_d = normal.Normal(loc=torch.zeros(1), scale=torch.ones(1)*sigma_theta)
        self.theta = self.theta_d.sample(sample_shape=torch.Size([self.k, dim]))
        self.log_pi_theta = self.theta_d.log_prob(self.theta).sum(1)
        self.y_ds = [normal.Normal(loc=theta_, scale=torch.ones(1)*sigma_n) for theta_ in self.theta]

    def sample(self, n: int) -> Tuple:
        zs = self.pi.sample(sample_shape=torch.Size([n]))
        cnt = torch.bincount(torch.cat([zs, torch.LongTensor([0, 1, 2])], 0)) - 1
        ys = []
        for i, c in enumerate(cnt):
            if c.item() > 0:
                y = self.y_ds[i].sample(sample_shape=torch.Size([c]))
                ys.append(y)
        ys = torch.cat(ys, dim=0)
        perm = torch.randperm(ys.shape[0])
        ys = ys[perm]
        zs = zs[perm]
        return (ys, zs)

    def log_likelihood(self, ys: torch.FloatTensor, zs: torch.LongTensor):
        log_l = 0
        for k in range(self.k):
            ids = (zs == k).nonzero(as_tuple=True)[0]
            y = ys[ids]
            log_pi_y = self.y_ds[k].log_prob(y).sum(1).sum(0)
            if k == 0:
                log_l = log_pi_y
            log_l += log_pi_y
        return log_l

    def log_prior(self, theta: torch.FloatTensor, zs: torch.LongTensor):
        #print(theta.shape, zs.shape)
        pi_z = self.pi.probs[zs].sum()
        log_p = pi_z
        log_pi_theta = self.theta_d.log_prob(theta).sum(1)
        bins = torch.bincount(torch.cat([zs, torch.LongTensor([0, 1, 2])], 0)) - 1
        log_p += (log_pi_theta * bins[:, None]).sum(0)[0]
        return log_p


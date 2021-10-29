import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

from general_utils import DotDict
from logistic_regression import import_covertype


torchType = torch.float32
class Target(nn.Module):
    """
    Base class for a custom target distribution
    """

    def __init__(self, kwargs):
        super().__init__()
        self.device = kwargs.device
        self.torchType = torchType
        self.device_zero = torch.tensor(0., dtype=self.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=self.torchType, device=self.device)

    def prob(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def log_prob(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError
        
    def energy(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        energy = -log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def sample(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

class Gaussian_mixture(Target):
    """
    Mixture of n gaussians (multivariate)
    """

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.num = kwargs['num_gauss']
        self.pis = kwargs['p_gaussians']
        self.locs = kwargs['locs']  # list of locations for each of these gaussians
        self.covs = kwargs['covs']  # list of covariance matrices for each of these gaussians
        self.peak = [None] * self.num
        for i in range(self.num):
            self.peak[i] = torch.distributions.MultivariateNormal(loc=self.locs[i], covariance_matrix=self.covs[i])

    def get_density(self, x):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.log_prob(x).exp()
        return density

    def log_prob(self, z, x=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent variable
        Output:
        log_density - log p(x)
        """
        log_p = torch.tensor([], device=self.device)
        #pdb.set_trace()
        for i in range(self.num):
            log_paux = (torch.log(self.pis[i]) + self.peak[i].log_prob(z.to(self.device))).view(-1, 1)
            log_p = torch.cat([log_p, log_paux], dim=-1)
        log_density = torch.logsumexp(log_p, dim=1) 
        return log_density
        
    def energy(self, z, x=None):
        return -self.log_prob(z, x)
        
class IndependentNormal(Target):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.loc = kwargs['loc']  
        self.scale = kwargs['scale']  
        self.distribution = torch.distributions.Normal(loc=self.loc, scale=self.scale)

    def get_density(self, x):
        density = self.log_prob(x).exp()
        return density

    def log_prob(self, z, x=None):
        log_density = (self.distribution.log_prob(z.to(self.device))).sum(dim=-1)
        return log_density
    
    def sample(self, n):
        return self.distribution.sample(n)
        
    def energy(self, z, x=None):
        return -self.log_prob(z, x)
        
def init_independent_normal(scale, n_dim, device, loc = 0.0):
    loc = loc*torch.ones(n_dim).to(device)
    scale = scale*torch.ones(n_dim).to(device)
    target_args = DotDict()
    target_args.device = device
    target_args.loc = loc
    target_args.scale = scale
    target = IndependentNormal(target_args)
    return target

def init_independent_normal_scale(scales, locs, device):
    target_args = DotDict()
    target_args.device = device
    target_args.loc = locs.to(device)
    target_args.scale = scales.to(device)
    target = IndependentNormal(target_args)
    return target



class Cauchy(Target):
    
    def __init__(self, kwargs):
        
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.loc  = kwargs['loc']
        self.scale = kwargs['scale']
        self.dim   = kwargs['dim']
        self.distr = torch.distributions.Cauchy(self.loc, self.scale)
        
    def __call__(self,x):
        
        return self
        
    def get_density(self,x):
        
        return self.log_prob(x).exp()
    
    def log_prob(self, z, x = None):
        
        log_target = torch.sum(self.distr.log_prob(z), -1)
        
        return log_target
    
    def sample(self, n = (1,)):
        
        return self.distr.sample((n[0], self.dim))
    
    
class Cauchy_mixture(Target):
    
    def __init__(self, kwargs):
        
        super().__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.mu  = kwargs['mu']
        self.cov = kwargs['cov']
        
    def get_density(self,x):
        
        return self.log_prob(x).exp()
    
    def log_prob(self, z, x = None):

        cauchy = Cauchy(self.mu, self.cov)
        cauchy_minus = Cauchy(-self.mu, self.cov)

        catted = torch.cat([cauchy.log_prob(z)[None,...],cauchy_minus.log_prob(z)[None,...]],0)

        log_target = torch.sum(torch.logsumexp(catted, 0) - torch.tensor(2.).log(),-1)
        
        return log_target + torch.tensor([8.]).log()


class Funnel(Target):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.a = kwargs.get('a', 1.) * torch.ones(1)
        self.b = kwargs.get('b', .5)
        self.dim = kwargs.get('dim', 16)
        self.distr1 = torch.distributions.Normal(torch.zeros(1), self.a)
        #self.distr2 = lambda z1: torch.distributions.MultivariateNormal(torch.zeros(self.dim-1), (2*self.b*z1).exp()*torch.eye(self.dim-1))
        #self.distr2 = lambda z1: -(z[...,1:]**2).sum(-1) * (-2*self.b*z1).exp() - np.log(self.dim) + 2*self.b*z1 
        
    def log_prob(self, z, x=None):
        #pdb.set_trace()
        logprob1 = self.distr1.log_prob(z[...,0])
        z1 = z[..., 0]
        #logprob2 = self.distr2(z[...,0])
        logprob2 = -(z[...,1:]**2).sum(-1) * (-2*self.b*z1).exp() - np.log(self.dim) + 2*self.b*z1 
        return logprob1+logprob2
    
    
class Banana(Target):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.Q = kwargs.get('Q', .01) * torch.ones(1)
        self.dim = kwargs.get('dim', 32)
        #assert self.dim % 2 == 0, 'Dimension should be divisible by 2'
        
    def log_prob(self, z, x=None):
        #n = self.dim/2
        even = np.arange(0, self.dim, 2)
        odd = np.arange(1, self.dim, 2)
        
        ll = - (z[..., even] - z[..., odd]**2)**2/self.Q - (z[..., odd]-1)**2   
        return ll.sum(-1)

    
    
class Covtype_logistic_regression(Target):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.c1 = kwargs.get('c1', 1)
        self.c2 = kwargs.get('c2', 2)
        self.n_data = kwargs.get('n_data', 500000)
        x1_train, x1_test, y1_train, y1_test = import_covertype(self.c1, 
                                                                self.c2, 
                                                                self.n_data)
        self.x_train = torch.tensor(x1_train, dtype=self.torchType, device=self.device)
        self.x_test =  torch.tensor(x1_test , dtype=self.torchType, device=self.device)
        self.y_train = torch.tensor(y1_train, dtype=self.torchType, device=self.device)
        self.y_test =  torch.tensor(y1_test , dtype=self.torchType, device=self.device)
        self.d = np.size(x1_train,1)
        self.n = np.size(y1_train)
        self.tau = kwargs.get('tau', 1)
        
        
    def log_prob(self, theta, x= None, y = None):
        if x is None:
            x = self.x_train
        if y is None:
            y = self.y_train
        P = 1. / (1. + torch.exp(-torch.matmul(x,theta.transpose(0,1))))
        ll =  torch.matmul(y,torch.log(P)) + torch.matmul(1-y,torch.log(1-P)) - self.tau/2 * (theta**2).sum(-1)
        return ll
        
    def grad_log_prob(self, theta, x= None, y = None):
        if x is None:
            x = self.x_train
        if y is None:
            y = self.y_train
        P = 1. / (1. + torch.exp(-torch.mm(x,theta)))
        return torch.matmul(torch.transpose(X, -2, -1), (y - P).view(self.n)) - self.tau * theta

from __future__ import print_function

import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import grad as torchgrad
import hmc
import utils


def ais_trajectory(model,
                   theta,
                   zs,
                   loader,
                   log_likelihood_fn,
                   log_prior_fn,
                   forward=True,
                   schedule=np.linspace(0., 1., 500),
                   n_sample=100):
  """Compute annealed importance sampling trajectories for a batch of data. 
  Could be used for *both* forward and reverse chain in BDMC.
  Args:
    model (vae.VAE): VAE model
    loader (iterator): iterator that returns pairs, with first component
      being `x`, second would be `z` or label (will not be used)
    forward (boolean): indicate forward/backward chain
    schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`
    n_sample (int): number of importance samples
  Returns:
      A list where each element is a torch.autograd.Variable that contains the 
      log importance weights for a single batch of data
  """

  def log_f_i(ys, theta, zs, t): #, log_likelihood_fn=utils.log_bernoulli):
    """Unnormalized density for intermediate distribution `f_i`:
        f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
    =>  log f_i = log p(z) + t * log p(x|z)
    """
    zeros = torch.zeros(B, model.dim)#.cuda()
    log_prior = log_prior_fn(theta, zs)
    log_likelihood = log_likelihood_fn(ys, zs)

    return log_prior + t * log_likelihood

  logws = []
  for i, batch in enumerate(loader):
    B = batch.size(0) #* n_sample
    #batch = batch#.cuda()
    #batch = utils.safe_repeat(batch) #, n_sample)
    ys = batch
    #print('ys', ys.shape)

    with torch.no_grad():
      epsilon = torch.ones(theta.shape).mul_(0.01)
      accept_hist = torch.zeros(theta.shape)#.cuda()
      logw = torch.zeros(len(schedule)) #theta.shape)#.cuda()

    # initial sample of z
    # if forward:
    #   current_z = torch.randn(B, model.latent_dim).cuda()
    # else:
    #   current_z = utils.safe_repeat(post_z, n_sample).cuda()
    theta = theta.requires_grad_()

    for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
      # update log importance weight
      log_int_1 = log_f_i(ys, theta, zs, t0)
      log_int_2 = log_f_i(ys, theta, zs, t1)
      logw[j] = log_int_2 - log_int_1

      # resample velocity
      v = torch.randn(theta.size())#.cuda()

      def U(theta):
        return -log_f_i(ys, theta, zs, t1)

      def grad_U(theta):
        # grad w.r.t. outputs; mandatory in this case
        grad_outputs = torch.ones(theta.shape)#.cuda()
        # torch.autograd.grad default returns volatile
        grad_theta = torchgrad(U(theta), theta)[0]#, grad_outputs=grad_outputs)[0]
        #grad_zs = torchgrad(U(theta), zs, grad_outputs=grad_outputs)[0]
        # clip by norm
        max_ = B * model.dim * 100.
        grad_theta = torch.clamp(grad_theta, -max_, max_)
        grad_theta.requires_grad_()
        #print(grad_theta.shape)
        return grad_theta

      #print()

      def normalized_kinetic(v):
        zeros = torch.zeros(theta.shape) #B, model.dim)#.cuda()
        return -utils.log_normal(v, zeros, zeros)

  
      new_theta, new_v = hmc.hmc_trajectory(theta, v, U, grad_U, epsilon)
      theta, epsilon, accept_hist = hmc.accept_reject(
          theta, v,
          new_theta, new_v,
          epsilon,
          accept_hist, j,
          U, K=normalized_kinetic)

      #print(ys.shape)
      #print((ys[:, None, :, 0] - theta[None, :, :, 0]).shape)
      zs = torch.argmin(torch.norm(ys[:, None, :, 0] - theta[None, :, :, 0], dim=-1), dim=1)
      #print('zs', zs.shape)

    print(logw.shape)
    logw = utils.log_mean_exp(logw)
    #logw = utils.log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
    if not forward:
      logw = -logw
    logws.append(logw.data)
    print('Last batch stats %.4f' % (logw.mean().cpu().data.numpy()))
    print(log_likelihood_fn(ys, zs))

  return logws, zs, theta
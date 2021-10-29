import os
import sys

path_to_tools = '/home/daniil/pycharm_dir/gans_sampling'

api_path_cifar = os.path.join(path_to_tools, 'tools', 'cifar10_utils')
api_path_sampling = os.path.join(path_to_tools, 'tools', 'sampling_utils')
api_path_gan_metrics = os.path.join(path_to_tools, 'tools', 'gan_metrics')
models_cifar_scratch_path = os.path.join(path_to_tools, 'models', 'models_cifar10')

sys.path.append(api_path_cifar)
sys.path.append(api_path_sampling)
sys.path.append(api_path_gan_metrics)

import numpy as np
import random

import torch
from distributions import IndependentNormal

from functools import partial

from dcgan import (Discriminator_cifar10,
                   Generator_cifar10)

from params_cifar10 import args
from general_utils import DotDict, Discriminator_logits

from ebm_sampling import (load_data_from_batches,
                          gan_energy,
                          langevin_sampling)

from sir_ais_sampling import (sir_independent_mala_sampling,
                              sir_independent_sampling,
                              cisir_adaptive_sampling)

args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = args.device

G = Generator_cifar10(ngpu=1)
D = Discriminator_cifar10(ngpu=1)

D.load_state_dict(torch.load(os.path.join(models_cifar_scratch_path, 'netD_epoch_199.pth')))
G.load_state_dict(torch.load(os.path.join(models_cifar_scratch_path, 'netG_epoch_199.pth')))

D_logits = Discriminator_logits(D, ngpu=1)

if torch.cuda.is_available():
    D = D.to(device).eval()
    G = G.to(device).eval()
    D_logits = D_logits.to(device).eval()

G.z_dim = 100
G.device = device
z_dim = 100

loc = torch.zeros(z_dim).to(device)
scale = torch.ones(z_dim).to(device)

proposal_args = DotDict()
proposal_args.device = device
proposal_args.loc = loc
proposal_args.scale = scale
proposal = IndependentNormal(proposal_args)

log_prob = True
normalize_to_0_1 = True


def z_transform(z):
    return z.unsqueeze(-1).unsqueeze(-1)


target_gan = partial(gan_energy,
                     generator=G,
                     discriminator=D_logits,
                     proposal=proposal,
                     normalize_to_0_1=normalize_to_0_1,
                     log_prob=log_prob,
                     z_transform=z_transform)

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

batch_size = 200
n = 1000
n_steps = 5001
N = 5

grad_step = 2*0.01
eps_scale = (2 * grad_step) ** 0.5

method_name = 'sir_mala_dcgan_cifar_recalc'
path_to_save = '/home/daniil/gans-mcmc/saved_numpy_arrays'
file_name = f'{method_name}_N_{N}_nsteps_{n_steps}_step_{grad_step}_eps_{eps_scale}'
every_step = 40
continue_z = None

acceptance_rule = 'Hastings'

z_last_np, zs = sir_independent_mala_sampling(target_gan, proposal, batch_size, n,
                                              path_to_save, file_name, every_step,
                                              continue_z, n_steps, N, grad_step, eps_scale,
                                              acceptance_rule)
print("SIR + MALA done!")

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

batch_size = 200
n = 1000
n_steps = 5001
N = 5

method_name = 'sir_dcgan_cifar_recalc_1k'
path_to_save = '/home/daniil/gans-mcmc/saved_numpy_arrays'
file_name = f'{method_name}_N_{N}_nsteps_{n_steps}'

every_step = 40
continue_z = None

z_last_np, zs = sir_independent_sampling(target_gan, proposal, batch_size, n,
                                         path_to_save, file_name, every_step,
                                         continue_z, n_steps, N)
print("SIR done!")

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

batch_size = 200
n = 1000
n_steps = 5001
N = 5
alpha = 0.99
bernoulli_prob_corr = 0.5

method_name = 'adaptive_cisir_dcgan_cifar_recalc_1k'
path_to_save = '/home/daniil/gans-mcmc/saved_numpy_arrays'
file_name = f'{method_name}_N_{N}_alpha_{alpha}_nsteps_{n_steps}_eps_{bernoulli_prob_corr}'
every_step = 40
continue_z = None

z_last_np, zs = cisir_adaptive_sampling(target_gan, proposal, batch_size, n,
                                                 path_to_save, file_name, every_step,
                                                 continue_z, n_steps, N, alpha, bernoulli_prob_corr)
print("EX2MCMC done!")

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

batch_size = 1000
n = 1000
n_steps = 5001
grad_step = 0.01
eps_scale = (2 * grad_step) ** 0.5

method_name = 'ula_dcgan_cifar_recalc_1k'
path_to_save = '/home/daniil/gans-mcmc/saved_numpy_arrays'
file_name = f'{method_name}_eps_{grad_step}_noise_scale_{eps_scale}_nsteps_{n_steps}'
every_step = 40
continue_z = None

z_last_np, zs = langevin_sampling(target_gan, proposal, batch_size, n,
                                  path_to_save, file_name, every_step,
                                  continue_z,
                                  n_steps, grad_step, eps_scale)

print("ULA done!")
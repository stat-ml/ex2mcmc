import os
import sys

path_to_tools = '/home/daniil/pycharm_dir/gans_sampling'

api_path_cifar = os.path.join(path_to_tools, 'tools', 'cifar10_utils')
api_path_sampling = os.path.join(path_to_tools, 'tools', 'sampling_utils')
api_path_gan_metrics = os.path.join(path_to_tools, 'tools', 'gan_metrics')
api_path_sngan_utils = os.path.join(path_to_tools, 'tools', 'sngan_cifar10_utils')
models_cifar_scratch_path = os.path.join(path_to_tools, 'models', 'models_cifar10')

sys.path.append(api_path_cifar)
sys.path.append(api_path_sampling)
sys.path.append(api_path_gan_metrics)
sys.path.append(api_path_sngan_utils)

import numpy as np
import random
import easydict

import torch
from distributions import IndependentNormal

from functools import partial

from general_utils import DotDict, Discriminator_logits

from ebm_sampling import (load_data_from_batches,
                          gan_energy, langevin_sampling)

from metrics_utils import (calculate_images_statistics)

import source.models.sngan as models

net_G_models = {
    'res32': models.ResGenerator32,
    'res48': models.ResGenerator48,
    'cnn32': models.Generator32,
    'cnn48': models.Generator48,
}

net_D_models = {
    'res32': models.ResDiscriminator32,
    'res48': models.ResDiscriminator48,
    'cnn32': models.Discriminator32,
    'cnn48': models.Discriminator48,
}

FLAGS = easydict.EasyDict({
    'arch': 'cnn32',
    'z_dim': 100
})

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

path_to_sngan = os.path.join(models_cifar_scratch_path, "model.pt")

G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
G.load_state_dict(torch.load(path_to_sngan)['net_G'])
G.eval()

D = net_D_models[FLAGS.arch]().to(device)

D.load_state_dict(torch.load(path_to_sngan)['net_D'])
D.eval()

G.z_dim = FLAGS.z_dim
G.device = device
z_dim = FLAGS.z_dim

loc = torch.zeros(z_dim).to(device)
scale = torch.ones(z_dim).to(device)

proposal_args = DotDict()
proposal_args.device = device
proposal_args.loc = loc
proposal_args.scale = scale
proposal = IndependentNormal(proposal_args)

log_prob = True
normalize_to_0_1 = True

target_gan = partial(gan_energy,
                     generator=G,
                     discriminator=D,
                     proposal=proposal,
                     normalize_to_0_1=normalize_to_0_1,
                     log_prob=log_prob)

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

batch_size = 200
n = 50000
n_steps = 41

grad_step = 0.01
eps_scale = (2 * grad_step) ** 0.5

method_name = 'ula_sngan_cifar_recalc'
path_to_save = '/home/daniil/gans-mcmc/saved_numpy_arrays'
file_name = f'{method_name}_nsteps_{n_steps}_step_{grad_step}_eps_{eps_scale}'
every_step = 40
continue_z = None

z_last_np, zs = langevin_sampling(target_gan, proposal, batch_size, n,
                                              path_to_save, file_name, every_step,
                                              continue_z, n_steps, grad_step, eps_scale)

load_np = load_data_from_batches(n, batch_size,
                                 path_to_save, file_name)

batch_size = 50
random_seed = 42
path_to_save_np = '/home/daniil/gans-mcmc/saved_numpy_arrays'
path_to_save = path_to_save_np
method_name = f'ula_sngan_stats_{n_steps}_steps_step_{grad_step}_eps_{eps_scale}'
dataset = "cifar10"
calculate_is = False

z_transform = lambda x: x

cisir_statistics = calculate_images_statistics(z_agg_step = load_np,
                                             G = G,
                                             device = device,
                                             batch_size = batch_size,
                                             path_to_save = path_to_save,
                                             path_to_save_np = path_to_save_np,
                                             method_name = method_name,
                                             random_seed = random_seed,
                                             every_step = every_step,
                                             dataset = dataset,
                                             calculate_is = calculate_is,
                                             z_transform = z_transform)
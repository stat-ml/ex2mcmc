"""
Code adapted from "Your GAN is Secretly an Energy-based Model and You Should use Discriminator Driven Latent Sampling".

@article{DBLP:journals/corr/abs-2003-06060,
  author    = {Tong Che and
               Ruixiang Zhang and
               Jascha Sohl{-}Dickstein and
               Hugo Larochelle and
               Liam Paull and
               Yuan Cao and
               Yoshua Bengio},
  title     = {Your {GAN} is Secretly an Energy-based Model and You Should use Discriminator
               Driven Latent Sampling},
  journal   = {CoRR},
  volume    = {abs/2003.06060},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.06060},
  eprinttype = {arXiv},
  eprint    = {2003.06060},
  timestamp = {Tue, 17 Mar 2020 14:18:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2003-06060.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

import yaml
from pathlib import Path
import torch
from functools import partial
from tqdm import trange
import argparse
import chainer
import numpy as np
import threading
from PIL import Image
import cupy
import os, sys, time
import shutil
import yaml

from chainer import training
import chainer.functions as F
from chainer import Variable
from chainer.training import extension
from chainer.training import extensions

import source.yaml_utils as yaml_utils
from source.miscs.random_samples import sample_continuous, sample_categorical
from evaluation import load_inception_model
from source.inception.inception_score import inception_score
# from source.inception.inception_score_tf import get_inception_score as inception_score_tf
# from source.inception.inception_score_tf import get_mean_and_cov as get_mean_cov_tf
from evaluation import get_mean_cov as get_mean_cov_chainer
from evaluation import FID


_RUN_BASELINE = False
N = 5000
EVERY = 20

def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    dis_conf = config.models['binary_discriminator']
    dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])
    return gen, dis


def gan_target(
    z,
    generator,
    discriminator,
    proposal,
    alpha=1,
    temperature=1
):
    logp_z = F.sum(proposal.log_prob(z), 1, keepdims=True)
    x = generator(batchsize=z.shape[0], z=z)
    dgz = discriminator(x)

    E = -(logp_z + dgz) * temperature

    return -E, logp_z, dgz


def grad_energy(point, target):
    minus_energy, logp_z, dgz = target(point)

    grad = chainer.grad((-minus_energy,), (point,))
    return -minus_energy, grad, logp_z, dgz


def langevin_dynamics(
    z, target, proposal, xp, n_steps, grad_step, eps_scale, verbose=True, every=10
):
    z_sp = []
    info = []
    batch_size, _ = z.shape[0], z.shape[1]

    range_gen = trange if verbose else range

    for _ in range_gen(n_steps):
        if _ % every == 0:
            z_sp.append(z.data)

        eps = eps_scale * proposal.sample((batch_size,))
        E, grad, logp_z, dgz = grad_energy(z, target)

        if _ % every == 0:
            log_p_grad = chainer.grad((logp_z,), (z,))[0].data
            d_grad = grad[0].data + log_p_grad

            print(dgz.data.mean(), logp_z.data.mean())

            info.append([
                cupy.linalg.norm(log_p_grad, -1).mean(), 
                cupy.linalg.norm(d_grad, -1).mean(), 
                cupy.linalg.norm(grad[0].data, -1).mean(),
                1.,
                logp_z.data.mean(),
                dgz.data.mean(),
                E.data.mean()
                ])

        z = z - grad_step * grad[0] + eps

    z_sp.append(z.data)
    return z_sp, info


class CorrelatedKernel:
    def __init__(self, xp, corr_coef=0, bernoulli_prob_corr=0):
        self.corr_coef = corr_coef
        self.bern = chainer.distributions.Bernoulli(xp.array(bernoulli_prob_corr).astype(xp.float32))
        self.xp=xp

    def __call__(self, z, proposal, N, batch_size=1):
        corr1 = F.cast(self.bern.sample(sample_shape=(batch_size,)), self.xp.float32)
        correlation = self.corr_coef * corr1
        
        latent_var = correlation[:, None] * z + (
            1.0 - correlation[:, None] ** 2
        ) ** 0.5 * proposal.sample(
            (batch_size,),
        )
        corr2 = F.cast(self.bern.sample((batch_size, N)), self.xp.float32)
        correlation_new = self.corr_coef * corr2
        z_new = correlation_new[..., None] * latent_var[:, None, :] + (
            1.0 - correlation_new[..., None] ** 2
        ) ** 0.5 * proposal.sample(
            (
                batch_size,
                N,
            ),
        )
        return z_new, corr2 * corr1[:, None]


def mala_dynamics(
    z,
    target,
    proposal,
    xp,
    n_steps=1,
    grad_step=0.01,
    eps_scale=0.1,
    # adapt_stepsize=False,
):
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    std_norm = chainer.distributions.Normal(xp.zeros((z_dim, ), dtype=xp.float32), xp.ones((z_dim, ), dtype=xp.float32))

    # if adapt_stepsize:
    #     eps_scale = (2 * grad_step) ** (1 / 2)
    uniform = chainer.distributions.Uniform(low=xp.array(0.0).astype(xp.float32), high=xp.array(1.0).astype(xp.float32))
    acceptance = xp.zeros((batch_size,), dtype=xp.float32)#.to(device)

    E, grad, logp_z, dgz = grad_energy(z, target)
    grad = grad[0]
    for _ in range(n_steps):
        eps = eps_scale * std_norm.sample((batch_size,))

        new_z = z - grad_step * grad + eps

        E_new, grad_new, logp_z, dgz = grad_energy(new_z, target)
        grad_new = grad_new[0]

        energy_part = E - E_new

        propose_vec_1 = z - new_z + grad_step * grad_new
        propose_vec_2 = new_z - z + grad_step * grad

        propose_part_1 = std_norm.log_prob(propose_vec_1 / eps_scale)
        propose_part_2 = std_norm.log_prob(propose_vec_2 / eps_scale)

        propose_part = F.sum(propose_part_1 - propose_part_2, 1, keepdims=False)

        energy_part = F.reshape(energy_part, (z.shape[0],))
        log_accept_prob = propose_part + energy_part


        generate_uniform_var = uniform.sample((batch_size,))
        log_generate_uniform_var = F.log(generate_uniform_var)

        mask = log_generate_uniform_var.array < log_accept_prob.array
        
        z = z.data 
        grad = grad.data
        grad_new = grad_new.data 
        E = E.data
        E_new = E_new.data
        new_z = new_z.data
        z[mask] = new_z[mask]
        grad[mask] = grad_new[mask]
        E[mask] = E_new[mask]
        z = Variable(z)
        z_sp.append(z)

        acceptance += mask

    acceptance /= n_steps

    return z_sp, acceptance


def ex2mcmc_dynamics(
    z,
    target,
    proposal,
    xp,
    n_steps,
    N,
    corr_coef=0.0,
    bernoulli_prob_corr=0.0,
    every=10,
    verbose=True,
    mala_steps=1,
    step_size=0.01,
    noise_scale=0.1,
):  # z assumed from proposal !
    z_sp = []
    batch_size, z_dim = z.shape[0], z.shape[1]

    range_gen = trange if verbose else range

    corr_ker = CorrelatedKernel(xp, corr_coef, bernoulli_prob_corr)
    acc = 0
    new = 0
    info = []

    for _ in range_gen(n_steps):
        z_pushed = z
        if _ % every == 0:
            z_sp.append(z_pushed.data)

        X, correlation = corr_ker(z, proposal, N, batch_size)
        X = F.concat((F.reshape(z, (batch_size, 1, z_dim)), X[:, 1:, :]), axis=1)
        
        X_view = F.reshape(X, (-1, z_dim))

        z_pushed = X_view
        minus_E, logp_z, dgz = target(z_pushed)
        log_weight = minus_E - logp_z

        log_weight = F.reshape(log_weight, (batch_size, N))
        max_logs = F.max(log_weight, axis=1)[:, None]
        log_weight = log_weight - max_logs
        weight = F.exp(log_weight)
        sum_weight = F.sum(weight, axis=1)
        weight = weight / sum_weight[:, None]

        weight = torch.FloatTensor(weight.array)
        weight[weight != weight] = 0.0
        weight[weight.sum(1) == 0.0] = 1.0

        indices = torch.multinomial(weight, 1).squeeze().tolist()

        correlation = correlation.data
        new = (np.array(indices) == 0).mean()
        correlation[np.array(indices) == 0, :] = 1
        new_indices = np.array(indices)
        new_indices[np.array(indices) == 0] = 1
        new_indices = new_indices - 1
        not_corr = (correlation[:, new_indices] == 0).mean()

        z = X[np.arange(batch_size), indices, :]

        if _ % every == 0:
            print(not_corr, new)
        
            logp_z = F.reshape(logp_z, (batch_size, N))[np.arange(batch_size), indices] #F.sum(proposal.log_prob(z), 1, keepdims=True)
            minus_E = F.reshape(minus_E, (batch_size, N))[np.arange(batch_size), indices]
            dgz = F.reshape(dgz, (batch_size, N))[np.arange(batch_size), indices]

            print(dgz.data.mean(), logp_z.data.mean())

            info.append([
                0, 
                0, 
                0,
                new,
                logp_z.data.mean(),
                dgz.data.mean(),
                -minus_E.data.mean()
                ])

        for _ in range(mala_steps):
            z, acc_ = mala_dynamics(z, target, proposal, xp, n_steps=1, grad_step=step_size, eps_scale=noise_scale)
            z = z[-1]
            acc += acc_.mean()

    z_pushed = z

    z_sp.append(z_pushed.data)
    acc /= n_steps
    
    return z_sp, info


def sample(gen, dis, config, n=50000, batchsize=100, method='ula'):
    ims = []
    zs = []
    infos = []

    xp = gen.xp
    z_dim = 128

    n_steps = config.n_steps
    batchsize = config.batchsize

    if 'ula' in config.method:
        dynamics = langevin_dynamics
    else:
        dynamics = ex2mcmc_dynamics
    kwargs = config.config['params']

    if 'hist_path' in config.config.keys():
        hist_zs = np.stack(np.split(np.load(Path(config.hist_path, 'latents/ula_latent_samples.npy').as_posix()).transpose(1, 0, 2), n // batchsize), 0)
        start = hist_zs[:, :, -1]
        hist_infos = list(np.loadtxt(Path(config.hist_path, 'info.txt').as_posix()))

    for i in range(0, n, batchsize):
        # with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        gen_ = gen.copy()
        with chainer.using_config('train', False):
            proposal = chainer.distributions.Normal(xp.zeros((z_dim, ), dtype=xp.float32), xp.ones((z_dim, ), dtype=xp.float32))
            target = partial(
                    gan_target,
                    generator=gen_,
                    discriminator=dis,
                    proposal=proposal,
                    alpha=config.alpha,
                    temperature=config.temperature
                )

            if 'hist_path' in config.config.keys():
                z = Variable(xp.asarray(start[i // batchsize]))
                z_sp, info = dynamics(z, target, proposal, xp, **kwargs, every=config.every)
            else:
                z = Variable(gen_.sample_z(batchsize))
                z_sp, info = dynamics(z, target, proposal, xp, **kwargs, every=config.every)

        x = gen(batchsize, z_sp[-1])
        x = chainer.cuda.to_cpu(x.data)
        ims.append(np.zeros((batchsize, 3, 32, 32), dtype=np.uint8))
        zs.append(np.stack([chainer.cuda.to_cpu(o) for o in z_sp], axis=0))
        infos.append(np.stack(info, 0))
        if i % 50 == 0:
            print(i)

    ims = np.asarray(ims)
    zs = np.stack(zs, axis=0)
    if 'hist_path' in config.config.keys():
        zs = np.concatenate([hist_zs.transpose(0, 2, 1, -1), zs[:, 1:]], 1)
    _, _, _, h, w = ims.shape
    ims = ims.reshape((n, 3, h, w))
    infos = np.mean(np.stack(infos, 0), 0)
    if 'hist_path' in config.config.keys():
        infos = np.concatenate([hist_infos, infos], 0)
    np.savetxt(Path(config.dst, 'info.txt').as_posix(), infos)
    return ims, zs
    

def parallel_apply(modules, config, n_list, devices, batchsize=100, method='ula'):
    lock = threading.Lock()
    results = {}

    def _worker(pid, module, n, device):
        try:
            with chainer.using_device(device):
                gen, dis = module
                ims, zs = sample(gen, dis, config, n, batchsize, method=method)
            with lock:
                results[pid] = (ims, zs)
        except Exception as e:
            with lock:
                results[pid] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, n, device))
                   for i, (module, n, device) in
                   enumerate(zip(modules, n_list, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], n_list[0], devices[0])

    im_outputs = []
    z_outputs = []
    for i in range(len(modules)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        ims, zs = output
        im_outputs.append(ims)
        z_outputs.append(zs)
    return im_outputs, z_outputs

def sample_multigpu(gen, dis, config, gpu_list, n=50000, batchsize=100, method='ula'):
    gen_list = [gen.copy() for _ in gpu_list]
    dis_list = [dis.copy() for _ in gpu_list]
    for gpu_id, gen in zip(gpu_list, gen_list):
        gen.to_gpu(gpu_id)
    for gpu_id, dis in zip(gpu_list, dis_list):
        dis.to_gpu(gpu_id)
    modules = list(zip(gen_list, dis_list))

    n_gpu = len(gpu_list)
    n_list = [n // batchsize // n_gpu for _ in gpu_list]
    n_list[0] = n_list[0] + (n // batchsize) % n_gpu
    n_list = [n * batchsize for n in n_list]
    print('n_list', n_list)

    ims, zs = parallel_apply(modules, config, n_list, gpu_list, batchsize, method=method)
    ims = np.vstack(ims)
    zs = np.concatenate(zs, axis=0)
    return ims, zs

# def langevin_sample_vis(gen, dis, config, dst, rows=10, cols=10, seed=0):
#     """Visualization of rows*cols images randomly generated by the generator."""
#     @chainer.training.make_extension()
#     def make_image(trainer):
#         np.random.seed(seed)
#         n_images = rows * cols
#         x = langevin_sample(gen, dis, config, n_images, batchsize=n_images)
#         _, _, h, w = x.shape
#         x = x.reshape((rows, cols, 3, h, w))
#         x = x.transpose(0, 3, 1, 4, 2)
#         x = x.reshape((rows * h, cols * w, 3))
#         preview_dir = Path(dst, 'preview')
#         preview_path = Path(preview_dir, 'image{:0>8}.png'.format(trainer.updater.iteration))
#         Path(preview_dir).mkdir(exist_ok=True, parents=True)
#         # if not os.path.exists(preview_dir):
#         #     os.makedirs(preview_dir)
#         Image.fromarray(x).save(preview_path)

#     return make_image


def baseline_gen_images(gen, n=50000, batchsize=100):
    ims = []
    xp = gen.xp
    for i in range(0, n, batchsize):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(batchsize)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        ims.append(x)
    ims = np.asarray(ims)
    _, _, _, h, w = ims.shape
    ims = ims.reshape((n, 3, h, w))
    return ims

def calc_inception_score(ims, inception_model, splits, dst=None, exp_name=''):    
    mean, std = inception_score(inception_model, ims, splits=splits)
    # mean, std = inception_score_tf(ims, splits=splits)
    eval_ret = {
        'inception_mean': mean,
        'inception_std': std
    }
    if dst is not None:
        preview_dir = Path(dst, 'stats')
        Path(preview_dir).mkdir(exist_ok=True, parents=True)
        preview_path = Path(preview_dir, 'inception_score_{}.txt'.format(exp_name))
        np.savetxt(preview_path, np.array([mean, std]))
    return eval_ret

def eval_inception(gen, dis, config, n_images, dst, gpu_list, splits=10, batchsize=250, path=None, method=''):
    if _RUN_BASELINE:
        gen.to_gpu(gpu_list[0])
        ims = baseline_gen_images(gen, n_images, batchsize=batchsize).astype("f")
    else:
        ims, zs = sample_multigpu(gen, dis, config, gpu_list, n_images, batchsize=batchsize, method=method)

        ims = ims.astype('f')
        zs = np.reshape(np.transpose(zs, axes=(1, 0, 2, 3)), (zs.shape[1], -1, zs.shape[-1]))
        if dst is not None:
            preview_dir = Path(dst, 'latents')
            preview_dir.mkdir(exist_ok=True, parents=True)
            save_path = Path(preview_dir, '{}_latent_samples'.format(method))
            np.save(save_path, zs)
    model = load_inception_model(path)
    model.to_gpu(gpu_list[0])
    eval_ret = calc_inception_score(ims, model, splits, dst)
    return eval_ret

def eval_inception_with_zs(gen, dis, config, n_images, dst, gpu_list, splits=10, batchsize=250, path=None, exp_name=''):
    preview_dir = Path(dst, 'latents')
    save_path = Path(preview_dir, '{}_latent_samples.npy'.format(exp_name))
    zs = np.load(save_path)
    model = load_inception_model(path)
    model.to_gpu(gpu_list[0])
    gen.to_gpu(gpu_list[0])
    xp = gen.xp
    for z_iter in range(0, zs.shape[0]):
        ims = []
        for batch_idx in range(0, n_images, batchsize):
            z_batch = xp.asarray(zs[z_iter, batch_idx : (batch_idx + batchsize), :])
            x = gen(batchsize, z_batch)
            x = chainer.cuda.to_cpu(x.data)
            x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
            ims.append(x)
        ims = np.asarray(ims)        
        _, _, _, h, w = ims.shape
        ims = ims.reshape((n_images, 3, h, w)).astype("f")
        print(ims.shape)
        eval_ret = calc_inception_score(ims, model, splits, dst)
        print('z_step', z_iter * 10, eval_ret)

def eval_fid_with_zs(gen, dis, config, n_images, dst, gpu_list, splits=10, batchsize=250, path=None, method='ula'):
    _use_tf = False
    if _use_tf:
        get_mean_cov = get_mean_cov_tf
    else:
        get_mean_cov = get_mean_cov_chainer
        model = load_inception_model(path)
        model.to_gpu(gpu_list[0])
        gen.to_gpu(gpu_list[0])
        # dis.to_gpu(gpu_list[0])
        xp = gen.xp
    preview_dir = Path(dst, 'latents')
    save_path = Path(preview_dir, '{}_latent_samples.npy'.format(method))

    stat_file = '../cifar-10-fid.npz'
    zs = np.load(save_path)
    stat = np.load(stat_file)


    if 'hist_path' in config.config.keys():
        start = zs.shape[0] - config.n_steps // config.every - 1
        #start = 0
    else:
        start = 0

    for z_iter in range(start, zs.shape[0]): #+start):
        ims = []
        #E = 0
        for batch_idx in range(0, n_images, batchsize):
            z_batch = xp.asarray(zs[z_iter][batch_idx : (batch_idx + batchsize), :])
            #with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(batchsize, z=z_batch)

            x = chainer.cuda.to_cpu(x.data)
            x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
            ims.append(x)
        
        ims = np.asarray(ims)  
        print(ims.shape)   
        _, _, _, h, w = ims.shape
        ims = ims.reshape((-1, 3, h, w)).astype(np.float32)[:n_images]
        fid_n = config.n_fid
        fids = []
        for k in range(0, n_images, fid_n):
            x = ims[k : k + fid_n]
            if _use_tf:
                mean, cov = get_mean_cov(x)
            else:
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    mean, cov = get_mean_cov(model, x, batch_size=batchsize)

            fid = FID(stat["mean"], stat["cov"], mean, cov)
            fids.append(fid)
            break

        print('z_step', z_iter * config.every, np.mean(fids), fids)
        np.savetxt(Path(dst, 'fid_{}.txt'.format(z_iter)), np.array(fids))

    #np.savetxt(Path(dst, 'energy_{}.txt'.format(config.every)), np.array(energies))


def viz_zs(gen, dst, gpu_list, batchsize=250, method='ula'):
    gen.to_gpu(gpu_list[0])
    xp = gen.xp
    preview_dir = Path(dst, 'latents')
    save_path = Path(preview_dir, '{}_latent_samples.npy'.format(method))

    zs = np.load(save_path)
    print(zs.shape)

    z_iter = len(zs) - 1 
    ims = []
    batchsize = 1000
    n = 20

    length = 30
    start_id = 0 #np.random.randint(0, len(zs[0]) - n - batchsize)
    #for batch_idx in trange(0, n, batchsize):
    for i in range(length+1, 1, -1):
        z_batch = xp.asarray(zs[-i, start_id : start_id + batchsize, :])
        #with chainer.using_config('train', True), chainer.using_config('enable_backprop', False):
        x = gen(batchsize, z=z_batch)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        ims.append(x)
    
    ims = np.asarray(ims)[:, :n]
    print(ims.shape)
    _, _, _, h, w = ims.shape
    x = ims.transpose(1, 3, 0, 4, 2)
    x = x.reshape((n * h, length * w, 3))
    Image.fromarray(x).save(Path(preview_dir, 'images.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gan_config', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('method_config', type=str)

    parser.add_argument('--gpu', type=str, default='0', help='index of gpu to be used')
    parser.add_argument('--results_dir', type=str, default='./results/gans',
                        help='directory to save the results to')
    parser.add_argument('--inception_model_path', type=str, default='./datasets/inception_model',
                        help='path to the inception model')
    parser.add_argument('--loaderjob', type=int,
                        help='number of parallel data loading processes')
    parser.add_argument('--gen_ckpt', type=str, default='../ResNetGenerator_50000.npz',
                        help='path to the saved generator snapshot model file to load')
    parser.add_argument('--dis_ckpt', type=str, default='results/gans/DiscriminatorPaper_4000.npz')
    parser.add_argument('--exp_name', type=str, default='test',
                        help='name of the experiment')
    parser.add_argument('--method', type=str, default='ula')
    parser.add_argument('--splits', type=int, default=1)
    parser.add_argument('--rows', type=int, default=10)
    parser.add_argument('--cols', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_zs', action='store_true')
    parser.add_argument('--eval_fid', action='store_true')
    parser.add_argument('--baseline', action='store_true')

    args = parser.parse_args()

    gan_config = yaml_utils.Config(yaml.load(open(args.gan_config)))
    method_config = yaml_utils.Config(yaml.load(open(args.method_config)))
    args.splits = method_config.n_repeat

    #if not args.vis:
    gpus = list(map(int, args.gpu.split(',')))
    chainer.cuda.get_device_from_id(gpus[0]).use()

    #Models
    gen, dis = load_models(gan_config)
    chainer.serializers.load_npz(args.gen_ckpt, gen)
    chainer.serializers.load_npz(gan_config.models['binary_discriminator']['path'], dis)
    
    out = Path(args.results_dir, '{}_{}'.format(method_config.method, Path(args.gan_config).stem))
    out.mkdir(exist_ok=True, parents=True)
    method_config.dst = out 
    
    method = args.method

    if args.eval:
        n_images = int(method_config.n_fid * args.splits)
        path = args.inception_model_path
        ret = eval_inception(gen, dis, method_config, n_images, dst=out, gpu_list=gpus, splits=args.splits, path=path, method=method, batchsize=100) #method_config.batchsize) #100)
        print(ret)
    if args.eval_zs:
        n_images = int(method_config.n_fid * args.splits)
        path = args.inception_model_path
        exp_name = args.exp_name
        ret = eval_inception_with_zs(gen, dis, method_config, n_images, dst=out, gpu_list=gpus, splits=args.splits, path=path, exp_name=exp_name)
    if args.vis:
         viz_zs(gen, dst=out, gpu_list=gpus, method=method)
         return
    
    #if args.eval_fid:
    n_images = int(method_config.n_fid * args.splits)
    path = args.inception_model_path
    exp_name = args.exp_name
    ret = eval_fid_with_zs(gen, dis, method_config, n_images, dst=out, gpu_list=gpus, splits=args.splits, path=path, method=method)

if __name__ == '__main__':
    main()
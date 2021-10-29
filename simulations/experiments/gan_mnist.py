import argparse
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from utils import DotConfig

from iterative_sir.models.dcgan import Discriminator, Generator
from iterative_sir.models.utils import Discriminator_logits
from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC
from iterative_sir.sampling_utils.distributions import IndependentNormal
from iterative_sir.sampling_utils.ebm_sampling import MALA, ULA, gan_energy
from iterative_sir.sampling_utils.metrics import inception_score


sns.set_theme(style="ticks", palette="deep")
warnings.filterwarnings("ignore")


def load_model(model_config, device):
    model_class = eval(model_config.model_class)
    model = model_class(**model_config.params.dict).to(device)
    model.load_state_dict(
        torch.load(model_config.weight_path, map_location=device)
    )
    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("model_config", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    return args


def main(config, model_config, run=True):
    colors = []
    names = []
    samples = []
    dists = []

    imageSize = 28
    dataset = dset.MNIST(
        root=config.data_root,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    real_sample = torch.stack(
        [dataset[i][0][0] for i in range(config.n_ref)], 0
    )

    if run:
        device = config.device

        G = load_model(model_config.generator, device=device).eval()
        D = load_model(model_config.discriminator, device=device).eval()
        D_logits = Discriminator_logits(D, ngpu=1).eval()

        D_logits = D_logits.eval()
        G = G.eval()

        z_dim = G.nz
        dim = z_dim
        loc = torch.zeros(z_dim).to(device)
        scale = torch.ones(z_dim).to(device)

        proposal = IndependentNormal(
            device=device, dim=z_dim, loc=loc, scale=scale
        )

        def z_transform(z):
            with torch.no_grad():
                z[-1, :] *= 0
            return z.unsqueeze(-1).unsqueeze(-1)

        log_prob = True
        normalize_to_0_1 = True
        target = partial(
            gan_energy,
            generator=G,
            discriminator=D_logits,
            proposal=proposal,
            normalize_to_0_1=normalize_to_0_1,
            log_prob=log_prob,
            z_transform=z_transform,
        )

        evols = dict()

        batch_size = config.batch_size

        # gan_sample = G(torch.randn(((batch_size - 1) * config.trunc_chain_len, z_dim, 1, 1))).detach().cpu()
        start = time.time()
        gan_sample = G(torch.randn((1000, z_dim, 1, 1))).detach().cpu()
        end = time.time()
        print(end - start)

        distances = torch.cdist(
            real_sample.reshape(real_sample.shape[0], -1),
            gan_sample.reshape(gan_sample.shape[0], -1),
        )
        gan_dist = distances.min(0).values.tolist()

        start = time.time()
        gan_sample2 = G(torch.randn((1000, z_dim, 1, 1))).detach().cpu()
        end = time.time()
        print(end - start)

        distances = torch.cdist(
            real_sample.reshape(real_sample.shape[0], -1),
            gan_sample2.reshape(gan_sample.shape[0], -1),
        )
        gan_dist2 = distances.min(0).values.tolist()

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            colors.append(info.color)
            names.append(method_name)
            mcmc_class = info.mcmc_class
            mcmc_class = eval(mcmc_class)
            mcmc = mcmc_class(**info.params.dict, dim=dim)

            z_0 = proposal.sample((config.batch_size,))
            out = mcmc(z_0, target, proposal, info.n_steps)
            # print(mcmc.grad_step)

            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

            sample = torch.stack(sample, 0)
            # slice = sample[-config.val_chain_len:, :-1, :].reshape(-1, dim)
            slice = sample[-1:, :-1, :].reshape(-1, dim)
            x_slice = G(slice.unsqueeze(-1).unsqueeze(-1))
            distances = torch.cdist(
                real_sample.reshape(real_sample.shape[0], -1),
                x_slice.reshape(x_slice.shape[0], -1),
            )
            distances = distances.min(0).values.tolist()
            # print(distances)
            dists.append(distances)

            shape = sample[-config.trunc_chain_len :].shape[:2]

            sample = (
                sample[-config.trunc_chain_len :, : config.nrows, :]
                .transpose(0, 1)
                .reshape(-1, dim)
            )
            x_gen = G(sample.unsqueeze(-1).unsqueeze(-1))
            x_gen = (
                x_gen.detach().cpu().numpy()
            )  # .reshape(*shape, *x_gen.shape[2:])
            samples.append(x_gen)

        if "respath" in config.dict.keys():
            dists = np.array(dists)
            Path(config.respath).mkdir(exist_ok=True, parents=True)
            np.save(Path(config.respath, "dists"), dists)
    else:
        dists = np.load(Path(config.respath, "dists").open("rb"))
        for method_name, info in config.methods.items():
            colors.append(info.color)

    if "figpath" in config.dict.keys():
        fig = plt.figure(figsize=(4, 4))
        for name, dist, color in zip(
            config.methods.dict.keys(), dists, colors
        ):
            sns.kdeplot(data=dist, label=name, color=color)
        sns.kdeplot(data=gan_dist, label="GAN", color="black")
        sns.kdeplot(data=gan_dist2, label="GAN", color="black")
        plt.grid()
        plt.legend()
        plt.title("Distance to nearest")
        plt.ylabel("Frequency")
        plt.savefig(Path(config.figpath, f"gan_mnist_dist.pdf"))

        if len(samples) > 0:
            for name, sample in zip(names, samples):
                images_np = sample.reshape(sample.shape[0], 28, 28)
                # R, C = 10, 10
                fig = plt.figure(figsize=(config.ncols, config.nrows))
                for i in range(config.nrows * config.ncols):
                    plt.subplot(config.nrows, config.ncols, i + 1)
                    plt.imshow(images_np[i], cmap="gray")
                    plt.axis("off")
                # fig.tight_layout()
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(Path(config.figpath, f"gan_mnist_{name}.pdf"))
                plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.load(Path(args.config).open("r"), Loader=yaml.FullLoader)
    config = DotConfig(config)

    model_config = yaml.load(
        Path(args.model_config).open("r"), Loader=yaml.FullLoader
    )
    model_config = DotConfig(model_config)

    if args.result_path is not None:
        run = False
        config.respath = args.result_path
    else:
        run = True
    main(config, model_config, run)

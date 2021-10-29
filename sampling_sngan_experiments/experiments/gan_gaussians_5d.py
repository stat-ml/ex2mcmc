import argparse
import pickle
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ot
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import trange
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC
from iterative_sir.sampling_utils.distributions import (
    GaussianMixture,
    IndependentNormal,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA, ULA, gan_energy
from iterative_sir.sampling_utils.metrics import Evolution
from iterative_sir.sampling_utils.visualization import plot_chain_metrics
from iterative_sir.toy_examples_utils import (
    prepare_25gaussian_data,
    prepare_gaussians,
)
from iterative_sir.toy_examples_utils.gan_fc_models import (
    Discriminator_fc,
    Generator_fc,
)


sns.set_theme(style="ticks", palette="deep")
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("model_config", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    return args


def load_model(model_config, device):
    model_class = eval(model_config.model_class)
    model = model_class(**model_config.params.dict).to(device)
    model.load_state_dict(
        torch.load(model_config.weight_path, map_location=device)
    )
    return model


def main(config, model_config, run=True):
    colors = []

    if run:
        if config.multistart:
            assert config.train_size == config.batch_size
        else:
            pass
            # assert config.train_size == config.every

        device = config.device

        G = load_model(model_config.generator, device=device).eval()
        D = load_model(model_config.discriminator, device=device).eval()
        x_dim = D.n_in

        coord_limits = 2.0
        sigma = 0.05
        num_gaussian_per_dim = 3
        X_train = prepare_gaussians(
            num_samples_in_cluster=100,
            dim=x_dim,
            num_gaussian_per_dim=num_gaussian_per_dim,
            coord_limits=coord_limits,
            sigma=sigma,
        )
        X_train = X_train[
            np.random.choice(
                np.arange(X_train.shape[0]), config.train_size, replace=False
            )
        ]

        scaler = None

        z_dim = G.n_dim
        loc_proposal = torch.zeros(z_dim).to(G.device)
        scale_proposal = torch.ones(z_dim).to(G.device)

        proposal = IndependentNormal(
            device=device, dim=z_dim, loc=loc_proposal, scale=scale_proposal
        )

        normalize_to_0_1 = True
        log_prob = True
        target = partial(
            gan_energy,
            generator=G,
            discriminator=D,
            proposal=proposal,
            normalize_to_0_1=normalize_to_0_1,
            log_prob=log_prob,
        )

        coords_per_dim = np.linspace(
            -coord_limits, coord_limits, num=num_gaussian_per_dim
        )
        copy_coords = list(np.tile(coords_per_dim, (x_dim, 1)))
        centers = np.array(
            np.meshgrid(*copy_coords), dtype=np.float64
        ).T.reshape(-1, x_dim)
        locs = torch.stack(
            [
                torch.tensor(centers[i]).to(device)
                for i in range(centers.shape[0])
            ],
            0,
        )

        evols = dict()

        z_0 = proposal.sample((config.batch_size,))

        batch_size = config.batch_size
        target_sample = X_train
        gan_sample = []
        # for _ in trange(config.batch_size):
        #     gan_sample.append(G(proposal.sample((config.trunc_chain_len,))).detach().cpu())
        # gan_sample = torch.cat(gan_sample, 0)

        gan_sample = G(z_0).detach().cpu()

        evol = defaultdict(list)
        for i, gs in enumerate(gan_sample.reshape(config.n_eval, -1, x_dim)):
            evolution = Evolution(
                target_sample[gs.shape[0] * i : (i + 1) * gs.shape[0]],
                locs=locs,
                # target_log_prob=target,
                sigma=sigma,
                scaler=scaler,
            )
            evolution.invoke(gs)
            evol_ = evolution.as_dict()
            for k, v in evol_.items():
                evol[k].append(v)
        for k, v in evol.items():
            evol[k] = (
                np.mean(np.array(v), 0),
                np.std(np.array(v), 0, ddof=1) / np.sqrt(batch_size),
            )
        evols["GAN"] = evol
        colors.append("black")
        print(evol)

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            colors.append(info.color)
            mcmc_class = info.mcmc_class
            mcmc_class = eval(mcmc_class)
            mcmc = mcmc_class(**info.params.dict, dim=x_dim)

            out = mcmc(z_0, target, proposal, info.burn_in)
            if isinstance(out, tuple):
                out = out[0]

            out = mcmc(out[-1], target, proposal, info.n_steps)

            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

            if config.multistart:
                sample = ULA(
                    **config.methods.dict["ULA"]["params"], dim=x_dim
                )(sample[-1], target, proposal, 2)

            sample = torch.stack(sample, 0)  # len x batch x dim
            sample = sample[-config.n_chunks * info.every :]
            sample = torch.stack(torch.split(sample, config.every, 0), 0)
            zs_gen = sample.permute(2, 0, 1, 3)

            Xs_gen = G(zs_gen).detach().cpu().numpy()
            if scaler:
                Xs_gen = scaler.inverse_transform(Xs_gen)

            # samples.append(Xs_gen[0].reshape(-1, G.n_dim))

            # Xs_gen = Xs_gen[:config.evol_batch_size]
            if config.multistart:
                Xs_gen = Xs_gen.transpose(2, 1, 0, 3)
                Xs_gen = Xs_gen.reshape(config.n_eval, 1, -1, x_dim)

            print(Xs_gen.shape)

            evol = defaultdict(list)
            for i, X_gen in enumerate(Xs_gen):
                if config.multistart:
                    b = X_gen.shape[1]
                    tar = target_sample[i * b : (i + 1) * b]
                else:
                    tar = target_sample

                evolution = Evolution(
                    tar,  # [:X_gen.shape[1]],
                    locs=locs,
                    # target_log_prob=target,
                    sigma=sigma,
                    scaler=scaler,
                )
                for chunk in X_gen:
                    evolution.invoke(torch.FloatTensor(chunk))
                evol_ = evolution.as_dict()
                for k, v in evol_.items():
                    evol[k].append(v)

            for k, v in evol.items():
                evol[k] = (
                    np.mean(np.array(v), 0),
                    np.std(np.array(v), 0, ddof=1) / np.sqrt(batch_size),
                )
            evols[method_name] = evol
            print(evol)

        if "respath" in config.dict:
            name = "gan_gaussians_5d_metrics"
            if config.multistart:
                name = f"{name}_multi"
            name = f"{name}.pkl"
            pickle.dump(
                evols,
                Path(config.respath, name).open("wb"),
            )

    else:
        evols = pickle.load(Path(config.respath).open("rb"))
        evols = {k: evols[k] for k in config.methods.dict.keys()}
        for method_name, info in config.methods.items():
            colors.append(info.color)

    print(evols)

    if "figpath" in config.dict:
        SMALL_SIZE = 19  # 8
        MEDIUM_SIZE = 23  # 10
        BIGGER_SIZE = 23  # 12
        # mpl.rcParams["mathtext.rm"]

        plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rc("lines", lw=3)  # fontsize of the figure title
        plt.rc("lines", markersize=7)  # fontsize of the figure title

        plot_chain_metrics(
            evols,
            colors=colors,
            every=info.every,
            savepath=Path(config.figpath, "gan_gaussians_5d.pdf"),
        )


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

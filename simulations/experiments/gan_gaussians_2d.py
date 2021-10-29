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

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.distributions import (
    GaussianMixture,
    IndependentNormal,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA, ULA, gan_energy
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.metrics import Evolution
from iterative_sir.sampling_utils.visualization import plot_chain_metrics
from iterative_sir.toy_examples_utils import prepare_25gaussian_data
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
    names = []
    samples = []

    if run:
        device = config.device

        G = load_model(model_config.generator, device=device).eval()
        D = load_model(model_config.discriminator, device=device).eval()

        X_train, means = prepare_25gaussian_data(
            config.train_size,
            config.sigma,
            config.random_seed,
        )

        scaler = StandardScaler()
        if scaler:
            X_train_std = scaler.fit_transform(X_train)
        else:
            X_train_std = X_train

        dim = 2
        num_gauss = 25
        n_col = 5
        n_row = num_gauss // n_col
        s = 1
        # create points
        # coef_gaussian = 1.0 / num_gauss
        locs = torch.stack(
            [
                torch.tensor([(i - 2) * s, (j - 2) * s] + [0] * (dim - 2)).to(
                    device,
                )
                for i in range(n_col)
                for j in range(n_row)
            ],
            0,
        )

        n_dim = G.n_dim
        loc_proposal = torch.zeros(n_dim).to(G.device)
        scale_proposal = torch.ones(n_dim).to(G.device)

        proposal = IndependentNormal(
            device=device, dim=n_dim, loc=loc_proposal, scale=scale_proposal
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

        evols = dict()

        z_0 = proposal.sample((config.batch_size,))

        batch_size = config.batch_size
        target_sample = X_train

        gan_sample = G(z_0).detach().cpu()

        evol = defaultdict(list)
        for i, gs in enumerate(gan_sample.reshape(config.n_eval, -1, dim)):
            gs = torch.FloatTensor(scaler.inverse_transform(gs))
            evolution = Evolution(
                target_sample[gs.shape[0] * i : (i + 1) * gs.shape[0]],
                locs=locs,
                sigma=config.sigma,
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
        names.append("GAN")
        colors.append("black")
        samples.append(gan_sample.reshape(-1, dim))
        print(evol)

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            colors.append(info.color)
            names.append(method_name)
            mcmc_class = info.mcmc_class
            mcmc_class = eval(mcmc_class)
            mcmc = mcmc_class(**info.params.dict, dim=G.n_dim)

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
                    **config.methods.dict["ULA"]["params"], dim=G.n_dim
                )(sample[-1], target, proposal, 2)

            sample = torch.stack(sample, 0)  # len x batch x dim
            sample = sample[-config.n_chunks * info.every :]
            sample = torch.stack(torch.split(sample, config.every, 0), 0)
            zs_gen = sample.permute(2, 0, 1, 3)

            Xs_gen = G(zs_gen).detach().cpu().numpy()
            if scaler:
                Xs_gen = scaler.inverse_transform(Xs_gen)

            if config.multistart:
                Xs_gen = Xs_gen.transpose(2, 1, 0, 3)
                Xs_gen = Xs_gen.reshape(config.n_eval, 1, -1, G.n_dim)
                samples.append(Xs_gen.reshape(-1, dim))
            else:
                samples.append(Xs_gen[0].reshape(-1, G.n_dim))

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
                    sigma=config.sigma,
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
            name = "gan_gaussians_2d_metrics"
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

        # plot_chain_metrics(
        #     evols,
        #     colors=colors,
        #     every=info.every,
        #     savepath=Path(config.figpath, "gan_gaussians_2d.pdf"),
        # )

        for name, sample in zip(names, samples):
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.scatter(
                X_train[:1000, 0],
                X_train[:1000, 1],
                alpha=0.3,
                s=100,
                color="grey",
            )
            ax.scatter(
                sample[-config.trunc_chain_len :, 0],
                sample[-config.trunc_chain_len :, 1],
                alpha=0.1,
                s=100,
                color="red",
            )
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # plt.legend()
            fig.tight_layout()
            plt.savefig(Path(config.figpath, f"gan_gaussians_2d_{name}.pdf"))
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

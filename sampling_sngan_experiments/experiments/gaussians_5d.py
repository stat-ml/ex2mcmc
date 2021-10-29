import argparse
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import ot
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.distributions import (
    GaussianMixture,
    IndependentNormal,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA, ULA
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.metrics import Evolution
from iterative_sir.sampling_utils.visualization import plot_chain_metrics
from iterative_sir.toy_examples_utils import prepare_gaussians


sns.set_theme(style="ticks", palette="deep")
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    return args


def main(config, run=True):
    colors = []

    if run:
        dim = 5
        device = config.device
        coord_limits = 2.0
        sigma = 0.05
        num_gaussian_per_dim = 3
        X_train = prepare_gaussians(
            num_samples_in_cluster=100,
            dim=dim,
            num_gaussian_per_dim=num_gaussian_per_dim,
            coord_limits=coord_limits,
            sigma=sigma,
        )
        X_train = X_train[
            np.random.choice(
                np.arange(X_train.shape[0]), config.train_size, replace=False
            )
        ]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        target_args = edict()
        target_args.device = device
        target_args.num_gauss = num_gaussian_per_dim ** dim

        ###create points
        coords_per_dim = np.linspace(
            -coord_limits, coord_limits, num=num_gaussian_per_dim
        )
        copy_coords = list(np.tile(coords_per_dim, (dim, 1)))
        centers = np.array(
            np.meshgrid(*copy_coords), dtype=np.float64
        ).T.reshape(-1, dim)

        coef_gaussian = 1.0 / target_args.num_gauss
        target_args.p_gaussians = [
            torch.tensor(coef_gaussian)
        ] * target_args.num_gauss
        locs = torch.stack(
            [
                torch.tensor(centers[i]).to(device)
                for i in range(centers.shape[0])
            ],
            0,
        )
        target_args.locs = locs
        target_args.covs = [
            (sigma ** 2) * torch.eye(dim, dtype=torch.float64).to(device)
        ] * target_args.num_gauss
        target_args.dim = dim
        target = GaussianMixture(**target_args).log_prob

        loc = torch.zeros(dim).to(device)
        scale = torch.ones(dim).to(device)
        proposal_args = edict()
        proposal_args.device = device
        proposal_args.loc = loc
        proposal_args.scale = scale
        proposal = IndependentNormal(**proposal_args, dim=dim)

        evols = dict()

        batch_size = config.batch_size
        target_sample = X_train

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            colors.append(info.color)
            mcmc_class = info.mcmc_class
            mcmc_class = eval(mcmc_class)
            mcmc = mcmc_class(**info.params.dict, dim=dim)

            z_0 = proposal.sample((config.batch_size,))

            if "flow" in info.dict.keys():
                verbose = mcmc.verbose
                mcmc.verbose = False
                flow = RNVP(info.flow.num_flows, dim=dim)

                flow_mcmc = FlowMCMC(
                    target,
                    proposal,
                    flow,
                    mcmc,
                    batch_size=info.flow.batch_size,
                    lr=info.flow.lr,
                )
                flow.train()
                out_samples, nll = flow_mcmc.train(
                    n_steps=info.flow.n_steps,
                )
                assert not torch.isnan(
                    next(flow.parameters())[0, 0],
                ).item()

                flow.eval()
                mcmc.flow = flow
                mcmc.verbose = verbose

            out = mcmc(z_0, target, proposal, info.burn_in)
            if isinstance(out, tuple):
                out = out[0]

            out = mcmc(out[-1], target, proposal, info.n_steps)

            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

            sample = torch.stack(sample, 0).detach().numpy()
            sample = sample[-config.n_chunks * info.every :].reshape(
                config.n_chunks, info.every, batch_size, sample.shape[-1]
            )
            Xs_gen = sample.transpose(2, 0, 1, 3)

            evol = defaultdict(list)
            for X_gen in Xs_gen:
                evolution = Evolution(
                    target_sample,
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
            pickle.dump(
                evols,
                Path(config.respath, "gaussians_5d_metrics.pkl").open("wb"),
            )
    else:
        evols = pickle.load(Path(config.respath).open("rb"))
        evols = {k: evols[k] for k in config.methods.dict.keys()}
        for method_name, info in config.methods.items():
            colors.append(info.color)

    print(evols)

    if "figpath" in config.dict:
        # SMALL_SIZE = 15  # 8
        # MEDIUM_SIZE = 20  # 10
        # BIGGER_SIZE = 20  # 12

        # plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        # plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
        # plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        # plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        # plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        # plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        # plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        SMALL_SIZE = 19  # 8
        MEDIUM_SIZE = 23  # 10
        BIGGER_SIZE = 23  # 12

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
            savepath=Path(config.figpath, "gaussians_2d.pdf"),
        )


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.load(Path(args.config).open("r"), Loader=yaml.FullLoader)
    config = DotConfig(config)

    if args.result_path is not None:
        run = False
        config.respath = args.result_path
    else:
        run = True
    main(config, run)

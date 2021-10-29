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
from iterative_sir.toy_examples_utils import prepare_25gaussian_data


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
    names = []
    samples = []

    if run:
        device = config.device
        X_train, means = prepare_25gaussian_data(
            config.train_size,
            config.sigma,
            config.random_seed,
        )

        scaler = None  # StandardScaler()
        # X_train_std = scaler.fit_transform(X_train)

        dim = 2
        num_gauss = 25
        target_args = edict()
        target_args.num_gauss = num_gauss
        n_col = 5
        n_row = num_gauss // n_col
        s = 1
        # create points
        coef_gaussian = 1.0 / num_gauss
        target_args.p_gaussians = [
            torch.tensor(coef_gaussian),
        ] * target_args.num_gauss
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
        target_args.locs = locs
        target_args.covs = [
            (config.sigma ** 2) * torch.eye(dim).to(device),
        ] * target_args.num_gauss
        target_args.dim = dim
        target = GaussianMixture(device=device, **target_args)

        loc_proposal = torch.zeros(dim).to(device)
        scale_proposal = torch.ones(dim).to(device)
        proposal = IndependentNormal(
            dim=dim,
            device=device,
            loc=loc_proposal,
            scale=scale_proposal,
        )

        evols = dict()

        batch_size = config.batch_size

        target_sample = X_train

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            names.append(method_name)
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
                acc = out[1]
                print(acc)
            else:
                sample = out

            sample = torch.stack(sample, 0).detach().numpy()
            sample = sample[-config.n_chunks * info.every :].reshape(
                config.n_chunks, info.every, batch_size, sample.shape[-1]
            )
            Xs_gen = sample.transpose(2, 0, 1, 3)

            samples.append(Xs_gen[0].reshape(-1, dim))

            evol = defaultdict(list)
            for X_gen in Xs_gen:
                evolution = Evolution(
                    target_sample,
                    locs=locs,
                    target_log_prob=target,
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

            M = ot.dist(Xs_gen.reshape(-1, dim), X_train)
            emd = ot.emd2([], [], M)
            print(emd)

        if "respath" in config.dict:
            pickle.dump(
                evols,
                Path(config.respath, "gaussians_2d_metrics.pkl").open("wb"),
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

        for name, sample in zip(names, samples):
            # print('hi')
            # color = evols['color']
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
                sample[-500:, 0],
                sample[-500:, 1],
                alpha=0.3,
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

    if args.result_path is not None:
        run = False
        config.respath = args.result_path
    else:
        run = True
    main(config, run)

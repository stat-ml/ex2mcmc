import argparse
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.distributions import (  # Gaussian,
    IndependentNormal,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.metrics import ESS, acl_spectrum


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
    dim = config.dim

    target = IndependentNormal(
        dim=dim, scale=torch.logspace(-2, 2, dim), device=config.device
    )
    dim = target.dim
    # metrics = MetricsTracker(fields=["method", "ess", "ess_per_s", "time"])
    names = config.methods.dict.keys()

    if run:
        proposal = IndependentNormal(dim=dim, device=config.device)
        colors = []
        samples = []
        acl_times = []

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            params = info.params  # ['params']
            colors.append(info.color)
            try:
                mcmc_class = eval(info.mcmc_class)
            except KeyError:
                print("Can't understand class")

            params = params.dict
            if "lr" in params:
                params["lr"] = eval(params["lr"])

            mcmc = mcmc_class(**params, dim=dim)
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
                    # start_optim=info.flow.start_optim,
                    jump_tol=1e3,
                )
                flow.train()
                out_samples, nll = flow_mcmc.train(n_steps=info.flow.n_steps)
                assert not torch.isnan(next(flow.parameters())[0, 0]).item()

                flow.eval()
                mcmc.flow = flow
                mcmc.verbose = verbose

            start = proposal.sample([config.batch_size])

            s = time.time()
            out = mcmc(start, target, proposal, n_steps=info.n_steps)
            e = time.time()
            elapsed = e - s
            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

            # ess_arr = []
            sample = torch.stack(sample, 0).detach().cpu().numpy()
            trunc_sample = sample[-config.trunc_chain_len :]
            batch_size = sample.shape[1]
            ess = ESS(
                acl_spectrum(trunc_sample - trunc_sample.mean(0)[None, ...]),
            )
            assert ess.shape[0] == batch_size

            # scale = 1
            # scale = target.scale.sum().item() ** .5
            acl = acl_spectrum(trunc_sample, n=trunc_sample.shape[0]).mean(-1)
            acl_mean = acl.mean(axis=-1)
            acl_std = acl.std(axis=-1, ddof=1)
            # - trunc_sample.mean(0)[None, ...])

            acl_times.append((acl_mean[:-1], acl_std[:-1]))

            print(
                f"Method: {method_name}, ESS: {ess.mean():.4f}, sampling time: {elapsed:.2f}, ESS/s: {ess.mean()*info.n_steps/elapsed:.2f}",
            )

        if "figpath" in config.dict.keys():
            MEDIUM_SIZE = 22  # 10
            BIGGER_SIZE = 22  # 12

            plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
            plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
            plt.rc(
                "axes", labelsize=MEDIUM_SIZE
            )  # fontsize of the x and y labels
            plt.rc(
                "xtick", labelsize=MEDIUM_SIZE
            )  # fontsize of the tick labels
            plt.rc(
                "ytick", labelsize=MEDIUM_SIZE
            )  # fontsize of the tick labels
            plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
            plt.rc(
                "figure", titlesize=BIGGER_SIZE
            )  # fontsize of the figure title

            fig = plt.figure(figsize=(7, 6))
            for name, acl, color in zip(names, acl_times, colors):
                acl_mean, acl_std = acl
                plt.plot(
                    np.arange(len(acl_mean)),
                    acl_mean,
                    label=fr"{name}",
                    color=color,
                    linewidth=3,
                )
                plt.fill_between(
                    np.arange(len(acl_mean)),
                    acl_mean - 1.96 * acl_std,
                    acl_mean + 1.96 * acl_std,
                    color=color,
                    alpha=0.3,
                )
                plt.ylim(0, 1)

            plt.xlabel("Sampling iterations")
            plt.ylabel("Autocorrelation")
            # plt.xscale("log")
            plt.grid()
            plt.legend()
            fig.tight_layout()
            plt.savefig(Path(config.figpath, "ill_cond_acl.pdf"))
            plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.load(Path(args.config).open("r"), Loader=yaml.FullLoader)
    config = DotConfig(config)
    main(config)

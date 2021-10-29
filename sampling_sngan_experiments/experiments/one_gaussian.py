import argparse
import datetime
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC
from iterative_sir.sampling_utils.distributions import (
    IndependentNormal,
    init_independent_normal,
    init_independent_normal_scale,
)
from iterative_sir.sampling_utils.metrics import ESS, acl_spectrum
from iterative_sir.sampling_utils.sir_ais_sampling import (
    run_experiments_gaussians,
)


SMALL_SIZE = 20  # 8
MEDIUM_SIZE = 20  # 10
BIGGER_SIZE = 20  # 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc(
    "axes",
    labelsize=MEDIUM_SIZE,
)  # fontsize of the x and y labels
plt.rc(
    "xtick",
    labelsize=SMALL_SIZE,
)  # fontsize of the tick labels
plt.rc(
    "ytick",
    labelsize=SMALL_SIZE,
)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc(
    "figure",
    titlesize=BIGGER_SIZE,
)  # fontsize of the figure title


def plot_metrics(
    method_metric_dict, colors, dim_arr, scale=1.0, save_dir=None
):
    figs = []
    axs = []
    for _ in range(3):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        figs.append(fig)
        axs.append(ax)

    # fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 4))
    # for (name, metric_dict), color in zip(method_metric_dict.items(), colors):

    axs[0].axhline(
        scale ** 2, label="True", color="black"
    )  # , linewidth = linewidth)
    axs[0].set_xlabel("dim")
    axs[0].set_ylabel("Variance")

    axs[1].axhline(
        0.0, label="True", color="black"
    )  # , linewidth = linewidth)
    axs[1].set_xlabel("dim")
    axs[1].set_ylabel("Mean")

    axs[2].set_xlabel("dim")
    axs[2].set_ylabel("ESS")

    # modes_to_plot = ['mean_var', 'mean_loc', 'ess']
    for (name, metric_dict), color in zip(method_metric_dict.items(), colors):

        for i, m in enumerate(["var", "mean", "ess"]):
            axs[i].plot(
                dim_arr,
                metric_dict[f"{m}_mean"],
                label=name,
                marker="o",
                color=color,
            )
            axs[i].fill_between(
                dim_arr,
                metric_dict[f"{m}_mean"] - 1.96 * metric_dict[f"{m}_std"],
                metric_dict[f"{m}_mean"] + 1.96 * metric_dict[f"{m}_std"],
                color=color,
                alpha=0.3,
            )
    axs[-1].legend()
    for fig, ax, name in zip(figs, axs, ["mean", "var", "ess"]):
        ax.grid()

        fig.tight_layout()

        if save_dir:
            fig.savefig(Path(save_dir, f"one_gauss_{name}.pdf"))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--result_path", type=str)

    args = parser.parse_args()
    return args


def main(config, run=True):
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    dim_arr = [30 * (i + 1) for i in range(10)]

    if run:
        method_metric_dict = defaultdict(lambda: defaultdict(list))
        for dim in dim_arr:
            print("=" * 10, dim, "=" * 10)

            # samples = []
            names = []
            colors = []

            proposal = IndependentNormal(dim=dim, scale=config.scale_proposal)
            target = IndependentNormal(dim=dim, scale=config.scale_target)
            for method_name, info in config.methods.items():
                colors.append(info.color)
                names.append(method_name)
                print(f"========== {method_name} =========== ")
                params = info.params.dict
                try:
                    mcmc_class = eval(info.mcmc_class)
                except KeyError:
                    print("Can't understand class")

                mcmc = mcmc_class(**params, dim=dim)

                start = proposal.sample((config.batch_size,)).to(device)
                out = mcmc(start, target, proposal, n_steps=info.n_steps)

                if isinstance(out, Tuple):
                    out = out[0]

                out = out[-5000:]

                out = torch.stack(out, 0).detach().cpu().numpy()
                var = np.var(out, axis=0, ddof=1).mean(
                    -1
                )  # .mean(axis=0).mean()
                mean = np.mean(out, axis=0).mean(-1)  # .mean(axis=0).mean()
                ess = ESS(acl_spectrum(out, n=150)).mean(-1)

                method_metric_dict[method_name]["ess_mean"].append(
                    np.mean(ess)
                )
                method_metric_dict[method_name]["ess_std"].append(
                    np.std(ess, ddof=1)
                )
                method_metric_dict[method_name]["mean_mean"].append(
                    np.mean(mean)
                )
                method_metric_dict[method_name]["mean_std"].append(
                    np.std(mean, ddof=1)
                )
                method_metric_dict[method_name]["var_mean"].append(
                    np.mean(var)
                )
                method_metric_dict[method_name]["var_std"].append(
                    np.std(var, ddof=1)
                )

                for m, list_val in method_metric_dict[method_name].items():
                    print(m, list_val[-1])

        for method_name in method_metric_dict.keys():
            for m, list_val in method_metric_dict[method_name].items():
                method_metric_dict[method_name][m] = np.array(
                    method_metric_dict[method_name][m]
                )

        sub = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")

        if "respath" in config.dict:
            method_metric_dict = dict(method_metric_dict)
            resdir = Path(config.respath)  # , "")
            resdir.mkdir(parents=True, exist_ok=True)
            respath = Path(resdir, f"{sub}.npy")
            pickle.dump(method_metric_dict, respath.open("wb"))
    else:
        method_metric_dict = pickle.load(Path(config.respath).open("rb"))
        colors = []
        for method_name, info in config.methods.items():
            colors.append(info.color)

    if "figpath" in config.dict:
        plot_metrics(
            method_metric_dict,
            colors,
            dim_arr,
            scale=config.scale_target,
            save_dir=config.figpath,
        )
    # if "figpath" in config.dict:
    #     plt.savefig(Path(config.figpath, "one_gaussian.pdf"))


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

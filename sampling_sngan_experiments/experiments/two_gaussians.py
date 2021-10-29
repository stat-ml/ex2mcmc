import argparse
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from utils import DotConfig, random_seed

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.distributions import (
    Distribution,
    GaussianMixture,
    IndependentNormal,
    init_independent_normal,
    init_independent_normal_scale,
)
from iterative_sir.sampling_utils.ebm_sampling import MALA
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.metrics import ESS, Evolution, acl_spectrum


sns.set_theme(style="ticks", palette="deep")


def define_target(
    loc_1_target=-3,
    loc_2_target=3,
    scale_target=1,
    dim=100,
    device="cpu",
):
    target_args = edict()
    target_args.device = device
    target_args.num_gauss = 2

    coef_gaussian = 1.0 / target_args.num_gauss
    target_args.p_gaussians = [
        torch.tensor(coef_gaussian),
    ] * target_args.num_gauss
    locs = torch.stack(
        [
            loc_1_target * torch.ones(dim, dtype=torch.float64).to(device),
            loc_2_target * torch.ones(dim, dtype=torch.float64).to(device),
        ],
        0,
    )
    # locs_numpy = locs.cpu().numpy()
    target_args.locs = locs
    target_args.covs = [
        (scale_target ** 2) * torch.eye(dim, dtype=torch.float64).to(device),
    ] * target_args.num_gauss
    target_args.dim = dim
    target = GaussianMixture(**target_args)
    return target


def compute_metrics(sample, target, trunc_chain_len=None):
    if trunc_chain_len is not None:
        trunc_sample = sample[(-trunc_chain_len - 1) : -1]
    else:
        trunc_sample = sample
    if isinstance(sample, list):
        sample = torch.stack(sample, axis=0).detach().cpu().numpy()
        trunc_sample = torch.stack(trunc_sample, axis=0).detach().cpu()
    chain_len, batch_size, dim = sample.shape

    locs = target.locs
    evolution = Evolution(None, locs=locs.cpu(), sigma=target.covs[0][0, 0])

    result_np = trunc_sample.detach().cpu().numpy()

    modes_var_arr = []
    modes_mean_arr = []
    hqr_arr = []
    jsd_arr = []
    ess_arr = []
    means_est_1 = torch.zeros(dim)
    means_est_2 = torch.zeros(dim)
    num_found_1_mode = 0
    num_found_2_mode = 0
    num_found_both_modes = 0

    ess = ESS(
        acl_spectrum(
            (trunc_sample - trunc_sample.mean(0)[None, ...])
            .detach()
            .cpu()
            .numpy(),
        ),
    ).mean()

    for i in range(batch_size):
        X_gen = trunc_sample[:, i, :]

        assignment = Evolution.make_assignment(
            X_gen,
            evolution.locs,
            evolution.sigma,
        )
        mode_var = Evolution.compute_mode_std(X_gen, assignment)[0].item() ** 2
        modes_mean, found_modes_ind = Evolution.compute_mode_mean(
            X_gen,
            assignment,
        )

        if 0 in found_modes_ind and 1 in found_modes_ind:
            num_found_both_modes += 1
        if 0 in found_modes_ind:
            num_found_1_mode += 1
            means_est_1 += modes_mean[0]
        if 1 in found_modes_ind:
            num_found_2_mode += 1
            means_est_2 += modes_mean[1]

        hqr = Evolution.compute_high_quality_rate(assignment).item()
        jsd = Evolution.compute_jsd(assignment).item()

        modes_var_arr.append(mode_var)
        hqr_arr.append(hqr)
        jsd_arr.append(jsd)

    jsd = np.array(jsd_arr).mean()
    modes_var = np.array(modes_var_arr).mean()
    hqr = np.array(hqr_arr).mean()
    # ess = np.mean(ess_arr)
    if num_found_1_mode == 0:
        print(
            "Unfortunalely, no points were assigned to 1st mode, default estimation - zero",
        )
        modes_mean_1_result = np.nan  # 0.0
    else:
        modes_mean_1_result = (means_est_1 / num_found_1_mode).mean().item()
    if num_found_2_mode == 0:
        print(
            "Unfortunalely, no points were assigned to 2nd mode, default estimation - zero",
        )
        modes_mean_2_result = np.nan  # 0.0
    else:
        modes_mean_2_result = (means_est_2 / num_found_2_mode).mean().item()
    if num_found_1_mode == 0 and num_found_2_mode == 0:
        modes_mean_1_result = modes_mean_2_result = trunc_sample.mean().item()

    result = dict(
        jsd=jsd,
        modes_var=modes_var,
        hqr=hqr,
        mode1_mean=modes_mean_1_result,
        mode2_mean=modes_mean_2_result,
        fraction_found2_modes=num_found_both_modes / batch_size,
        fraction_found1_mode=(
            num_found_1_mode + num_found_2_mode - 2 * num_found_both_modes
        )
        / batch_size,
        ess=ess,
    )
    return result


def plot_metrics(
    dims, found_both, ess, ess_per_sec, hqr, colors=None, savedir=None
):
    SMALL_SIZE = 18  # 8
    MEDIUM_SIZE = 20  # 10
    BIGGER_SIZE = 20  # 12

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
    figs = []
    axs = []
    for _ in range(4):
        fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
        figs.append(fig)
        axs.append(ax)

    for i, (method_name, arr) in enumerate(found_both.items()):
        if colors is not None:
            color = colors[i]
            axs[0].plot(dims, arr, label=method_name, marker="o", color=color)
        else:
            axs[0].plot(dims, arr, label=method_name, marker="o")
    axs[0].set_xlabel("dim")
    axs[0].set_ylabel("captured 2 modes")
    axs[0].grid()
    # axs[0].legend()

    for i, (method_name, arr) in enumerate(ess.items()):
        if colors is not None:
            color = colors[i]
            axs[1].plot(dims, arr, label=method_name, marker="o", color=color)
        else:
            axs[1].plot(dims, arr, label=method_name, marker="o")
    axs[1].set_xlabel("dim")
    axs[1].set_ylabel("ESS")
    axs[1].grid()
    # axs[1].legend()

    for i, (method_name, arr) in enumerate(ess_per_sec.items()):
        if colors is not None:
            color = colors[i]
            axs[2].plot(dims, arr, label=method_name, marker="o", color=color)
        else:
            axs[2].plot(dims, arr, label=method_name, marker="o")
    axs[2].set_xlabel("dim")
    axs[2].set_ylabel("ESS/s")
    axs[2].grid()
    # axs[2].legend()

    for i, (method_name, arr) in enumerate(hqr.items()):
        if colors is not None:
            color = colors[i]
            axs[3].plot(dims, arr, label=method_name, marker="o", color=color)
        else:
            axs[3].plot(dims, arr, label=method_name, marker="o")
    axs[3].set_xlabel("dim")
    axs[3].set_ylabel("HQR")
    axs[3].grid()
    axs[3].legend()

    for ax, fig, name in zip(
        axs, figs, ["captured", "ESS", "ESS_per_sec", "HQR"]
    ):
        fig.tight_layout()
        fig.savefig(Path(savedir, f"2_gauss_{name}.pdf"))


# return fig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--result_path")
    args = parser.parse_args()

    return args


def main(config, run=True):
    device = torch.device(config.device)

    args = edict()
    args.loc_1_target = -1.5
    args.loc_2_target = 1.5
    args.scale_target = 1.0
    args.scale_proposal = 2
    args.loc_proposal = 0

    args.dim = np.arange(10, 30, 2)
    args.batch_size = 200

    found_both = defaultdict(list)
    ess = defaultdict(list)
    sampling_time = defaultdict(list)
    ess_per_sec = defaultdict(list)
    hqr_dict = defaultdict(list)

    for dim in args.dim:
        print(f"dim = {dim}")
        target = define_target(
            args.loc_1_target,
            args.loc_2_target,
            args.scale_target,
            dim,
            device=device,
        )  # .log_prob
        proposal = init_independent_normal(
            args.scale_proposal,
            dim,
            device,
            args.loc_proposal,
        )

        colors = []
        for method_name, info in config.methods.items():
            color = info.color
            colors.append(color)
            print(f"============ {method_name} =============")
            mcmc_class = eval(info.mcmc_class)
            mcmc = mcmc_class(**info.params.dict, dim=dim)

            start = proposal.sample([args.batch_size])

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

            s = time.time()
            out = mcmc(start, target, proposal, n_steps=info.n_steps)
            e = time.time()
            elapsed = e - s  # / 60
            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

            trunc_chain_len = int(0.9 * len(sample))
            result = compute_metrics(
                sample,
                target,
                trunc_chain_len=trunc_chain_len,
            )
            print(method_name, result)
            print(f"Elapsed: {elapsed:.2f} s")

            found_both[method_name].append(result["fraction_found2_modes"])
            ess[method_name].append(result["ess"])
            sampling_time[method_name].append(elapsed)
            ess_per_sec[method_name].append(
                result["ess"] * trunc_chain_len / elapsed,
            )
            hqr_dict[method_name].append(result["hqr"])

    if "figpath" in config.dict:
        plot_metrics(
            args.dim,
            found_both,
            ess,
            ess_per_sec,
            hqr_dict,
            colors=colors,
            savedir=Path(config.figpath),
        )
        # plt.savefig(Path(config.figpath, "2_gaussians.pdf"))


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

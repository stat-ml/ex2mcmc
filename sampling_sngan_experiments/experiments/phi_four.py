import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from pyro.ops.stats import autocorrelation
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import DotConfig, random_seed

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.adaptive_sir_loss import MixKLLoss
from iterative_sir.sampling_utils.distributions import (
    IndependentNormal,
    PhiFour,
)
from iterative_sir.sampling_utils.flows import RNVP, RealNVP_MLP
from iterative_sir.sampling_utils.metrics import ESS, acl_spectrum


sns.set_theme(style="ticks", palette="deep")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--result_path", type=str)
    # parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))

    config = parser.parse_args()
    return config


def construct_init(n_init, dim, target, proposal, ratio_pos_init=0.5):
    if ratio_pos_init == -1:
        return proposal.sample(n_init)
    else:
        x_init = torch.ones(n_init, dim)  # , device=device)
        n_pos = int(ratio_pos_init * n_init)  # config.batch_size)
        if target.tilt is None:
            x_init[n_pos:, :] = -1
        else:
            n_tilt = int(target.tilt["val"] * dim)
            x_init[n_pos:, n_tilt:] = -1
            x_init[:n_pos, : (dim - n_tilt)] = -1
        return x_init


def main(config, run=True):
    device = torch.device(config.device)

    if config.tilt_value is not None:
        tilt = {"val": config.tilt_value, "lambda": config.tilt_lambda}
    else:
        tilt = None

    target = PhiFour(
        config.a_coupling,
        config.b_field,
        config.N,
        dim_phys=config.dim_phys,
        beta=config.beta,
        tilt=tilt,
    )

    dim = config.N

    prior_arg = {
        "type": config.prior_type,
        "alpha": config.a_coupling,
        "beta": config.beta,
    }
    beta_prior = prior_arg["beta"]
    coef = prior_arg["alpha"] * dim
    prec = torch.eye(dim) * (3 * coef + 1 / coef)
    prec -= coef * torch.triu(
        torch.triu(torch.ones_like(prec), diagonal=-1).T,
        diagonal=-1,
    )
    prec = prior_arg["beta"] * prec
    prior_prec = prec.to(device)
    prior_log_det = -torch.logdet(prec)
    proposal = MultivariateNormal(
        torch.zeros((dim,), device=device),
        precision_matrix=prior_prec,
    )

    class DistWrapper:
        def __init__(self, dist, prior_prec, prior_log_det, dim):
            self.dist = dist
            self.prior_prec = prior_prec
            self.prior_log_det = prior_log_det
            self.dim = dim

        def __call__(self, z):
            prior_ll = -0.5 * torch.einsum(
                "ki,ij,kj->k",
                z,
                self.prior_prec,
                z,
            )
            prior_ll -= 0.5 * (
                self.dim * np.log(2 * np.pi) + self.prior_log_det
            )
            return prior_ll

        def sample(self, n):
            return self.dist.sample(n)

    proposal = DistWrapper(proposal, prior_prec, prior_log_det, dim)

    short_names = []
    colors = []
    flow_samples = []
    mixing_samples = []
    neg_log_likelihood = []
    acl_times = []
    for method_name, info in config.methods.items():
        short_names.append(info.short_name)
        colors.append(info.color)

    for method_name, info in config.methods.items():
        print(f"========== {method_name} ===========")
        mcmc_class = eval(info.mcmc_class)
        mcmc = mcmc_class(**info.params.dict, dim=dim, beta=config.beta)

        if "flow" in info.dict.keys():
            # flow = RNVP(info.flow.num_flows, dim=dim, init_weight_scale=1e-6).to(device)
            flow = RealNVP_MLP(
                dim,
                config.depth_blocks,
                1,
                hidden_dim=config.hidden_dim,
                init_weight_scale=1e-6,
                prior_arg=prior_arg,
            ).to(device)
            x_init = construct_init(
                config.batch_size, dim, target, proposal, config.ratio_pos_init
            ).to(device)

            if run:
                # burn-in
                if (
                    "burn_in_steps" in info.flow.dict.keys()
                ):  # MALA(**info.params.dict, dim=dim, beta=config.beta)
                    burn_in_sample = mcmc(
                        x_init,
                        target,
                        proposal,
                        flow=None,  # flow,
                        n_steps=info.flow.burn_in_steps,
                        verbose=True,
                    )
                    if isinstance(burn_in_sample, Tuple):
                        burn_in_sample = burn_in_sample[0]
                    x_init = burn_in_sample[-1]

                    if "figpath" in config.dict.keys():
                        for i in range(x_init.shape[0]):
                            plt.plot(
                                x_init[i, :],
                                alpha=0.2,
                                c="b",
                            )

                        plt.savefig(
                            Path(config.figpath, "allen_cahn_burn_in.pdf")
                        )
                        plt.close()
                else:
                    burn_in_sample = []

                verbose = mcmc.verbose
                mcmc.verbose = False
                loss = MixKLLoss(target, proposal, flow, gamma=0.0)
                flow_mcmc = FlowMCMC(
                    target,
                    proposal,
                    flow,
                    mcmc,
                    batch_size=config.batch_size,  # info.flow.batch_size,
                    lr=info.flow.lr,
                    loss=loss,
                )
                flow.train()
                out_samples, nll = flow_mcmc.train(
                    n_steps=info.flow.n_steps,
                    init_points=x_init,
                    start_optim=info.flow.start_optim,
                    # alpha=1.0,
                )
                if "respath" in config.dict.keys():
                    sub = datetime.now().strftime("%d-%m-%Y_%H:%M")
                    Path(config.respath).mkdir(exist_ok=True, parents=True)
                    # method_name_ = '_'.join(method_name.split('/'))
                    torch.save(
                        {
                            "flow_state_dict": flow.state_dict(),
                        },
                        Path(config.respath, f"{info.short_name}_{sub}.pth"),
                    )

                # scale = 1
                X = np.stack(burn_in_sample + out_samples, 0)
                n = X.shape[0]
                # acl_time = acl_spectrum(X - X.mean(0)[None, ...], n=n, scale=scale).mean(-1).mean(-1)
                acl_time = acl_spectrum(X[:, :10, :], n=n).mean(-1).mean(-1)
                print("acl computed")
                if "respath" in config.dict.keys():
                    # sub = datetime.now().strftime("%d-%m-%Y_%H:%M")
                    # Path(config.respath).mkdir(exist_ok=True, parents=True)
                    np.save(
                        Path(config.respath, f"{info.short_name}_{sub}"),
                        acl_time,
                    )

                # del acl_time
                # del out_samples
                # del X

                mcmc.verbose = verbose

                neg_log_likelihood.append(nll)

            else:
                model_path = sorted(
                    Path(config.respath).glob(f"{info.short_name}_0*.pth")
                )[-1]
                flow.load_state_dict(torch.load(model_path)["flow_state_dict"])
                acl_time = np.load(
                    sorted(
                        Path(config.respath).glob(f"{info.short_name}_0*.npy")
                    )[-1]
                )
                print(
                    sorted(
                        Path(config.respath).glob(f"{info.short_name}*.npy")
                    )[-1]
                )
                print(info.short_name, acl_time.shape)

            acl_times.append(acl_time[1:-50])

            flow.eval()
            mcmc.flow = flow

            prop = proposal.sample((100,))
            x_gen = flow.forward(prop)[0]
            flow_samples.append(x_gen)

            # prop = proposal.sample((10,))
            prop = torch.zeros(1, dim)
            x_gen = mcmc(
                prop, target, proposal, mala_steps=1, flow=flow, n_steps=21
            )
            # x_gen = mcmc(prop, target, proposal, flow=None, n_steps=100)
            if isinstance(x_gen, Tuple):
                x_gen = x_gen[0]
            # x_gen = x_gen[-10:]
            # mixing_samples.append(x_gen[-1])
            mixing_samples.append(torch.stack(x_gen[1:], 0).reshape(-1, dim))

    if "figpath" in config.dict.keys():
        MEDIUM_SIZE = 23  # 10
        BIGGER_SIZE = 23  # 12

        plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        names = config.methods.dict.keys()
        # fig, axs = plt.subplots(ncols=len(names), figsize=(6 * len(names), 5))
        for sample, name, short_name in zip(flow_samples, names, short_names):
            fig = plt.figure(figsize=(7, 7))
            for i in range(sample.shape[0]):
                plt.plot(
                    sample[i, :].detach().cpu(),
                    alpha=0.05,
                    c="b",
                    linewidth=3,
                )
            # plt.title(fr"{name}")
            fig.tight_layout()
            plt.savefig(
                Path(config.figpath, f"allen_cahn_flow_{short_name}.pdf")
            )
            plt.close()

        if run:
            fig = plt.figure(figsize=(7, 7))
            for name, nlls in zip(names, neg_log_likelihood):
                plt.plot(np.arange(len(nlls)), nlls, label=fr"{name}")

            plt.xlabel("Burn-in and training iterations")
            plt.ylabel("- Log Likelihood + Const")
            plt.xscale("log")
            plt.grid()
            plt.legend()
            fig.tight_layout()
            plt.savefig(Path(config.figpath, "allen_cahn_nll.pdf"))
            plt.close()

        fig = plt.figure(figsize=(6.3, 6))
        for name, acl, color in zip(names, acl_times, colors):
            plt.plot(
                np.arange(len(acl)),
                acl,
                label=fr"{name}",
                color=color,
                linewidth=5,
            )

        # plt.box_asp
        plt.xlabel("Burn-in and training iterations")  # , loc='left')
        axes = plt.gca()
        axes.xaxis.set_label_coords(1.1, -2)
        axes.set_box_aspect(1.0)
        plt.ylabel("Autocorrelation")
        plt.xscale("log")
        plt.grid()
        plt.legend(fontsize=26)
        fig.tight_layout()
        plt.savefig(Path(config.figpath, "allen_cahn_acl.pdf"))
        plt.close()

        # fig, axs = plt.subplots(ncols=len(names), figsize=(6 * len(names), 5))
        for sample, name, short_name in zip(
            mixing_samples, names, short_names
        ):
            fig = plt.figure(figsize=(7, 7))
            for i in range(len(sample)):
                plt.plot(
                    sample[i].detach().cpu(),
                    alpha=0.3,
                    c="g",
                    linewidth=3,
                )
            # plt.title(fr"{name}")
            fig.tight_layout()
            axes = plt.gca()
            axes.set_box_aspect(1.0)
            plt.yticks([-1, 0, 1])
            plt.savefig(
                Path(config.figpath, f"allen_cahn_mixing_{short_name}.pdf")
            )
            plt.close()

        # x_gen = out_samples[-1]
        # for i in range(x_gen.shape[0]):
        #     plt.plot(x_gen[i, :].detach().cpu(), alpha=0.2, c='b')

        # plt.savefig(Path(config.figpath, "allen_cahn_mcmc_out.pdf"))
        # plt.close()


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

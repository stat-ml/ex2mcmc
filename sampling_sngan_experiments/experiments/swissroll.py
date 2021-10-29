import argparse
import pickle
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path

import jax
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import DotConfig

from iterative_sir.sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from iterative_sir.sampling_utils.distributions import IndependentNormal
from iterative_sir.sampling_utils.ebm_sampling import MALA, ULA, gan_energy
from iterative_sir.sampling_utils.flows import RNVP
from iterative_sir.sampling_utils.metrics import Evolution, acl_spectrum
from iterative_sir.sampling_utils.total_variation import (
    average_total_variation,
)
from iterative_sir.sampling_utils.visualization import plot_chain_metrics
from iterative_sir.toy_examples_utils import prepare_swissroll_data
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
    acl_times = []
    names = []
    samples = []
    evols = dict()

    if run:
        device = config.device

        G = load_model(model_config.generator, device=device).eval()
        D = load_model(model_config.discriminator, device=device).eval()

        X_train = prepare_swissroll_data(
            config.train_size,
        )

        # scaler = StandardScaler()
        # X_train_std = scaler.fit_transform(X_train)
        scaler = None

        dim = G.n_dim
        loc = torch.zeros(dim).to(G.device)
        scale = torch.ones(dim).to(G.device)
        normalize_to_0_1 = True
        log_prob = True

        proposal = IndependentNormal(
            dim=dim, device=device, loc=loc, scale=scale
        )

        target = partial(
            gan_energy,
            generator=G,
            discriminator=D,
            proposal=proposal,
            normalize_to_0_1=normalize_to_0_1,
            log_prob=log_prob,
        )

        z_0 = proposal.sample((config.batch_size,))

        batch_size = config.batch_size
        target_sample = X_train

        gan_sample = G(z_0).detach().cpu()
        evol = defaultdict(list)
        for i, gs in enumerate(gan_sample.reshape(config.n_eval, -1, dim)):
            tv = []
            evolution = Evolution(
                target_sample[gs.shape[0] * i : (i + 1) * gs.shape[0]],
            )
            evolution.invoke(gs)
            key = jax.random.PRNGKey(0)
            tracker = average_total_variation(
                key,
                X_train,
                gs.numpy(),
                n_steps=10,  # n_steps,
                n_samples=gs.shape[0],  # n_samples,
            )

            tv.append(tracker.mean())
            evol_ = evolution.as_dict()
            for k, v in evol_.items():
                evol[k].append(v)
            evol["tv"].append(tv)
        for k, v in evol.items():
            evol[k] = (
                np.mean(np.array(v), 0),
                np.std(np.array(v), 0, ddof=1) / np.sqrt(batch_size),
            )
        evols["GAN"] = evol
        colors.append("black")
        names.append("GAN")
        samples.append(gan_sample.reshape(-1, dim))
        print(evol)

        for method_name, info in config.methods.items():
            print(f"========= {method_name} ========== ")
            colors.append(info.color)
            names.append(method_name)
            mcmc_class = info.mcmc_class
            mcmc_class = eval(mcmc_class)
            mcmc = mcmc_class(**info.params.dict, dim=dim)

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

                # if "figpath" in config.dict:
                #     fig = plot_learned_density(
                #         flow,
                #         proposal,
                #         xlim=target.xlim,
                #         ylim=target.ylim,
                #     )
                #     plt.savefig(
                #         Path(
                #             config.figpath,
                #             f"flow_{config.dist}_{dim}.pdf",
                #         ),
                #     )

            # z_0 = proposal.sample((config.batch_size,))
            out = mcmc(z_0, target, proposal, info.burn_in)
            if isinstance(out, tuple):
                out = out[0]

            out = mcmc(out[-1], target, proposal, info.n_steps)

            if isinstance(out, tuple):
                sample = out[0]
            else:
                sample = out

            if config.multistart:
                sample = ULA(**config.methods.dict["ULA"]["params"], dim=dim)(
                    sample[-1], target, proposal, 2
                )

            sample = torch.stack(sample, 0)  # len x batch x dim
            sample = sample[-config.n_chunks * info.every :]
            sample = torch.stack(torch.split(sample, config.every, 0), 0)
            zs_gen = sample.permute(2, 0, 1, 3)

            Xs_gen = G(zs_gen).detach().cpu().numpy()
            if scaler:
                Xs_gen = scaler.inverse_transform(Xs_gen)

            if config.multistart:
                Xs_gen = Xs_gen.transpose(2, 1, 0, 3)
                Xs_gen = Xs_gen.reshape(config.n_eval, 1, -1, dim)
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

                tv = []
                evolution = Evolution(
                    tar,  # [:X_gen.shape[1]],
                    # locs=locs,
                    # target_log_prob=target,
                    # sigma=sigma,
                    scaler=scaler,
                )
                for chunk in X_gen:
                    evolution.invoke(torch.FloatTensor(chunk))
                    key = jax.random.PRNGKey(0)
                    tracker = average_total_variation(
                        key,
                        X_train,
                        chunk,
                        n_steps=10,  # n_steps,
                        n_samples=chunk.shape[0],  # n_samples,
                    )

                    tv.append(tracker.mean())
                    # evolution["tv_mean"] = tracker.mean()
                    # evolution["tv_conf_sigma"] = tracker.std_of_mean()

                evol_ = evolution.as_dict()
                for k, v in evol_.items():
                    evol[k].append(v)
                evol["tv"].append(tv)

            for k, v in evol.items():
                evol[k] = (
                    np.mean(np.array(v), 0),
                    np.std(np.array(v), 0, ddof=1) / np.sqrt(batch_size),
                )
            evols[method_name] = evol
            print(evol)

            # out = mcmc(z_0, target, proposal, info.n_steps)

            # if isinstance(out, tuple):
            #     sample = out[0]
            # else:
            #     sample = out

            # sample = torch.stack(sample, 0).detach().numpy()
            # sample = sample[-config.n_chunks * config.every :].reshape(
            #     (config.n_chunks, batch_size, -1, sample.shape[-1]),
            # )
            # zs_gen = sample.reshape(
            #     batch_size,
            #     config.n_chunks,
            #     -1,
            #     sample.shape[-1],
            # )

            # Xs_gen = G(torch.FloatTensor(zs_gen).to(device)).detach().cpu().numpy()
            # if scaler is not None:
            #     Xs_gen = scaler.inverse_transform(Xs_gen.reshape(-1, Xs_gen.shape[-1])).reshape(Xs_gen.shape)

            # sample = Xs_gen.reshape(-1, batch_size, G.n_dim)

            # acl = acl_spectrum(sample, n=sample.shape[0]).mean(-1)
            # acl_mean = acl.mean(axis=-1)
            # acl_std = acl.std(axis=-1, ddof=1)
            # acl_times.append((acl_mean[:-1], acl_std[:-1]))

            # samples.append(sample[:, 0])

            # evol = defaultdict(list)
            # for X_gen in Xs_gen:
            #     evolution = Evolution(
            #         target_sample,
            #         #target_log_prob=target,
            #         #sigma=config.sigma,
            #         #scaler=scaler,
            #     )
            #     tv = []
            #     for chunk in X_gen:
            #         evolution.invoke(torch.FloatTensor(chunk))
            #         key = jax.random.PRNGKey(0)
            #         tracker = average_total_variation(
            #             key,
            #             X_train,
            #             chunk,
            #             n_steps=10, #n_steps,
            #             n_samples=chunk.shape[0], #n_samples,
            #         )

            #         tv.append(tracker.mean())
            #         #evolution["tv_mean"] = tracker.mean()
            #         #evolution["tv_conf_sigma"] = tracker.std_of_mean()

            #     evol_ = evolution.as_dict()
            #     for k, v in evol_.items():
            #         evol[k].append(v)
            #     evol['tv'].append(tv)

            # for k, v in evol.items():
            #     evol[k] = (
            #         np.mean(np.array(v), 0),
            #         np.std(np.array(v), 0, ddof=1) / np.sqrt(batch_size),
            #     )
            # evols[method_name] = evol

        if "respath" in config.dict:
            name = "gan_swissroll_metrics"
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

    if "figpath" in config.dict:
        # SMALL_SIZE = 15  # 8
        # MEDIUM_SIZE = 20  # 10
        # BIGGER_SIZE = 20  # 12
        # #mpl.rcParams["mathtext.rm"]

        # plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
        # plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
        # plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        # plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        # plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        # plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
        # plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # # plt.rc("lines", lw=3)  # fontsize of the figure title
        # # plt.rc("lines", markersize=7)  # fontsize of the figure title

        # plot_chain_metrics(
        #     evols,
        #     keys=['emd'],
        #     colors=colors,
        #     every=config.every,
        #     savepath=Path(config.figpath, "gan_swissroll.pdf"),
        # )

        # fig = plt.figure(figsize=(6, 6))
        # for name, acl, color in zip(names, acl_times, colors):
        #     acl_mean, acl_std = acl
        #     plt.plot(
        #         np.arange(len(acl_mean)),
        #         acl_mean,
        #         label=fr"{name}",
        #         color=color,
        #         linewidth=3,
        #     )

        # plt.xlabel("Iterations")
        # plt.ylabel("Autocorrelation")
        # # plt.xscale("log")
        # plt.grid()
        # plt.legend()
        # fig.tight_layout()
        # plt.savefig(Path(config.figpath, "swissroll_acl.pdf"))
        # plt.close()

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
            plt.savefig(Path(config.figpath, f"swissroll_{name}.pdf"))
            plt.close()

        # fig = plt.figure(figsize=(6, 6))
        # for name in evols.keys():
        #     tv_mean, tv_std = evols[name]['tv']
        #     color = evols[name]['color']
        #     plt.plot(
        #         config.every * np.arange(1, len(tv_mean)+1),
        #         tv_mean,
        #         label=fr"{name}",
        #         color=color,
        #         marker="o",
        #         #linewidth=3,
        #         #markersize=6,
        #     )
        #     plt.fill_between(
        #         config.every * np.arange(1, len(tv_mean)+1),
        #         tv_mean - 1.96 * tv_std,
        #         tv_mean + 1.96 * tv_std,
        #         color=color,
        #         alpha=0.3)
        # plt.axhline(y=0., color='black', linestyle='-', label='real')
        # plt.title('Sliced TV')
        # plt.xlabel("Iterations")
        # plt.ylabel("Sliced TV")
        # #plt.xticks(info.every*np.arange(len(tv_mean)))
        # # plt.xscale("log")
        # plt.grid()
        # plt.legend()
        # fig.tight_layout()
        # plt.savefig(Path(config.figpath, f"swissroll_tv.pdf"))
        # plt.close()


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

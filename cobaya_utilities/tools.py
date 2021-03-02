import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def print_chains_size(mcmc_samples, tablefmt="html"):
    """Print MCMC sample size given a set of directories

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value.
    tablefmt: str
      the format of the table (default: html)
    """

    nchains = {}
    for k, v in mcmc_samples.items():
        files = sorted(glob.glob(os.path.join(v, "mcmc.?.txt")))
        nchains[k] = [sum(1 for line in open(f)) for f in files]
        nchains[k] += [sum(nchains[k])]

    from tabulate import tabulate

    return tabulate(
        [(k, *v) for k, v in nchains.items()],
        headers=["mcmc {}".format(i) for i in range(1, 5)] + ["total"],
        tablefmt=tablefmt,
    )


def plot_chains(mcmc_dir, params, title=None, ncol=None):
    """Plot MCMC sample evolution

    Parameters
    ----------
    mcmc_dir: str
      a base directory holding the MCMC samples
    params: list
      a list of parameter names
    title: str
      a title for the figure
    ncol: int
      the number of columns within the plot
    """
    ncol = ncol if ncol is not None else len(params)
    nrow = len(params) // ncol + 1 if ncol is not None else 1
    fig = plt.figure(figsize=(15, 2 * nrow))
    if title is not None:
        fig.suptitle(title)
    ax = [plt.subplot(nrow, ncol, i + 1) for i in range(len(params))]

    # Loop over files independently
    files = sorted(glob.glob(os.path.join(mcmc_dir, "mcmc.?.txt")))
    for f in files:
        from getdist import loadMCSamples

        sample = loadMCSamples(f[:-4], no_cache=True)
        color = "C{}".format(f.split(".")[-2])

        # Get param lookup table
        lookup = {
            par.name: dict(pos=i, label=par.label)
            for i, par in enumerate(sample.getParamNames().names)
        }
        for i, p in enumerate(params):
            ax[i].set_ylabel(r"${}$".format(lookup[p].get("label")))
            ax[i].plot(sample.samples[:, lookup[p].get("pos")], alpha=0.75, color=color)
    plt.tight_layout()


def plot_progress(mcmc_samples):
    """Plot Gelman R-1 parameter and acceptance rate

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value.
    """

    nrow = (len(mcmc_samples) + 1) // 2
    ncol = 2
    fig, ax = plt.subplots(2 * nrow, ncol, figsize=(15, 10), sharex=True)

    for i, (k, v) in enumerate(mcmc_samples.items()):
        files = sorted(glob.glob(os.path.join(v, "mcmc.?.progress")))
        for f in files:
            cols = [a.strip() for a in open(f).readline().lstrip("#").split()]
            df = pd.read_csv(
                f, names=cols, comment="#", sep=" ", skipinitialspace=True, index_col=False
            )
            idx = f.split(".")[-2]
            kwargs = dict(label=f"mcmc #{idx}", color="C{}".format(idx), alpha=0.75)
            ax[i, 0].semilogy(df.N, df.Rminus1, "-o", **kwargs)
            ax[i, 0].set_ylabel(r"$R-1$")
            ax[i, 1].plot(df.N, df.acceptance_rate, "-o", **kwargs)
            ax[i, 1].set_ylabel(r"acceptance rate")
        leg = ax[i, 1].legend(
            title=k, bbox_to_anchor=(1, 1), loc="upper left", labelcolor="linecolor"
        )
        leg._legend_box.align = "left"

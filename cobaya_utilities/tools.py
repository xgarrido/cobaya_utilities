import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def print_chains_size(mcmc_dir, mcmc_names=None, tablefmt="html"):
    """Print MCMC sample size given a set of directories

    Parameters
    ----------
    mcmc_dir: str or list
      a base directory holding the MCMC samples. Either a string holding regex expression
      or a list of directories
    mcmc_names: list
      a list of names which corresponds to the directory names.
    tablefmt: str
      the format of the table (default: html)
    """

    if isinstance(mcmc_dir, str):
        mcmc_dir = glob.glob(mcmc_dir)

    if mcmc_names is None:
        mcmc_names = [os.path.basename(d) for d in mcmc_dir]

    nchains = {}
    for i, d in enumerate(mcmc_dir):
        key = mcmc_names[i]
        files = sorted(glob.glob(os.path.join(d, "mcmc.?.txt")))
        nchains[key] = [sum(1 for line in open(f)) for f in files]
        nchains[key] += [sum(nchains[key])]

    from tabulate import tabulate

    return tabulate(
        [(k, *v) for k, v in nchains.items()],
        headers=["mcmc {}".format(i) for i in range(1, 5)] + ["total"],
        tablefmt=tablefmt,
    )


def plot_chains(mcmc_dir, params, title=None, nrow=None, ncol=None):
    """Plot MCMC sample evolution

    Parameters
    ----------
    mcmc_dir: str
      a base directory holding the MCMC samples
    params: list
      a list of parameter names
    title: str
      a title for the figure
    nrow: int
      the number of rows within the plot
    ncol: int
      the number of columns within the plot
    """

    files = sorted(glob.glob(os.path.join(mcmc_dir, "mcmc.?.txt")))

    fig = plt.figure(figsize=(15, 10))
    if title is not None:
        fig.suptitle(title)
    nrow = (len(params) + 1) // 2 if nrow is None else nrow
    ncol = (len(params) + 1) // 2 if ncol is None else ncol
    ax = [plt.subplot(nrow, ncol, i + 1) for i in range(len(params))]

    # Loop over files independently
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


def plot_progress(mcmc_dir, mcmc_names=None):
    """Plot Gelman R-1 parameter and acceptance rate

    Parameters
    ----------
    mcmc_dir: str or list
      a base directory holding the MCMC samples. Either a string holding regex expression
      or a list of directories
    mcmc_names: list
      a list of names which corresponds to the directory names.
    """

    if isinstance(mcmc_dir, str):
        mcmc_dir = glob.glob(mcmc_dir)

    if mcmc_names is None:
        mcmc_names = [os.path.basename(d) for d in mcmc_dir]

    nrow = (len(mcmc_dir) + 1) // 2
    ncol = (len(mcmc_dir) + 1) // 2
    fig, ax = plt.subplots(2 * nrow, ncol, figsize=(15, 10), sharex=True)

    for i, d in enumerate(mcmc_dir):
        files = sorted(glob.glob(os.path.join(d, "mcmc.?.progress")))
        for f in files:
            cols = [a.strip() for a in open(f).readline().lstrip("#").split()]
            df = pd.read_csv(
                f, names=cols, comment="#", sep=" ", skipinitialspace=True, index_col=False
            )
            idx = f.split(".")[-2]
            kwargs = dict(label="mcmc #{}".format(idx), color="C{}".format(idx), alpha=0.75)
            ax[i, 0].semilogy(df.N, df.Rminus1, "-o", **kwargs)
            ax[i, 0].set_ylabel(r"$R-1$")
            ax[i, 1].plot(df.N, df.acceptance_rate, "-o", **kwargs)
            ax[i, 1].set_ylabel(r"acceptance rate")
        leg = ax[i, 1].legend(title=mcmc_names[i], bbox_to_anchor=(1, 1), loc="upper left")
        leg._legend_box.align = "left"

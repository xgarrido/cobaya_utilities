import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_chains_size(mcmc_samples):
    """Print MCMC sample size given a set of directories

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value.
    """

    nchains, status = {}, {}
    for k, v in mcmc_samples.items():
        files = sorted(glob.glob(os.path.join(v, "mcmc.?.txt")))
        nchains[k] = 4 * [0]
        status[k] = 5 * [""]
        total = 0
        for f in files:
            i = int(f.split(".")[-2]) - 1
            nchains[k][i] = sum(1 for line in open(f))
            total += nchains[k][i]
            # Check status
            for line in open(f.replace(".txt", ".log")):
                if "[mcmc] The run has converged!" in line:
                    status[k][i] = "done"
                if "[mcmc] *ERROR*" in line:
                    status[k][i] = "error"
        nchains[k] += [total]
        if sum(np.array(status[k]) == "done") == 4:
            status[k][-1] = "done"

    df = pd.DataFrame(nchains, index=[f"mcmc {i}" for i in range(1, 5)] + ["total"]).T
    status = pd.DataFrame(status, index=[f"mcmc {i}" for i in range(1, 5)] + ["total"]).T

    def _style_table(x):
        df1 = pd.DataFrame("", index=x.index, columns=x.columns)
        mask = status == "done"
        df1[mask] = "background-color: lightgreen"
        mask = status == "error"
        df1[mask] = "background-color: lightcoral"
        return df1

    return df.style.apply(_style_table, axis=None)


def print_results(mcmc_samples, params, labels, limit=1):
    """Print results given a set of MCMC samples and a list of parameters

    Parameters
    ----------
    mcmc_samples: list
      a list of MCSamples.
    params: list
      a list of parameters.
    labels: list
      a list of labels to be used a row index.
    limit: int
      the confidence limit of the results (default: 1 i.e. 68%).
    """
    d = {}

    r = re.compile(r".*\$(.*)\$.*&.*\$(.*)\$.*")
    for sample in mcmc_samples:
        result = sample.getTable(limit=limit, paramList=params)

        for line in result.lines:
            found = r.findall(line)
            if len(found):
                name, value = found[0]
                name, value = f"${name}$", f"${value}$"
                if d.get(name):
                    d[name] += [value]
                else:
                    d[name] = [value]

    df = pd.DataFrame(d, index=labels)
    return df


def plot_chains(mcmc_dir, params, title=None, ncol=None, ignore_rows=0.0, no_cache=False):
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

        sample = loadMCSamples(f[:-4], no_cache=no_cache, settings={"ignore_rows": ignore_rows})
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
    fig, ax = plt.subplots(2 * nrow, ncol, figsize=(15, 5 * nrow), sharex=True)

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

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_chains_size(mcmc_samples, with_bar=True, bar_color="#b9b9b9"):
    """Print MCMC sample size given a set of directories

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value.
    with_bar: bool
      showing an histogram bar for each cell given the number of mcmc samples
    bar_color: str
      bar color as hex string
    """

    nchains, status = {}, {}
    for k, v in mcmc_samples.items():
        files = sorted(glob.glob(os.path.join(v, "mcmc.?.txt")))
        assert len(files) > 0, "Missing mcmc chains!"
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
        for i, n in enumerate(nchains[k]):
            if n / total < 0.10 and status[k][i] == "":
                status[k][i] = "warning"

    df = pd.DataFrame(nchains, index=[f"mcmc {i}" for i in range(1, 5)] + ["total"]).T
    status = pd.DataFrame(status, index=[f"mcmc {i}" for i in range(1, 5)] + ["total"]).T

    def _style_table(x):
        css_tmpl = "background-color: {}"
        df1 = pd.DataFrame("", index=x.index, columns=x.columns)
        colors = dict(warning="bisque", done="lightgreen", error="lightcoral")
        for state, color in colors.items():
            mask = status == state
            df1[mask] = css_tmpl.format(color)
        return df1

    s = df.style
    if with_bar:
        s = s.bar(color=bar_color)
    return s.apply(_style_table, axis=None)


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

    r = re.compile(r"(.*)(=|<|>)(.*)")
    for param in params:
        for sample in mcmc_samples:
            found = r.findall(sample.getInlineLatex(param, limit=limit))
            assert (
                len(found) == 1
            ), f"Something gets wrong when retrieving limits for '{param}' parameter!"
            name, sign, value = found[0]
            if "---" in value:
                value = " "
            d.setdefault(f"${name}$", []).append(
                f"${value}$" if sign == "=" else f"${sign}{value}$"
            )
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
    from getdist import loadMCSamples

    ncol = ncol if ncol is not None else len(params)
    nrow = len(params) // ncol + 1 if ncol is not None else 1
    fig = plt.figure(figsize=(15, 2 * nrow))
    ax = [plt.subplot(nrow, ncol, i + 1) for i in range(len(params))]

    # Loop over files independently
    files = sorted(glob.glob(os.path.join(mcmc_dir, "mcmc.?.txt")))
    for f in files:
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
    leg = fig.legend(
        [f"mcmc #{i}" for i in range(1, len(files) + 1)],
        bbox_to_anchor=(1.0, 0.6),
        labelcolor="linecolor",
        loc="upper left",
        title=title,
    )
    leg._legend_box.align = "left"
    plt.tight_layout()


def plot_progress(mcmc_samples):
    """Plot Gelman R-1 parameter and acceptance rate

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value.
    """
    nrows = len(mcmc_samples)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharex=True)
    axes = np.atleast_2d(axes)

    for i, (k, v) in enumerate(mcmc_samples.items()):
        files = sorted(glob.glob(os.path.join(v, "mcmc.?.progress")))
        for f in files:
            cols = [a.strip() for a in open(f).readline().lstrip("#").split()]
            df = pd.read_csv(
                f, names=cols, comment="#", sep=" ", skipinitialspace=True, index_col=False
            )
            idx = f.split(".")[-2]
            kwargs = dict(label=f"mcmc #{idx}", color=f"C{idx}", alpha=0.75)
            axes[i, 0].semilogy(df.N, df.Rminus1, "-o", **kwargs)
            axes[i, 0].set_ylabel(r"$R-1$")

            axes[i, 1].plot(df.N, df.acceptance_rate, "-o", **kwargs)
            axes[i, 1].set_ylabel(r"acceptance rate")
        leg = axes[i, 1].legend(
            title=k, bbox_to_anchor=(1, 1), loc="upper left", labelcolor="linecolor"
        )
        leg._legend_box.align = "left"
    plt.tight_layout()

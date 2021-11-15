import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_chain_filenames(path, prefix="mcmc.", suffix=".txt"):
    return sorted(
        [
            f
            for f in glob.glob(os.path.join(path, f"{prefix}*{suffix}"))
            if re.match(f"{prefix}[0-9]+{suffix}", os.path.basename(f))
        ]
    )


def print_chains_size(mcmc_samples, with_bar=False, bar_color="#b9b9b9"):
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

    index = set()
    nchains, status = {}, {}
    for name, path in mcmc_samples.items():
        files = _get_chain_filenames(path)
        assert len(files) > 0, f"Missing mcmc chains in '{path}' directory!"
        nchains[name], status[name] = len(files) * [0], (len(files) + 1) * [""]
        total = 0
        for f in files:
            i = int(f.split(".")[-2]) - 1
            index.add(f"mcmc {i + 1}")
            nchains[name][i] = sum(1 for line in open(f))
            total += nchains[name][i]
            # Check status
            for line in open(f.replace(".txt", ".log")):
                if "[mcmc] The run has converged!" in line:
                    status[name][i] = "done"
                if "[mcmc] *ERROR*" in line:
                    status[name][i] = "error"
        nchains[name] += [total]
        if sum(np.array(status[name]) == "done") == len(files):
            status[name][-1] = "done"
        for i, n in enumerate(nchains[name]):
            if n / total < 0.10 and status[name][i] == "":
                status[name][i] = "warning"

    df = pd.DataFrame(nchains, index=sorted(index.union({"total"}))).T
    status = pd.DataFrame(status, index=sorted(index.union({"total"}))).T

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
    df = pd.DataFrame(d, index=labels[: len(mcmc_samples)])
    return df


def plot_chains(
    mcmc_dir, params, title=None, ncol=None, highlight_burnin=0.4, ignore_rows=0.0, no_cache=False
):
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
    ignore_rows: float
      the fraction of samples to ignore
    no_cache: bool
      remove the getdist cache
    """
    from getdist import loadMCSamples
    from matplotlib.lines import Line2D

    ncol = ncol if ncol is not None else len(params)
    nrow = len(params) // ncol + 1 if ncol is not None else 1
    fig = plt.figure(figsize=(15, 2 * nrow))
    ax = [plt.subplot(nrow, ncol, i + 1) for i in range(len(params))]

    # Loop over files independently
    files = _get_chain_filenames(mcmc_dir)
    for f in files:
        sample = loadMCSamples(f[:-4], no_cache=no_cache, settings={"ignore_rows": ignore_rows})
        color = f"C{f.split('.')[-2]}"

        # Get param lookup table
        lookup = {
            par.name: dict(pos=i, label=par.label)
            for i, par in enumerate(sample.getParamNames().names)
        }
        for i, p in enumerate(params):
            ax[i].set_ylabel(r"${}$".format(lookup[p].get("label")))
            y = sample.samples[:, lookup[p].get("pos")]
            x = np.arange(len(y))
            idx_burnin = -int(highlight_burnin * len(y))
            ax[i].plot(x[idx_burnin:], y[idx_burnin:], alpha=0.75, color=color)
            if highlight_burnin > 0.0:
                ax[i].plot(x[:idx_burnin], y[:idx_burnin], alpha=0.25, color=color)

    leg = fig.legend(
        [Line2D([0], [0], color=f"C{f.split('.')[-2]}") for f in files],
        [f"mcmc #{f.split('.')[-2]}" for f in files],
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=True)
    axes = np.atleast_2d(axes)

    for i, (k, v) in enumerate(mcmc_samples.items()):
        files = _get_chain_filenames(v, suffix=".progress")
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

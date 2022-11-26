import glob
import os
import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _get_chain_filenames(path, prefix="mcmc.", suffix=".txt"):
    return sorted(
        [
            f
            for f in glob.glob(os.path.join(path, f"{prefix}*{suffix}"))
            if re.match(f"{prefix}[0-9]+{suffix}", os.path.basename(f))
        ]
    )


def plot_chains_progress(mcmc_samples):
    r = re.compile("\[mcmc\] Progress @ (.*) : (.*) steps taken, and (.*) accepted.")
    data = []
    for name, path in mcmc_samples.items():
        files = _get_chain_filenames(path, suffix=".log")
        for fn in files:
            with open(fn) as f:
                for line in f:
                    found = r.findall(line)
                    if len(found) == 0:
                        continue
                    time, total_steps, accepted_steps = found[0]
                    data.append([name, fn.split(".")[-2], time, "total", total_steps])
                    data.append([name, fn.split(".")[-2], time, "accepted", accepted_steps])

    return pd.DataFrame(
        data=data,
        columns=["simulation id", "mcmc id", "time", "status", "steps"],
    )


def print_chains_size(
    mcmc_samples,
    with_bar=False,
    bar_color="#b9b9b9",
    with_color_gradient=True,
    color_palette="vlag",
    hide_status=True,
):
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
    r = re.compile("\[mcmc\] Progress @ (.*) : (.*) steps taken, and (.*) accepted.")

    data = {}
    for irow, (name, path) in enumerate(mcmc_samples.items()):
        files = _get_chain_filenames(path, suffix=".log")
        for fn in sorted(files):
            mcmc_name = f"mcmc {fn.split('.')[-2]}"
            status = dict(done="[mcmc] The run has converged!", error="[mcmc] *ERROR*")
            data.setdefault(name, {}).update({(mcmc_name, "status"): "running"})
            with open(fn) as f:
                for line in f:
                    for state, msg in status.items():
                        if msg in line:
                            data[name].update({(mcmc_name, "status"): state})

                    found = r.findall(line)
                    if len(found) == 0:
                        continue
                    time, total_steps, accepted_steps = found[0]
            total_steps, accepted_steps = int(total_steps), int(accepted_steps)
            rate = accepted_steps / total_steps
            for field, content in zip(
                ["accept.", "total", "rate"], [accepted_steps, total_steps, rate]
            ):
                data[name].update({(mcmc_name, field): content})

    df = pd.DataFrame.from_dict(data, orient="index")
    mcmc_names = df.columns.get_level_values(0).unique()

    s = df.style
    if hide_status:
        s.hide([(name, "status") for name in mcmc_names], axis="columns")
    if with_bar:
        s.bar(
            subset=[(name, "accept.") for name in mcmc_names],
            color=bar_color,
            height=50,
            width=60,
        )
    if with_color_gradient:
        cm = sns.color_palette(color_palette, as_cmap=True)
        s.background_gradient(
            subset=[(name, "rate") for name in mcmc_names],
            cmap=cm,
            axis=None,
            # vmin=0.0,
            # vmax=0.3,
            # low=0,
            # high=0.3,
        )

    def _style_table(x):
        df = pd.DataFrame("", index=x.index, columns=x.columns)
        states = dict(running="#55A868", done="#55A868", error="#C44E52")
        for name, (state, color) in product(mcmc_names, states.items()):
            mask = x.loc[:, (name, "status")] == state
            css = f"""color: {color}; text-decoration: {color} underline;
            text-decoration-thickness: 5px; font-weight: {'normal' if state=="running" else 'bold'}"""
            for col in ["total"]:
                df.loc[:, (name, col)][mask] = css
        return df

    s.format({sub: "{:.1%}".format for sub in [(name, "rate") for name in mcmc_names]}).apply(
        _style_table, axis=None
    )
    return s


def print_results(samples, params, labels, limit=1):
    """Print results given a set of MCMC samples and a list of parameters

    Parameters
    ----------
    samples: list
      a list of MCSamples.
    params: list
      a list of parameters.
    labels: list
      a list of labels to be used a row index.
    limit: int
      the confidence limit of the results (default: 1 i.e. 68%).
    """
    params = params if isinstance(params, list) else list(params.keys())
    labels = labels if isinstance(labels, list) else list(labels.keys())
    d = {}
    r = re.compile(r"(.*)(=|<|>)(.*)")
    for param in params:
        for sample in samples:
            latex = None
            for par in param.split("|"):
                if sample.getParamNames().hasParam(par):
                    latex = sample.getInlineLatex(par, limit=limit)
                    if "<" in latex or ">" in latex:
                        latex = sample.getInlineLatex(par, limit=2)
                    break
            assert latex is not None, f"Parameter '{param}' not found!"
            found = r.findall(latex)
            assert (
                len(found) == 1
            ), f"Something gets wrong when retrieving limits for '{param}' parameter!"
            name, sign, value = found[0]
            name = name.replace(" ", "")
            if "---" in value:
                value = "$-$"
            d.setdefault(f"${name}$", []).append(
                f"${value}$" if sign == "=" else f"${sign}{value}$"
            )
    df = pd.DataFrame(d, index=labels[: len(samples)])
    return df


def plot_chains(
    mcmc_samples,
    params,
    ncol=None,
    highlight_burnin=0.4,
    ignore_rows=0.0,
    show_mean_std=True,
    show_only_mcmc=None,
    no_cache=False,
    markers={},
    markers_args={},
):
    """Plot MCMC sample evolution

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value.
    params: dict or list
      a dict holding the parameter names for the different mcmc_samples or
      a unique list of parameter names
    ncol: int
      the number of columns within the plot
    highlight_burnin: float
      the fraction of samples to highlight (below the burnin value, the color is faded)
    ignore_rows: float
      the fraction of samples to ignore
    show_mean_std: bool
      show the mean/std values over the different samples
    show_only_mcmc: int or list
      only show chains given their number
    no_cache: bool
      remove the getdist cache
    """
    from getdist import loadMCSamples
    from matplotlib.lines import Line2D

    if not isinstance(params, (list, dict)):
        raise ValueError("Parameter list must be either a list or a dict!")

    if show_only_mcmc is not None:
        if not isinstance(show_only_mcmc, (int, list)):
            raise ValueError("Parameter 'show_only_mcmc' must be either a int or a list of int!")
        if isinstance(show_only_mcmc, int):
            show_only_mcmc = [show_only_mcmc]

    if ignore_rows > 0.0:
        highlight_burnin = 0.0

    markers_args = markers_args or dict(color="0.15", ls="--", lw=1)
    stored_axes = {}
    for name, path in mcmc_samples.items():
        axes = None

        # Loop over files independently
        files = _get_chain_filenames(path)

        chains = {}
        min_chain_size = np.inf
        for f in files:
            imcmc = int(f.split(".")[-2])
            if show_only_mcmc and imcmc not in show_only_mcmc:
                continue
            sample = loadMCSamples(f[:-4], no_cache=no_cache, settings={"ignore_rows": ignore_rows})

            # Get param lookup table
            lookup = {
                par.name: dict(pos=i, label=par.label)
                for i, par in enumerate(sample.getParamNames().names)
            }

            if axes is None:
                # Keep only relevant parameters
                selected_params = [par for par in params if par in lookup]
                ncol = ncol if ncol is not None else len(selected_params)
                nrow = len(selected_params) // ncol + 1 if ncol is not None else 1
                fig = plt.figure(figsize=(15, 2 * nrow))
                axes = [plt.subplot(nrow, ncol, i + 1) for i in range(len(selected_params))]

            color = f"C{imcmc}"
            if sample.samples.shape[0] < min_chain_size:
                min_chain_size = sample.samples.shape[0]
            for i, p in enumerate(selected_params):
                axes[i].set_ylabel(r"${}$".format(lookup[p].get("label")))
                y = sample.samples[:, lookup[p].get("pos")]
                x = np.arange(len(y))
                idx_burnin = -int(highlight_burnin * len(y))
                axes[i].plot(x[idx_burnin:], y[idx_burnin:], alpha=0.75, color=color)
                if highlight_burnin > 0.0:
                    axes[i].plot(x[: idx_burnin + 1], y[: idx_burnin + 1], alpha=0.25, color=color)
                if p in markers:
                    axes[i].axhline(markers[p], **markers_args)
                chains.setdefault(p, []).append(y)

        if show_mean_std:
            for i, p in enumerate(selected_params):
                data = np.array([chain[:min_chain_size] for chain in chains[p]])
                mu, std = np.mean(data), np.std(data)
                axes[i].axhline(mu, color="0.6", lw=1)
                for sign in [-1, +1]:
                    axes[i].axhline(mu + std * sign, color="0.6", ls="--", lw=1)

        leg = fig.legend(
            [Line2D([0], [0], color=f"C{f.split('.')[-2]}") for f in files],
            [f"mcmc #{f.split('.')[-2]}" for f in files],
            bbox_to_anchor=(1.0, 0.6),
            labelcolor="linecolor",
            loc="upper left",
            title=name,
        )
        leg._legend_box.align = "left"
        fig.tight_layout()
        stored_axes[name] = {p: axes[i] for i, p in enumerate(selected_params)}

    return stored_axes


def plot_progress(mcmc_samples, sharex=True):
    """Plot Gelman R-1 parameter and acceptance rate

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value.
    sharex: bool
      share the x-axis between the several plot progress (default: True)
    """
    nrows = len(mcmc_samples)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=sharex)
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

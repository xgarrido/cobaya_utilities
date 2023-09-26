import glob
import os
import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_default_root_path = "../output"


def _get_chain_filenames(path, prefix="mcmc", suffix=".txt"):
    return sorted(
        [
            f
            for f in glob.glob(os.path.join(path, f"{prefix}*{suffix}"))
            if re.match(rf"{prefix}(\.|)([0-9]+|){re.escape(suffix)}", os.path.basename(f))
        ]
    )


def _get_path(name, value):
    path = value
    if isinstance(value, dict):
        path = value.get("path", os.path.join(_default_root_path, name))
    if not os.path.exists(path):
        raise ValueError(f"'{name} chains can not be located in '{path}'")
    return path


def create_symlink(mcmc_samples, prefix="mcmc"):
    """Create missing files when running chains without mpi support (as it is done in CC in2p3)"""
    regex = re.compile(rf".*{prefix}.*\.([0-9]+.txt)")
    for name, value in mcmc_samples.items():
        path = _get_path(name, value)

        if files := _get_chain_filenames(path, prefix=prefix):
            continue
        if not (files := _get_chain_filenames(path, prefix=prefix + ".*", suffix=".txt")):
            raise ValueError("Missing chain files!")

        print("Create MCMC symlinks...")
        for f in files:
            if m := regex.match(f):
                dest = f.replace(m.group(1), "")
                os.symlink(os.path.basename(f), dest + "txt")
                updated_yaml = os.path.join(os.path.dirname(f), f"{prefix}.updated.yaml")
                if not os.path.exists(updated_yaml):
                    os.symlink(os.path.basename(dest) + "updated.yaml", updated_yaml)


def print_chains_size(
    mcmc_samples,
    with_bar=False,
    bar_color="#b9b9b9",
    with_color_gradient=True,
    color_palette="Greens",
    hide_status=True,
    with_gelman_rubin=True,
    prefix="mcmc",
):
    """Print MCMC sample size given a set of directories

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value
      or a dict configuration
    with_bar: bool
      showing an histogram bar for each cell given the number of mcmc samples
    bar_color: str
      bar color as hex string
    color_palette: str
      color palette to be used for background gradient
    hide_status: bool
      either to hide chain status (running, error, done)
    with_gelman_rubin: bool
      add Gelman-Rubin metric aka R-1
    prefix: str
      prefix for chain names (default is "mcmc.")
    """
    create_symlink(mcmc_samples, prefix)
    r = re.compile(r"\[mcmc\] Progress @ (.*) : (.*) steps taken, and (.*) accepted.")
    regex_log = re.compile(r".*mcmc\.([0-9]+).log")
    regex_progress = re.compile(r".*mcmc\.([0-9]+).progress")

    found_rminus1 = []
    data = {}
    for irow, (name, value) in enumerate(mcmc_samples.items()):
        path = _get_path(name, value)
        name = value.get("label", name) if isinstance(value, dict) else name
        files = _get_chain_filenames(path, prefix=prefix, suffix=".log")
        if not files:
            print(f"Missing log files for chains '{name}' within path '{path}'!")
            return
        for fn in sorted(files):
            total_steps = 0
            idx = 1 if not (m := regex_log.match(fn)) else m.group(1)
            mcmc_name = f"mcmc {idx}"
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
                    time, current_steps, accepted_steps = found[0]
                    current_steps = int(current_steps)
                    total_steps = (
                        current_steps
                        if current_steps > total_steps
                        else total_steps + current_steps
                    )
            accepted_steps = int(accepted_steps)
            rate = accepted_steps / total_steps
            for field, content in zip(
                ["accept.", "total", "rate", "R-1"], [accepted_steps, total_steps, rate, None]
            ):
                data[name].update({(mcmc_name, field): content})
        if with_gelman_rubin:
            files = _get_chain_filenames(path, suffix=".progress")
            for fn in sorted(files):
                idx = 1 if not (m := regex_progress.match(fn)) else m.group(1)
                mcmc_name = f"mcmc {idx}"
                with open(fn) as f:
                    for line in f:
                        pass
                    if line.startswith("#"):
                        continue
                    data[name].update({(mcmc_name, "R-1"): f"{float(line.split()[-2]):.2f}"})
                    found_rminus1 += [mcmc_name]

    df = pd.DataFrame.from_dict(data, orient="index")
    df.dropna(axis=1, how="all", inplace=True)

    mcmc_names = list(df.columns.get_level_values(0).unique())

    # Append total count
    all_name = "all mcmc"
    dfs = [
        pd.DataFrame(
            {(all_name, field): df.loc[:, df.columns.get_level_values(1) == field].sum(axis=1)}
        )
        for field in ["accept.", "total"]
    ]
    df = pd.concat([df] + dfs, axis=1)
    df[(all_name, "rate")] = df[(all_name, "accept.")] / df[(all_name, "total")]
    all_mcmc_names = mcmc_names + [all_name]

    s = df.style
    if hide_status:
        s.hide([(name, "status") for name in mcmc_names], axis="columns")
    if with_bar:
        s.bar(
            subset=[(name, "accept.") for name in all_mcmc_names],
            color=bar_color,
            height=50,
            width=60,
        )
    if with_color_gradient:
        cm = sns.color_palette(color_palette, as_cmap=True)
        s.background_gradient(subset=[(name, "rate") for name in mcmc_names], cmap=cm, axis=None)
        # if with_gelman_rubin and found_rminus1:
        #     cm = sns.color_palette(color_palette + "_r", as_cmap=True)
        #     s.text_gradient(
        #         subset=[(name, "R-1") for name in mcmc_names if name in found_rminus1],
        #         cmap=cm,
        #         axis=None,
        #     )
        #     # s.highlight_null(color="white")
        s.background_gradient(
            subset=[(all_name, "total")], cmap=sns.color_palette("Blues", as_cmap=True)
        )
        s.background_gradient(
            subset=[(all_name, "rate")], cmap=sns.color_palette("Reds", as_cmap=True)
        )

    def _style_table(x):
        df = pd.DataFrame("", index=x.index, columns=x.columns)
        states = dict(running="#55A868", done="#4C72B0", error="#C44E52")
        for name, (state, color) in product(mcmc_names, states.items()):
            mask = x.loc[:, (name, "status")] == state
            css = f"""color: {color}; text-decoration: {color} underline;
            text-decoration-thickness: 5px; font-weight: {'normal' if state=="running" else 'bold'}"""
            df.loc[:, (name, "total")][mask] = css
            if state == "error":
                df.loc[:, (name, "rate")][mask] = css

        return df

    s.format(
        {sub: "{:.1%}".format for sub in [(name, "rate") for name in all_mcmc_names]}, na_rep=""
    ).apply(_style_table, axis=None)
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
      a list of labels to be used as row index.
    limit: int
      the confidence limit of the results (default: 1 i.e. 68%).
    """
    params = params if isinstance(params, list) else list(params.keys())
    labels = labels if isinstance(labels, list) else list(labels.keys())
    d, cols = {}, {}
    r = re.compile(r"(.*)(=|<|>)(.*)")
    for param in params:
        for sample in samples:
            latex = None
            sign = ""
            for par in param.split("|"):
                if sample.getParamNames().hasParam(par):
                    latex = sample.getInlineLatex(par, limit=limit)
                    if "<" in latex or ">" in latex:
                        latex = sample.getInlineLatex(par, limit=2)
                    break
            if not latex:
                value = " "
            else:
                # assert latex is not None, f"Parameter '{param}' not found!"
                found = r.findall(latex)
                assert (
                    len(found) == 1
                ), f"Something gets wrong when retrieving limits for '{param}' parameter!"
                name, sign, value = found[0]
                if param not in cols:
                    cols[param] = rf"${name.strip()}$"
            if "---" in value:
                value = "-"
            d.setdefault(param, []).append(f"${value}$" if sign == "=" else f"${sign}{value}$")
    df = pd.DataFrame(d, index=labels[: len(samples)])
    return df.rename(columns=cols)


def plot_chains(
    mcmc_samples,
    params,
    ncol=None,
    highlight_burnin=0.4,
    ignore_rows=0.0,
    show_mean_std=True,
    show_only_mcmc=None,
    no_cache=False,
    markers=None,
    markers_args=None,
    prefix="mcmc",
):
    """Plot MCMC sample evolution

        Parameters
        ----------
        mcmc_samples: dict
          a dict holding a name as key for the sample and a corresponding directory as value
          or a dict configuration
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
        markers: dict
    :      dictionnary holding the "expected" value for a parameter
        markers_args: dict
          marker kwargs to pass to plt.axhline
        prefix: str
          prefix name of chains
    """
    create_symlink(mcmc_samples, prefix)
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

    markers = markers or {}
    markers_args = markers_args or dict(color="0.15", ls="--", lw=1)
    stored_axes, color_palettes = {}, {}
    regex = re.compile(rf".*{prefix}\.([0-9]+).txt")
    for name, value in mcmc_samples.items():
        path = _get_path(name, value)
        if isinstance(value, dict):
            name = value.get("label", name)
        axes = None

        # Loop over files independently
        files = _get_chain_filenames(path, prefix=prefix)
        if not files:
            raise ValueError("Missing chain files!")

        chains = {}
        min_chain_size = np.inf
        sample = None
        for f in files:
            imcmc = 0 if not (m := regex.match(f)) else int(m.group(1))
            if show_only_mcmc and imcmc not in show_only_mcmc:
                continue
            kwargs = dict(no_cache=no_cache, settings={"ignore_rows": ignore_rows})
            try:
                sample = loadMCSamples(f[:-4], **kwargs)
                samples = sample.samples
            except AttributeError:
                sample = sample or loadMCSamples(os.path.join(os.path.dirname(f), prefix), **kwargs)
                samples = sample.getSeparateChains()[imcmc - 1].samples

            # Get param lookup table
            lookup = {
                par.name: dict(pos=i, label=par.label)
                for i, par in enumerate(sample.getParamNames().names)
            }

            if axes is None:
                # Keep only relevant parameters
                selected_params = [par for par in params if par in lookup]
                if ncol is None:
                    ncol, nrow = len(selected_params), 1
                else:
                    nrow = len(selected_params) // ncol
                    nrow += 1 if len(selected_params) > nrow * ncol else 0
                fig, axes = plt.subplots(nrow, ncol, sharex=True, figsize=(15, 2 * nrow))
                axes = axes.flatten()

            color = f"C{imcmc}"
            # if color_name:
            #     if color_name not in color_palettes:
            #         color_palettes[color_name] = sns.blend_palette(
            #             ["white", color_name], n_colors=len(files) + 1
            #         )
            #     color = color_palettes[color_name][imcmc]
            if samples.shape[0] < min_chain_size:
                min_chain_size = samples.shape[0]
            for i, p in enumerate(selected_params):
                axes[i].set_ylabel(r"${}$".format(lookup[p].get("label")))
                y = samples[:, lookup[p].get("pos")]
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

        # Remove axes with no data inside
        _ = [fig.delaxes(ax) for ax in axes if not len(ax.get_lines())]
        leg = fig.legend(
            [Line2D([0], [0], color=f"C{f.split('.')[-2]}") for f in files],
            [f"mcmc #{f.split('.')[-2]}" for f in files],
            bbox_to_anchor=(1.0, 0.5 if nrow > 1 else 1.0),
            labelcolor="linecolor",
            loc="center left",
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

    regex = re.compile(r".*mcmc\.([0-9]+).progress")
    for i, (name, value) in enumerate(mcmc_samples.items()):
        path = _get_path(name, value)
        if isinstance(value, dict):
            name = value.get("label", name)

        files = _get_chain_filenames(path, suffix=".progress")
        is_empty = True
        for fn in files:
            with open(fn) as f:
                cols = [a.strip() for a in f.readline().lstrip("#").split()]
            df = pd.read_csv(
                fn, names=cols, comment="#", sep=" ", skipinitialspace=True, index_col=False
            )
            is_empty &= len(df) == 0
            idx = 1 if not (m := regex.match(fn)) else m.group(1)
            kwargs = dict(label=f"mcmc" + (f" #{idx}" if idx else ""), color=f"C{idx}", alpha=0.75)
            axes[i, 0].semilogy(df.N, df.Rminus1, "-o", **kwargs)
            axes[i, 0].set_ylabel(r"$R-1$")

            axes[i, 1].plot(df.N, df.acceptance_rate, "-o", **kwargs)
            axes[i, 1].set_ylabel(r"acceptance rate")
        if is_empty:
            fig.delaxes(axes[i, 0])
            fig.delaxes(axes[i, 1])
        if len(files) > 1:
            leg = axes[i, 1].legend(
                title=name, bbox_to_anchor=(1, 1), loc="upper left", labelcolor="linecolor"
            )
            leg._legend_box.align = "left"
    plt.tight_layout()
    return axes

import glob
import os
import re
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cobaya.yaml import yaml_load_file
from getdist.paramnames import mergeRenames
from matplotlib.lines import Line2D

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
        path = value.get("path", os.path.join(_default_root_path, str(name)))
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
    mpi_run=True,
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
    mpi_run: bool
      mpi run flag (default true)
    """
    create_symlink(mcmc_samples, prefix)
    regex_log = re.compile(r".*mcmc\.([0-9]+).log")
    regex_progress = re.compile(r".*mcmc\.([0-9]+).progress")

    found_rminus1 = []
    data = {}
    status = dict(done="mcmc] The run has converged!", error="mcmc] *ERROR*")
    for irow, (name, value) in enumerate(mcmc_samples.items()):
        path = _get_path(name, value)
        name = value.get("label", name) if isinstance(value, dict) else name
        files = _get_chain_filenames(path, prefix=prefix, suffix=".log")
        if not files:
            print(f"Missing log files for chains '{name}' within path '{path}'!")
            return
        if len(files) == 1 and mpi_run:
            total_steps = {}
            data.setdefault(name, {})
            r = re.compile(
                r"\[(.*) : mcmc\] Progress @ (.*) : (.*) steps taken, and (.*) accepted."
            )
            with open(files[0]) as f:
                for line in f:
                    for state, msg in status.items():
                        if msg in line:
                            data[name].update({(mcmc_name, "status"): state})
                    found = r.findall(line)
                    if len(found) == 0:
                        continue
                    idx, time, current_steps, accepted_steps = found[0]
                    mcmc_name = f"mcmc {int(idx)+1}"
                    data[name].update({(mcmc_name, "status"): "running"})

                    current_steps = int(current_steps)
                    ts = total_steps.setdefault(idx, 0)
                    total_steps[idx] = current_steps if current_steps > ts else ts + current_steps
                    accepted_steps = int(accepted_steps)
                    rate = accepted_steps / total_steps[idx] if total_steps[idx] != 0 else None
                    data[name].update({(mcmc_name, "R-1"): None})
                    data[name].update({(mcmc_name, "accept."): accepted_steps})
                    data[name].update({(mcmc_name, "total"): total_steps[idx]})
                    data[name].update({(mcmc_name, "rate"): rate})
        else:
            r = re.compile(r"\[mcmc\] Progress @ (.*) : (.*) steps taken, and (.*) accepted.")
            for fn in sorted(files):
                total_steps = {}
                irun = 0
                idx = 1 if not (m := regex_log.match(fn)) else m.group(1)
                mcmc_name = f"mcmc {idx}"
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
                        if irun in total_steps and total_steps[irun] > current_steps:
                            irun += 1
                        total_steps[irun] = current_steps
                total_steps = sum(total_steps.values())
                accepted_steps = int(accepted_steps)
                rate = accepted_steps / total_steps if total_steps != 0 else None
                for field, content in zip(
                    ["R-1", "accept.", "total", "rate"], [None, accepted_steps, total_steps, rate]
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
    df.sort_index(level=0, axis=1, sort_remaining=False, inplace=True)
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
            mask = x[(name, "status")] == state
            css = f"""color: {color}; text-decoration: {color} underline;
            text-decoration-thickness: 5px; font-weight: {'normal' if state=="running" else 'bold'}"""
            df.loc[mask, (name, "total")] = css
            if state == "error":
                df.loc[mask, (name, "rate")] = css

        return df

    s.format(
        {sub: "{:.1%}".format for sub in [(name, "rate") for name in all_mcmc_names]}, na_rep=""
    ).apply(_style_table, axis=None)
    return s


def print_results(samples, params, labels=None, limit=1, **kwargs):
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
    labels = labels or kwargs.get("legend_labels")
    labels = labels if isinstance(labels, list) else list(labels.keys())
    d, cols = {}, {}

    merge_renames = {}
    for sample in samples:
        merge_renames = mergeRenames(
            sample.getParamNames().getRenames(keep_empty=True), merge_renames
        )
    reverse_renames = {vv: [k] for k, v in merge_renames.items() for vv in v}

    r = re.compile(r"(.*)(=|<|>)(.*)")
    for param in params:
        for sample in samples:
            latex = None
            sign = ""
            for par in param.split("|"):
                pars = [par] + merge_renames.get(par, []) + reverse_renames.get(par, [])
                for p in pars:
                    if sample.getParamNames().hasParam(p):
                        latex = sample.getInlineLatex(p, limit=limit)
                        if "<" in latex or ">" in latex:
                            latex = sample.getInlineLatex(p, limit=2)
                        break
                if latex:
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
    priors=None,
    priors_args=None,
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
    show_mean_std: bool or str
      show the mean/std values over the different samples. It can either True/False or "rolling".
    show_only_mcmc: int or list
      only show chains given their number
    no_cache: bool
      remove the getdist cache
    markers: dict
      dictionnary holding the "expected" value for a parameter
    markers_args: dict
      marker kwargs to pass to plt.axhline
    prefix: str
      prefix name of chains
    """
    create_symlink(mcmc_samples, prefix)
    from getdist import loadMCSamples

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
    priors = priors or {}
    priors_args = priors_args or dict(color="0.75")
    stored_axes, color_palettes = {}, {}
    regex = re.compile(rf".*{prefix}\.([0-9]+).txt")
    merge_renames = {}
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

            # Add renames
            merge_renames = mergeRenames(
                sample.getParamNames().getRenames(keep_empty=True), merge_renames
            )
            # Get param lookup table
            lookup = {}
            for i, par in enumerate(sample.getParamNames().names):
                lookup[par.name] = dict(pos=i, label=par.label)
                if renames := merge_renames.get(par.name):
                    lookup.update({r: lookup[par.name] for r in renames})

            if axes is None:
                # Keep only relevant parameters
                selected_params = [par for par in params if par in lookup]
                if ncol is None:
                    ncol, nrow = len(selected_params), 1
                else:
                    nrow = len(selected_params) // ncol
                    nrow += 1 if len(selected_params) > nrow * ncol else 0
                fig, axes = plt.subplots(nrow, ncol, sharex=True, figsize=(16, 2 * nrow))
                axes = axes.flatten()

            # if color_name:
            #     if color_name not in color_palettes:
            #         color_palettes[color_name] = sns.blend_palette(
            #             ["white", color_name], n_colors=len(files) + 1
            #         )
            #     color = color_palettes[color_name][imcmc]
            if samples.shape[0] < min_chain_size:
                min_chain_size = samples.shape[0]
            for i, p in enumerate(selected_params):
                # In case a parameter has not moved, getdist remove it from the list of parameters
                if p not in lookup:
                    continue

                axes[i].set_ylabel(rf"${lookup[p].get('label')}$")
                y = samples[:, lookup[p].get("pos")]
                x = np.arange(len(y))
                idx_burnin = -int(highlight_burnin * len(y))
                axes[i].plot(
                    x[idx_burnin:], y[idx_burnin:], alpha=0.75, color=(color := f"C{int(imcmc)-1}")
                )
                if highlight_burnin > 0.0:
                    axes[i].plot(x[: idx_burnin + 1], y[: idx_burnin + 1], alpha=0.25, color=color)
                if p in markers:
                    axes[i].axhline(markers[p], **markers_args)
                if prior := priors.get(p):
                    mu, sigma = prior
                    axes[i].axhspan(mu - sigma, mu + sigma, **priors_args)
                chains.setdefault(p, []).append(y)

        if show_mean_std:

            def _moving_average(x, w):
                return np.convolve(x, np.ones(w), "valid") / w

            for i, p in enumerate(selected_params):
                if show_mean_std == "rolling":
                    max_size = np.max([chain.size for chain in chains[p]])
                    data = np.full((len(chains[p]), max_size), np.nan)
                    for j, chain in enumerate(chains[p]):
                        data[j, : len(chain)] = chain
                    x = np.arange(max_size)
                    mu, std = np.nanmean(data, axis=0), np.nanstd(data, axis=0)
                    std[std == 0.0] = np.nan
                    window = int(0.2 * max_size)
                    x = _moving_average(x, window)
                    mu = _moving_average(mu, window)
                    std = _moving_average(std, window)
                    axes[i].plot(x, mu, color="0.6", lw=1)
                    for sign in [-1, +1]:
                        axes[i].plot(x, mu + std * sign, color="0.6", ls="--", lw=1)
                else:
                    data = np.array([chain[:min_chain_size] for chain in chains[p]])
                    mu, std = np.mean(data), np.std(data)
                    axes[i].axhline(mu, color="0.6", lw=1)
                    for sign in [-1, +1]:
                        axes[i].axhline(mu + std * sign, color="0.6", ls="--", lw=1)

        # Remove axes with no data inside
        _ = [fig.delaxes(ax) for ax in axes if not len(ax.get_lines())]
        leg = fig.legend(
            [Line2D([0], [0], color=f"C{int(f.split('.')[-2])-1}") for f in files],
            [f"mcmc #{f.split('.')[-2]}" for f in files],
            bbox_to_anchor=(0.5, 1.025),
            labelcolor="linecolor",
            loc="center",
            title=name,
            ncol=len(files),
            title_fontsize="large",
        )
        # leg._legend_box.align = "center"
        fig.tight_layout()
        stored_axes[name] = {p: axes[i] for i, p in enumerate(selected_params)}

    return stored_axes


def plot_progress(mcmc_samples, sharex=True, share_fig=False):
    """Plot Gelman R-1 parameter and acceptance rate

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value.
    sharex: bool
      share the x-axis between the several plot progress (default: True)
    share_fig: bool
      plots are done within the same figure
    """
    nrows = len(mcmc_samples) if not share_fig else 1
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=sharex)
    axes = np.atleast_2d(axes)

    regex = re.compile(r".*mcmc\.([0-9]+).progress")
    for i, (name, value) in enumerate(mcmc_samples.items()):
        path = _get_path(name, value)
        if isinstance(value, dict):
            name = value.get("label", name)

        files = _get_chain_filenames(path, suffix=".progress")
        for fn in files:
            with open(fn) as f:
                cols = [a.strip() for a in f.readline().lstrip("#").split()]
            df = pd.read_csv(
                fn, names=cols, comment="#", sep=" ", skipinitialspace=True, index_col=False
            )
            if df.N.empty:
                continue

            idx = 1 if not (m := regex.match(fn)) else m.group(1)
            color = f"C{int(idx)-1}" if not share_fig else f"C{i}"
            kwargs = dict(label=f"mcmc" + (f" #{idx}" if idx else ""), color=color, alpha=0.75)
            ix = i if not share_fig else 0
            axes[ix, 0].semilogy(df.N, df.Rminus1, "-o", **kwargs)
            axes[ix, 0].set_ylabel(r"$R-1$")

            axes[ix, 1].plot(df.N, df.acceptance_rate, "-o", **kwargs)
            axes[ix, 1].set_ylabel(r"acceptance rate")
        if len(files) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                leg = axes[i, 1].legend(
                    title=name,
                    bbox_to_anchor=(1, 1),
                    loc="upper left",
                    labelcolor="linecolor",
                    alignment="left",
                )
    if share_fig:
        leg = fig.legend(
            [Line2D([0], [0], color=f"C{i}") for i, _ in enumerate(mcmc_samples)],
            mcmc_samples,
            bbox_to_anchor=(1, 1),
            labelcolor="linecolor",
            loc="center left",
        )
    fig.tight_layout()

    for i in range(nrows):
        if not axes[i, 0].lines:
            fig.delaxes(axes[i, 0])
            fig.delaxes(axes[i, 1])
    if not fig.axes:
        plt.close(fig)
    return fig


def get_sampled_parameters(
    mcmc_samples, prefix="mcmc", return_params=False, with_priors=False, column_width="150px"
):
    """Print MCMC sampled parameters

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value
      or a dict configuration
    prefix: str
      prefix for chain names (default is "mcmc.")
    return_params: bool
      return dict of params for each MCMC chain (default false)
    column_width: str
      the column width of the output dataframe (default: 150px)
    """

    create_symlink(mcmc_samples, prefix)
    r1 = re.compile(r".*mcmc\] \*.*[0-9+] : (.*)")
    r2 = re.compile(r".*model\].*Input: (.*)")

    rprior = re.compile(r".*lambda (.*):.*stats.norm.logpdf.*loc=(.*),.*scale=(.*).*\)")

    params_info = {}
    sampled_params = {}
    for name, value in mcmc_samples.items():
        path = _get_path(name, value)
        name = value.get("label", name) if isinstance(value, dict) else name
        files = _get_chain_filenames(path, prefix=prefix, suffix=".log")
        if not files:
            print(f"Missing log files for chains '{name}' within path '{path}'!")
            return

        updated_yaml = yaml_load_file(os.path.join(path, f"{prefix}.updated.yaml"))
        # To convert latex symbol to mathml
        # from latex2mathml.converter import convert
        # par: convert(rf"{meta.get('latex', par)}")
        latex_table = {
            par: rf"${meta.get('latex', par)}$"
            for par, meta in updated_yaml.get("params", {}).items()
        }
        with open(files[0]) as f:
            for line in f:
                found = r1.findall(line) or r2.findall(line)
                if len(found) == 0:
                    continue
                params = eval(found[0])
                sampled_params.setdefault(name, []).extend(params)
                params_info.setdefault((name, "parameter"), []).extend(
                    [latex_table.get(par, par) for par in params]
                )
                if "Sampling!" in line:
                    break
        if not with_priors:
            continue

        # Get priors
        external_priors = {}
        for k, v in updated_yaml.get("prior", {}).items():
            found = rprior.findall(v)
            if len(found) == 0:
                continue
            param, loc, scale = found[0]
            external_priors[param] = rf"$\mathcal{{G}}({float(loc)}, {float(scale)})$"
        for param in sampled_params.get(name, []):
            if param not in (params := updated_yaml.get("params")):
                raise ValueError("Sampled paremeter can not be found within input parameters !")

            if param in external_priors:
                params_info.setdefault((name, "prior"), []).append(external_priors[param])
                continue

            input_priors = params.get(param).get("prior", {})
            if dist := input_priors.get("dist"):
                if dist == "norm":
                    loc, scale = input_priors.get("loc"), input_priors.get("scale")
                    params_info.setdefault((name, "prior"), []).append(
                        rf"$\mathcal{{G}}({loc}, {scale})$"
                    )
            elif input_priors.get("min") or input_priors.get("max"):
                min, max = input_priors.get("min"), input_priors.get("max")
                params_info.setdefault((name, "prior"), []).append(
                    rf"$\mathcal{{U}}({min}, {max})$"
                )

    # print(params_info)
    df = pd.DataFrame.from_dict(params_info, orient="index").T.fillna("").drop_duplicates()
    df.columns = params_info.keys() if with_priors else sampled_params.keys()
    df = df.style.set_properties(width=column_width)
    if return_params:
        return df, sampled_params
    return df

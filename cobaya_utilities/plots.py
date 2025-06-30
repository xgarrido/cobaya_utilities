import logging
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from getdist import plots
from getdist.plots import get_subplot_plotter
from matplotlib.lines import Line2D
from scipy import stats

from .tools import _get_chain_filenames, _get_path


def set_style(
    use_seaborn=False,
    seaborn_style="ticks",
    seaborn_context="paper",
    palette="tab10",
    use_svg=False,
    use_tex=False,
    print_load_details=False,
    logging_level="error",
    backend="inline",
    **rc,
):
    """Set default plot settings

    Parameters
    ----------
    use_seaborn: bool
      use seaborn theming option
    seaborn_style: str
      set seaborn style (default: `ticks`)
    seaborn_context: str
      set seaborn context (default: `paper`)
    palette: str
      color palette name to be used for individual colors (default: tab10)
    use_svg: bool
      use `svg` output format for figure
    use_tex: bool
      use LaTeX axis labels
    print_load_details: bool
      print load details when getdist loads samples
    logging_level: str
      logging level to be passed to getdist
    backend: str
      matplotlib backend (default: inline)
    rc: dict
      overload matplotlib rc parameters
    """

    logging.getLogger("root").setLevel(getattr(logging, logging_level.upper(), logging.error))

    colors = None
    if palette == "the-lab":
        colors = dict(
            blue="#015692",
            orange="#b75501",
            green="#54790d",
            red="#c02d2e",
            purple="#803378",
            gray="#656e77",
        )
    elif palette == "getdist":
        colors = dict(blue="#006FED", red="#E03424", green="#009966")
    elif use_seaborn:
        palette = "deep"

    _default_color_cycle = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "yellow",
        "cyan",
    ]
    if colors:
        for color, code in colors.items():
            rgb = mpl.colors.colorConverter.to_rgb(color)
            mpl.colors.colorConverter.colors[code] = rgb
            mpl.colors.colorConverter.cache[code] = rgb

        _default_prop_cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
            color=[
                colors.get(color, _default_prop_cycle[i])
                for i, k in enumerate(_default_color_cycle)
            ]
        )
    elif palette in ["tab10", "pastel", "muted", "bright", "deep", "colorblind", "dark"]:
        for code, color in zip(_default_color_cycle, sns.color_palette(palette)):
            rgb = mpl.colors.colorConverter.to_rgb(color)
            mpl.colors.colorConverter.colors[code] = rgb
            mpl.colors.colorConverter.cache[code] = rgb
    else:
        sns.set_palette(palette, n_colors=10)

    import getdist

    getdist.chains.print_load_details = print_load_details

    if use_tex and use_svg:
        print("Using latex text with svg output format does not play well. Disable the later one")
        use_svg = False

    if use_svg:
        import matplotlib_inline

        matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

    rc = rc or {"axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}
    if use_tex:
        rc.update({"text.usetex": True})
    if use_seaborn:
        sns.set_theme(rc=rc, style=seaborn_style, context=None if use_tex else seaborn_context)
    else:
        plt.rcParams.update(rc)

    # Fix for jupyter hub @ NERSC
    get_ipython().run_line_magic("matplotlib", backend)


def get_default_settings(colors=None, linewidth=1, num_plot_contours=3):
    """Set default getdist plot settings

    Parameters
    ----------
    colors: list
      list of colors to be used
    linewidth: float
      line width of contours and KDE distributions (default: 2)
    num_plot_contours: int
      number of contour levels (default: 3)
    """
    if not colors:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    from getdist.plots import GetDistPlotSettings

    plot_settings = GetDistPlotSettings()
    plot_settings.solid_colors = colors
    plot_settings.line_styles = colors
    plot_settings.num_plot_contours = num_plot_contours
    plot_settings.linewidth = linewidth
    plot_settings.legend_fontsize = 15
    plot_settings.legend_colored_text = True
    plot_settings.legend_frame = False
    # plot_settings.figure_legend_loc = "best"
    plot_settings.scaling = False
    return plot_settings


def triangle_plot(*args, **kwargs):
    """Overloaded triangle_plot function with additional features"""
    g = get_subplot_plotter(settings=get_default_settings())
    g.triangle_plot(*args, **kwargs)

    if kwargs.get("despine", True):
        despine(g)

    if priors := kwargs.get("priors"):
        show_priors(
            g,
            priors,
            color=kwargs.get("prior_color", "gray"),
            with_legend=kwargs.get("prior_legend", True),
        )

    return g


def plots_1d(*args, **kwargs):
    """Overloaded plots_1d function with additional features"""
    default_plotter_options = {"width_inch": 4}
    plotter_kwargs = {k: kwargs.get(k, v) for k, v in default_plotter_options.items()}

    if legend_labels := kwargs.get("legend_labels"):
        kwargs.update(dict(legend_labels=[]))

    g = get_subplot_plotter(settings=get_default_settings(), **plotter_kwargs)
    g.plots_1d(*args, **kwargs)

    if priors := kwargs.get("priors"):
        show_priors(
            g,
            priors,
            color=kwargs.get("prior_color", "gray"),
            with_legend=kwargs.get("prior_legend", True),
            legend_kwargs=dict(loc=kwargs.get("prior_legend_loc", "upper right")),
        )

    if kwargs.get("despine", True):
        despine(g, all_axes=True)

    if legend_labels:
        default_kwargs = dict(
            bbox_to_anchor=(0.5, 1.025),
            labelcolor="linecolor",
            loc="center",
            fontsize="x-large",
            handlelength=0,
        )
        kwargs = default_kwargs | {
            k.replace("legend_", ""): v for k, v in kwargs.items() if k.startswith("legend")
        }
        kwargs.update(labels=legend_labels)
        g.fig.legend(**kwargs)

    return g


def plots_2d(*args, **kwargs):
    """Overloaded plots_2d function with additional features"""
    default_plotter_options = {"width_inch": 4}
    plotter_kwargs = {k: kwargs.get(k, v) for k, v in default_plotter_options.items()}

    g = get_subplot_plotter(settings=get_default_settings(), **plotter_kwargs)
    g.plots_2d(*args, **kwargs)

    if titles := kwargs.get("titles"):
        for title, ax in zip(titles, g.fig.axes):
            ax.set_title(title)

    return g


def get_mc_samples(
    mcmc_samples,
    prefix="mcmc",
    burnin=0.4,
    no_cache=False,
    as_dict=True,
    selected=None,
    excluded=None,
    select_first=None,
    no_progress_bar=True,
    Rminus1_max_value=None,
):
    """Print MCMC sample size given a set of directories

    Parameters
    ----------
    mcmc_samples: dict
      a dict holding a name as key for the sample and a corresponding directory as value
      or a dict configuration
    prefix: str
      filename prefix (default: 'mcmc')
    burnin: float
      burning fraction (default: 0.4)
    no_cache: bool
      either use or not the pickle cache file
    as_dict: bool
      return a dictionnary holding the kwargs to be used in getdist plot
    selected: list
      list of selected samples
    excluded: list
      list of excluded samples
    select_fist: str
      set the name of the first sample to return
    no_progress_bar: bool
      either enable or disable progress bar from tqdm
    Rminus1_max_value: float
      the maximal accepted R-1 value
    """
    from getdist import loadMCSamples
    from tqdm.auto import tqdm

    from .tools import create_symlink

    create_symlink(mcmc_samples, prefix)

    selected = selected or list(mcmc_samples.keys())
    excluded = excluded or []

    if isinstance(selected, str):
        selected = [selected]
    if isinstance(excluded, str):
        excluded = [excluded]

    for e in excluded:
        selected.remove(e)

    if select_first:
        selected.remove(select_first)
        selected = [select_first] + selected

    default_prefix = prefix
    samples, labels, colors = [], [], []
    regex = re.compile(r".*mcmc\.([0-9]+).progress")

    for name in (pbar := tqdm(selected, disable=no_progress_bar)):
        pbar.set_description(f"Loading '{name}'")
        value = mcmc_samples[name]
        if isinstance(value, str):
            value = dict(path=value)
        path = _get_path(name, value)
        prefix = value.get("prefix", default_prefix)
        labels += [value.get("label", name)]
        colors += [value.get("color", f"C{selected.index(name)}")]
        if Rminus1_max_value:
            exclude = []
            files = _get_chain_filenames(path, prefix, suffix=".progress")
            for fn in files:
                idx = 1 if not (m := regex.match(fn)) else int(m.group(1))
                with open(fn) as f:
                    cols = [a.strip() for a in f.readline().lstrip("#").split()]
                df = pd.read_csv(
                    fn, names=cols, comment="#", sep=" ", skipinitialspace=True, index_col=False
                )
                if df.N.empty:
                    exclude += [idx]
                elif df.Rminus1.iloc[-1] > Rminus1_max_value:
                    exclude += [idx]
                value.update(exclude=exclude)
        try:
            samples += [
                loadMCSamples(
                    os.path.join(path, prefix),
                    settings={"ignore_rows": burnin},
                    no_cache=no_cache,
                    chain_exclude=value.get("exclude"),
                )
            ]
        except OSError:
            print(f"WANRING: No chains of {name} found or fulfill the requirement!")
    if as_dict:
        return samples, dict(legend_labels=labels, colors=colors, diag1d_kwargs={"colors": colors})
    return samples, labels, colors


def show_results(g, results, with_legend=True, legend_kwargs=None):
    """Show results values on a given set of axes
    Parameters
    ----------
    g: getdist.plots
      the getdist plotter instance
    results: dict
      dictionary holding result values. The dict can be either a parameter: loc/scale combo or
      parameter: dict(loc/scale +  and other settings such as color, linestyle)
    with_legend: bool
      add legend
    legend_kwargs: dict
      set legend kwargs
    """
    for name, result in results.items():
        show_priors(
            g,
            result.get("values", result),
            color=result.get("color"),
            ls=result.get("ls", "-"),
            with_legend=False,
        )
    kwargs = dict(
        obj=g.fig.axes[-1],
        labels=results.keys(),
        colors=[v.get("color") for v in results.values()],
        fontsize=10,
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )
    kwargs |= legend_kwargs or {}
    if with_legend:
        add_legend(**kwargs)


def show_priors(
    g, priors, color="gray", ls="--", with_label=False, with_legend=True, legend_kwargs=None
):
    """Show prior values on a given set of axes
    Parameters
    ----------
    g: getdist.plots
      the getdist plotter instance
    priors: dict
      dictionary holding loc/scale value for normal distribution
    color: str
      the color name
    ls: str
      the line style
    with_label: bool
      append parameter name to the legend (default: False)
    with_legend: bool
      add legend
    legend_kwargs: dict
      set legend kwargs
    """
    for par, val in priors.items():
        if not (ax := g.get_axes_for_params(par)):
            continue
        label = rf"${ax.getdist_params[0].label}$ prior" if with_label else "prior"
        if isinstance(val, float):
            ax.axvline(val, color=color, ls=ls)
        else:
            x = np.linspace(*ax.get_xlim(), 100)
            y = stats.norm.pdf(x, *val)
            ax.plot(x, y / y.max(), color=color, ls=ls, label=label)
        if with_legend:
            legend_kwargs = legend_kwargs or dict(loc="upper left", bbox_to_anchor=(1, 1))
            ax.legend(labelcolor="linecolor", **legend_kwargs)


def show_tau_prior(g, loc=0.054, scale=0.0073, color="gray", ls="--"):
    """Dedicated function to plot tau prior

    Parameters
    ----------
    g: getdist.plots
      the getdist plotter instance
    loc: float
      the central tau value
    scale: float
      the scale/sigma of tau value
    color: str
      the color name
    ls: str
      the line style
    """

    show_priors(g, priors=dict(tau=(loc, scale)), color=color, ls=ls)


def despine(g, all_axes=False, **kwargs):
    """Special function to remove left spine in getdist plots

    Parameters
    ----------
    g: getdist.plots
      the getdist plotter instance
    all_axes: bool
      either to remove the spines for all the axes or not
    """
    kwargs = {"left": True} or kwargs
    if all_axes:
        sns.despine(fig=g.fig, **kwargs)
    else:
        sns.despine(ax=g.subplots[0, 0], **kwargs)


def plot_mean_values(
    samples, params, colors=None, palette=None, labels=None, figsize=None, **kwargs
):
    """Plot mean and sigma values from posterior distributions

    Parameters
    ----------
    samples: list
      a list holding getdist MCSamples
    params: dict or list
      a dict holding the parameter names and its default value or
      a unique list of parameter names
    colors: list or str
      the colors of the different markers. If colors == "chi2" then the markers will be colored
      relatively to their chi2 values (if default values are given for parameters)
    palette: str
      the color palette to be used in case no colors are provided
    labels: list
      a list of labels describing each sample
    figsize: tuple
      the figure size
    """
    nparams, nsamples = len(params), len(samples)
    figsize = figsize or (24, 7)
    params = params if isinstance(params, dict) else {k: None for k in params}
    fig, axes = plt.subplots(nrows=1, ncols=nparams, sharey=True, figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0.15)

    use_stats_color = None
    if colors is not None:
        if isinstance(colors, str):
            use_stats_color = colors
            colors = nsamples * [None if use_stats_color in ["chi2", "p-value"] else colors]
    else:
        colors = sns.color_palette(palette, n_colors=nsamples)
    cmap = sns.color_palette(palette or "flare", as_cmap=True)

    shape = (nparams, nsamples)
    chi2s, means, sigmas = np.full(shape, None), np.zeros(shape), np.zeros(shape)
    for i, sample in enumerate(samples):
        marge = sample.getMargeStats()
        latex = sample.getLatex(params)[0]
        for j, name in enumerate(params):
            par = marge.parWithName(name)
            x, xerr = par.mean, par.err
            means[j, i], sigmas[j, i] = x, xerr
            if params.get(name) is not None:
                chi2s[j, i] = ((x - params[name]) / xerr) ** 2
                if use_stats_color == "chi2":
                    colors[i] = cmap(chi2s[j, i])
                if use_stats_color == "p-value":
                    colors[i] = cmap(stats.chi2.sf(chi2s[j, i], 1))
            if axes[j].get_xlabel() == "":
                axes[j].set(xlabel=r"${}$".format(latex[j]), yticks=[])
            *_, bars = axes[j].errorbar(x, i, xerr=xerr, fmt="o", color=colors[i])
            for bar in bars:
                bar.set(alpha=0.5, color=colors[i], linewidth=3)

    # Customize axes and labels
    unique_color = "black" if len(np.unique(colors)) > 1 else colors[0]
    for j, name in enumerate(params):
        mu = np.average(means[j], weights=1 / sigmas[j] ** 2)
        sigma = np.mean(sigmas[j])
        axes[j].axvline(mu, color=unique_color, linestyle="--")
        axes[j].axvspan((mu - sigma), (mu + sigma), color="gray", alpha=0.15)
        if params.get(name) is not None:
            axes[j].spines["left"].set_position(("data", params[name]))
        else:
            axes[j].spines["left"].set_visible(False)
        if not np.all(chi2s[j] == None):
            pvalue = stats.chi2.sf(np.sum(chi2s[j]), nsamples)
            axes[j].set_title(f"$P(\chi^2)$ = {pvalue:.3f}", fontsize=10)

    # Add labels
    labels = labels or kwargs.get("legend_labels")
    if labels is not None:
        ax = axes[0]
        for i, label in enumerate(labels):
            ax.text(
                ax.get_xlim()[0],
                i,
                label,
                va="center",
                ha="right",
                fontsize=8,
                color=unique_color if use_stats_color in ["chi2", "p-value"] else colors[i],
            )
    # Add X² colorbar
    if use_stats_color in ["chi2", "p-value"]:
        from matplotlib import cm, colors

        vmin = chi2s.min() if use_stats_color == "chi2" else 0.0
        vmax = chi2s.max() if use_stats_color == "chi2" else 1.0
        fig.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
            ax=axes,
            label=r"$\chi^2$" if use_stats_color == "chi2" else r"$p$-value",
            shrink=0.8,
        )


def plot_mean_distributions(
    samples, params, priors=None, colors="0.7", return_results=False, nx=None
):
    """Plot mean KDE distributions

    Parameters
    ----------
    samples: list
      a list holding getdist MCSamples
    params: dict or list
      a dict holding the parameter names and its default value or
      a unique list of parameter names
    colors: list or str
      the colors of the different markers. If colors == "chi2" then the markers will be colored
      relatively to their chi2 values (if default values are given for parameters)
    return_results: bool
      return a pandas Dataframe holding the results
    """
    from getdist.plots import get_subplot_plotter

    nsamples = len(samples)
    g = get_subplot_plotter(subplot_size=3, subplot_size_ratio=1.4)
    g.settings.line_styles = (
        nsamples * ["-" + colors] if isinstance(colors, str) else ["-" + color for color in colors]
    )
    nx = nx or len(params)
    g.plots_1d(samples, params, nx=nx, legend_labels=[], lws=2)

    priors = priors or {}
    results = {}
    axes = g.subplots.flatten()
    for i, name in enumerate(params):
        ax = axes[i]
        if not ax:
            continue
        means, sigmas = np.full(nsamples, np.nan), np.full(nsamples, np.nan)
        for j, sample in enumerate(samples):
            marge = sample.getMargeStats()
            par = marge.parWithName(name)
            if par:
                means[j], sigmas[j] = par.mean, par.err
        mu = np.average(means, weights=1 / sigmas**2)
        mu = np.nansum(means / sigmas**2) / np.nansum(1 / sigmas**2)
        # sigma = np.nanmean(sigmas)
        sigma = np.nanstd(means)
        x = np.linspace(*ax.get_xlim(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y / y.max(), color="black", lw=2.5)
        if (param := params.get(name)) is not None:
            if isinstance(param, (tuple, list)):
                y = stats.norm.pdf(x, *param)
                ax.plot(x, y / y.max(), color="black", ls="--", lw=2.5)
            else:
                ax.axvline(param, color="red", lw=3)
            labels, handles = [], []
            if priors.get(name):
                labels = [rf"prior"]
                handles = [Line2D([0], [0], color="blue", ls="--")]

            legend = ax.legend(
                handles,
                labels,
                labelcolor="linecolor",
                bbox_to_anchor=(1, 1),
                loc="upper center",
                title=r"$\frac{{{:.1f}\,\sigma}}{{\sqrt{{N_{{\rm sim}}}}}}$".format(
                    (mu - param) / sigma * np.sqrt(nsamples)
                ),
                title_fontsize=16,
            )
        results[ax.get_xlabel()] = {"value rec.": mu, "$\sigma$ rec.": sigma}
        ax.spines.left.set_visible(False)

    if priors:
        show_priors(g, priors, color="blue", with_legend=False)

    if return_results:
        return pd.DataFrame.from_dict(results, orient="index")


def add_legend(obj, labels=None, colors=None, ls=None, **kwargs):
    """Add legend

    Parameters
    ----------
    obj: figure or axis
      a matplotlib figure or axis
    params: dict or list
      a dict holding the parameter names and its default value or
      a unique list of parameter names
    colors: list or str
      the colors of the different markers. If colors == "chi2" then the markers will be colored
      relatively to their chi2 values (if default values are given for parameters)
    return_results: bool
      return a pandas Dataframe holding the results
    """

    labels = labels or kwargs.get("legend_labels")

    colors = colors or [None for label in labels]
    ls = ls or ["-" for label in labels]
    handles = [Line2D([0], [0], color=colors[i], ls=ls[i]) for i, label in enumerate(labels)]

    leg = obj.legend(
        handles,
        labels,
        bbox_to_anchor=kwargs.get("bbox_to_anchor", (0.5, 1)),
        labelcolor=kwargs.get("labelcolor", "linecolor"),
        loc=kwargs.get("loc", "center"),
        title=kwargs.get("title"),
        ncol=kwargs.get("ncol", 1),
        alignment=kwargs.get("alignment", "left"),
        fontsize=kwargs.get("fontsize", 15),
    )


def plot_correlation_matrix(samples, params, nx=None, figsize=None):
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    samples = np.atleast_1d(samples)

    ncols = nx or len(samples)
    nrows = len(samples) // ncols
    figsize = figsize or 2 * [len(params) // 2]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ax, sample in zip(np.atleast_2d(axes).flatten(), samples):
        latex = [rf"${tex}$" for tex in sample.getLatex(params)[0]]
        corr = sample.corr(params)
        matrix = pd.DataFrame(data=corr, index=latex, columns=latex)

        mask = np.triu(np.ones_like(matrix, dtype=bool))
        sns.heatmap(
            matrix,
            mask=mask,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            annot=True,
            annot_kws={"size": 6},
            fmt=".2f",
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar=False,
        )
    return fig

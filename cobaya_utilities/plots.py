import os
from dataclasses import dataclass
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .tools import _get_path


def set_style(
    use_seaborn=False,
    seaborn_style="ticks",
    seaborn_context="paper",
    palette="tab10",
    use_svg=False,
    use_tex=False,
    print_load_details=False,
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
    rc: dict
      overload matplotlib rc parameters
    """

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

    rc = rc or {"axes.spines.top": False, "axes.spines.right": False}
    if use_tex:
        rc.update({"text.usetex": True})
    if use_seaborn:
        sns.set_theme(rc=rc, style=seaborn_style, context=None if use_tex else seaborn_context)
    else:
        plt.rcParams.update(rc)


def get_default_settings(colors=None, linewidth=2, num_plot_contours=3):
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
    return plot_settings


def get_mc_samples(
    mcmc_samples,
    prefix="mcmc",
    burnin=0.4,
    no_cache=False,
    as_dict=False,
    selected=None,
    excluded=None,
    select_first=None,
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
    """
    from getdist.plots import loadMCSamples

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

    samples, labels, colors = [], [], []
    for name in selected:
        value = mcmc_samples[name]
        path = _get_path(name, value)
        if isinstance(value, dict):
            labels += [value.get("label", name)]
            colors += [value.get("color")]
        else:
            labels += [name]
            colors += [None]

        samples += [
            loadMCSamples(
                os.path.join(path, prefix), settings={"ignore_rows": burnin}, no_cache=no_cache
            )
        ]
    if as_dict:
        return samples, dict(legend_labels=labels, colors=colors, diag1d_kwargs={"colors": colors})
    return labels, samples, colors


def show_inputs(g, inputs, color=None, ls="--"):
    """Show input/reference values on a given set of axes
    Parameters
    ----------
    g: getdist.plots
      the getdist plotter instance
    inputs: dict
      dictionary holding loc/scale value for normal distribution
    color: str
      the color name
    ls: str
      the line style
    """
    for par, val in inputs.items():
        if not (ax := g.get_axes_for_params(par)):
            continue
        if isinstance(val, float):
            ax.axvline(val, color=color, ls=ls)
        else:
            x = np.linspace(*ax.get_xlim(), 100)
            y = stats.norm.pdf(x, *val)
            ax.plot(x, y / y.max(), color=color, ls=ls)


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
    from matplotlib.lines import Line2D

    show_inputs(g, inputs=dict(tau=(loc, scale)), color=color, ls=ls)
    g.get_axes_for_params("tau").legend(
        [Line2D([0], [0], color=color, ls=ls)],
        [r"$\tau$ prior"],
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )


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


def plot_mean_values(samples, params, colors=None, palette=None, labels=None, figsize=None):
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


def plot_mean_distributions(samples, params, colors="0.7", return_results=False, nx=None):
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
    nx = nx or nsamples
    g.plots_1d(samples, params, nx=nx, legend_labels=[], lws=2)

    results = {}
    axes = g.subplots.flatten()
    for i, name in enumerate(params):
        ax = axes[i]
        if not ax:
            continue
        means, sigmas = np.empty(nsamples), np.empty(nsamples)
        for j, sample in enumerate(samples):
            marge = sample.getMargeStats()
            par = marge.parWithName(name)
            means[j], sigmas[j] = par.mean, par.err
        mu = np.average(means, weights=1 / sigmas**2)
        sigma = np.mean(sigmas)
        x = np.linspace(*ax.get_xlim(), 1000)
        y = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y / y.max(), color="black", lw=2.5)
        if params.get(name) is not None:
            param = params.get(name)
            if isinstance(param, (tuple, list)):
                y = stats.norm.pdf(x, *param)
                ax.plot(x, y / y.max(), color="black", ls="--", lw=2.5)
            else:
                ax.axvline(param, color="red", lw=3)
            legend = ax.legend([], loc="upper right")
            legend.set_title(
                r"$\frac{{{:.1f}\,\sigma}}{{\sqrt{{N_{{\rm sim}}}}}}$".format(
                    (mu - param) / sigma * np.sqrt(nsamples)
                ),
                prop={"size": 16},
            )
        results[ax.get_xlabel()] = {"value rec.": mu, "$\sigma$ rec.": sigma}
        ax.spines.left.set_visible(False)

    if return_results:
        return pd.DataFrame.from_dict(results, orient="index")


def add_legend(fig=None, ax=None, labels=None, colors=None, ls=None, **kwargs):
    from matplotlib.lines import Line2D

    if not fig and not ax:
        raise ValueError("Missing either fig or axis instance!")

    colors = colors or [None for label in labels]
    ls = ls or ["-" for label in labels]
    handles = [Line2D([0], [0], color=colors[i], ls=ls[i]) for i, label in enumerate(labels)]

    obj = fig or ax
    leg = obj.legend(
        handles,
        labels,
        bbox_to_anchor=kwargs.get("bbox_to_anchor", (0.5, 1)),
        labelcolor=kwargs.get("labelcolor", "linecolor"),
        loc=kwargs.get("loc", "center"),
        title=kwargs.get("title"),
        ncol=kwargs.get("ncol", 1),
    )
    leg._legend_box.align = "left"

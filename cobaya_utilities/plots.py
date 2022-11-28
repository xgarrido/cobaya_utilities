import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

_use_seaborn = False


def set_style(
    use_seaborn=False, seaborn_style="ticks", use_svg=False, print_load_details=False, **rc
):
    """Set default plot settings

    Parameters
    ----------
    use_seaborn: bool
      use seaborn theming option
    seaborn_style: str
      set seaborn style (default: `ticks`)
    use_svg: bool
      use `svg` output format for figure
    print_load_details: bool
      print load details when getdist loads samples
    rc: dict
      overload matplotlib rc parameters
    """
    import getdist

    getdist.chains.print_load_details = print_load_details

    if use_svg:
        import matplotlib_inline

        matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

    rc = {} or rc
    if use_seaborn:
        _use_seaborn = True
        sns.set_theme(
            rc={"axes.spines.top": False, "axes.spines.right": False, "figure.figsize": (8, 6)},
            style=seaborn_style,
        )


def get_default_settings():
    """Set default getdist plot settings"""
    from getdist.plots import GetDistPlotSettings

    plot_settings = GetDistPlotSettings()
    plot_settings.num_plot_contours = 3
    plot_settings.solid_colors = "tab10"
    plot_settings.line_styles = "tab10"
    plot_settings.linewidth = 2
    plot_settings.legend_fontsize = 15
    plot_settings.legend_colored_text = True
    return plot_settings


def show_inputs(axes, inputs, colors=None):
    """Show inputs values on a given set of axes

    Parameters
    ----------
    axes: array
      array of matplotlib axes
    inputs: dict
      dictionary holding loc/scale value for normal distribution
    colors: list
      list of colors to be applied
    """
    for i, ax in enumerate(axes):
        x = np.linspace(*ax.get_xlim(), 100)
        for j, values in enumerate(inputs):
            mean, sigma = values[i]
            if mean is None:
                continue
            y = stats.norm.pdf(x, mean, sigma)
            color = colors[j] if colors is not None else f"C{j}"
            ax.plot(x, y / np.max(y), color=color, ls="--")


def show_tau_prior(ax, loc=0.054, scale=0.0073):
    """Dedicated function to plot tau prior

    Parameters
    ----------
    ax: matplotlib.axis
      the axis where to plot tau prior
    loc: float
      the central tau value
    scale: float
      the scale/sigma of tau value
    """
    from matplotlib.lines import Line2D

    show_inputs([ax], inputs=[[(loc, scale)]], colors=["gray"])
    ax.legend(
        [Line2D([0], [0], color="gray", ls="--")],
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

    use_chi2_color = False
    if colors is not None:
        if isinstance(colors, str):
            use_chi2_color = colors == "chi2"
            colors = nsamples * [None if use_chi2_color else colors]
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
                if use_chi2_color:
                    colors[i] = cmap(chi2s[j, i])
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
                color=unique_color if use_chi2_color else colors[i],
            )
    # Add X² colorbar
    if use_chi2_color:
        from matplotlib import cm, colors

        fig.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=chi2s.min(), vmax=chi2s.max()), cmap=cmap),
            ax=axes,
            label=r"$\chi^2$",
            shrink=0.8,
        )


def plot_mean_distributions(samples, params, nx=None):
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
    palette: str
      the color palette to be used in case no colors are provided
    labels: list
      a list of labels describing each sample
    figsize: tuple
      the figure size
    """
    from getdist import plots

    nsamples = len(samples)
    g = plots.get_subplot_plotter(subplot_size=3, subplot_size_ratio=1.4)
    g.settings.line_styles = nsamples * ["-0.7"]
    nx = nx or nsamples
    g.plots_1d(samples, params, nx=nx, legend_labels=[], lws=2)

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
        x = np.linspace(*ax.get_xlim(), 1000)
        mu = np.average(means, weights=1 / sigmas**2)
        sigma = np.mean(sigmas)
        y = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y / y.max(), color="black", lw=2.5)
        if params.get(name) is not None:
            ax.axvline(params[name], color="red", lw=3)
            legend = ax.legend([], loc="upper right")
            legend.set_title(
                r"$\frac{{{:.1f}\,\sigma}}{{\sqrt{{N_{{\rm sim}}}}}}$".format(
                    (mu - params[name]) / sigma * np.sqrt(nsamples)
                ),
                prop={"size": 16},
            )
        ax.spines.left.set_visible(False)

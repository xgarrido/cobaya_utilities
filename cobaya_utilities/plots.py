import numpy as np
import seaborn as sns

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
    from scipy.stats import norm

    for i, ax in enumerate(axes):
        x = np.linspace(*ax.get_xlim(), 100)
        for j, values in enumerate(inputs):
            mean, sigma = values[i]
            if mean is None:
                continue
            y = norm.pdf(x, mean, sigma)
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


def despine(g, **kwargs):
    """Special function to remove left spine in getdist triangle plots"""
    kwargs = {"left": True} or kwargs
    sns.despine(ax=g.subplots[0, 0], **kwargs)

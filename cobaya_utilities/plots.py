import numpy as np
import seaborn as sns
from scipy.stats import norm

_use_seaborn = False


def set_style(use_seaborn=False, use_svg=False, print_load_details=False, **rc):
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
            style="ticks",
        )


def get_default_settings():
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
    from matplotlib.lines import Line2D

    show_inputs([ax], inputs=[[(loc, scale)]], colors=["gray"])

    ax.legend(
        [Line2D([0], [0], color="gray", ls="--")],
        [r"$\tau$ prior"],
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )


def despine(g, **kwargs):
    kwargs = {"left": True} or kwargs
    sns.despine(ax=g.subplots[0, 0], **kwargs)

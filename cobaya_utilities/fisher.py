import os
import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns

_default_packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "cobaya_utilities"
)

from getdist.types import NumberFormatter

_formatter = NumberFormatter()
_cached_fisher_matrix = None


def _get_sampled_params(params):
    sampled_params = deepcopy(params)
    sampled_params.update(
        {
            k: {"prior": {"min": 0.9 * v if v != 0 else -0.5, "max": 1.1 * v if v != 0 else +0.5}}
            for k, v in sampled_params.items()
        }
    )
    # Special treatment for logA and As
    if "logA" in params:
        sampled_params["logA"].update({"drop": True})
        sampled_params["As"] = {"value": "lambda logA: 1e-10*np.exp(logA)"}
    return sampled_params


def compute_fisher_matrix(
    likelihood_name,
    likelihood_config=None,
    params=None,
    return_correlation=False,
    epsilon=0.01,
    verbose=True,
    packages_path=None,
):
    """Compute and return the Fisher matrix

    Parameters
    ----------
    likelihood_name: str
      the name of the likelihood to be used
    likelihood_config: dict
      the likelihood config to give to cobaya
    params: dict
      the set of parameters to be used
    return_correlation: bool
      compute and return the correlation matrix rather than the covariance matrix
    epsilon: float
      the relative value to change the parameter when computing derivatives
    verbose: bool
      verbosity flag
    packages_path: str
      path to cobaya packages path
    """
    global _cached_fisher_matrix

    from cobaya.install import install
    from cobaya.log import get_logger
    from cobaya.model import get_model

    logger = get_logger("fisher")
    packages_path = packages_path or _default_packages_path

    likelihood_info = {"likelihood": {likelihood_name: likelihood_config}}
    install(likelihood_info, path=packages_path)

    params = params or {}
    info = {
        **likelihood_info,
        **{
            "params": _get_sampled_params(params),
            "theory": {
                "camb": {"extra_args": {"lens_potential_accuracy": 1}},
                "mflike.BandpowerForeground": None,
            },
        },
    }

    model = get_model(info, packages_path=packages_path)
    likelihood = model.likelihood[likelihood_name]
    theory = model.theory["camb"]
    foregrounds = model.theory["mflike.BandpowerForeground"]

    # First grab the constant params and then update with the sampled one. We finally check for
    # missing parameters
    defaults = model.parameterization.constant_params()
    defaults.update({k: params.get(k) for k in model.parameterization.sampled_params().keys()})
    for k, v in defaults.items():
        if v is None:
            raise ValueError(f"Parameter '{k}' must be set!")

    deriv = {}
    for param in params:

        def _get_power_spectra(epsilon):
            point = defaults.copy()
            point.update(
                {
                    param: (
                        point[param] * (1 + epsilon)
                        if point[param] != 0
                        else point[param] + epsilon
                    )
                }
            )
            model.logposterior(point)
            return likelihood._get_power_spectra(
                theory.get_Cl(ell_factor=True), foregrounds.get_fg_totals(), **point
            )

        delta = (_get_power_spectra(+epsilon) - _get_power_spectra(-epsilon)) / 2 / epsilon
        if defaults[param] != 0:
            delta /= defaults[param]

        if np.all(delta == 0):
            logger.warning(
                f"Sampling a parameter '{param}' that do not have "
                "any effect on power spectra! You should remove it from "
                "cobaya parameter dictionary."
            )
            continue

        deriv[param] = delta
        if verbose:
            logger.info(f"Computing parameter '{param}' done")

    fisher_params = list(deriv.keys())
    nparams = len(fisher_params)
    fisher_matrix = np.empty((nparams, nparams))
    for i1, p1 in enumerate(fisher_params):
        for i2, p2 in enumerate(fisher_params):
            fisher_matrix[i1, i2] = deriv[p1] @ likelihood.inv_cov @ deriv[p2]
    _cached_fisher_matrix = np.linalg.inv(fisher_matrix)

    from cobaya_utilities.utilities import _cosmo_labels

    labels = [
        f"${_cosmo_labels.get(name, model.parameterization.labels().get(name))}$"
        for name in fisher_params
    ]
    values = np.array(list(params.values()))
    sigmas = np.sqrt(np.diag(_cached_fisher_matrix))
    signal_over_noise = values / sigmas

    format_array = lambda array: [_formatter.formatNumber(n) for n in array]
    # [format_array(values), format_array(sigmas), format_array(signal_over_noise)]
    summary = pd.DataFrame(
        data=np.array([values, sigmas, signal_over_noise]).T,
        index=labels,
        columns=["value", r"$\sigma$", "S/N"],
    )
    summary["param"] = list(params.keys())
    _cached_fisher_matrix = pd.DataFrame(data=_cached_fisher_matrix, index=labels, columns=labels)
    if verbose:
        logger.info(f"Computing fisher matrix done")

    if return_correlation:
        return summary, _cached_fisher_matrix.div(np.outer(sigmas, sigmas))
    return summary


def plot_fisher_matrix(matrix=None, use_relplot=True, reset_cache=False, **matrix_args):
    """Compute and return the Fisher matrix

    Parameters
    ----------
    matrix: pandas.Dataframe
      the Fisher matrix to be plotted. In case no matrix is provided then
      the function will call the `compute_fisher_matrix` function
    use_relplot: bool
      use a scatter plot representation rather than a matrix view
    matrix_args: dict
      the paramaters to pass to the `compute_fisher_matrix` function
    reset_cache: bool
      recompute previous Fisher estimation
    """
    global _cached_fisher_matrix
    if reset_cache:
        _cached_fisher_matrix = None

    if _cached_fisher_matrix is not None:
        sigmas = np.sqrt(np.diag(_cached_fisher_matrix))
        matrix = _cached_fisher_matrix / np.outer(sigmas, sigmas)

    if matrix is None:
        matrix_args = matrix_arags.update({"return_correlation": True})
        summary, matrix = compute_fisher_matrix(**matrix_args)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    if use_relplot:
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        corr_mat = matrix.mask(~mask).stack().reset_index(name="correlation")
        with sns.axes_style("whitegrid"):
            g = (
                sns.relplot(
                    data=corr_mat,
                    x="level_0",
                    y="level_1",
                    hue="correlation",
                    size="correlation",
                    palette=cmap,
                    hue_norm=(-1, 1),
                    edgecolor="0.7",
                    height=8,
                    sizes=(50, 200),
                    size_norm=(-1, 1),
                )
                .set(xlabel="", ylabel="", aspect="equal")
                .set_xticklabels(rotation=90)
                .despine(left=True, bottom=True)
            )
            for artist in g.legend.legend_handles:
                artist.set(markeredgecolor="0.7", markeredgewidth=1)
    else:
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        sns.heatmap(
            matrix,
            mask=mask,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
    return g.fig.axes


def generate_yaml_config(
    summary=None,
    sigma_scale=10,
    proposal_scale=0.1,
    with_ref=False,
    ref_scale=1,
    only_params=None,
    filename=None,
    error_if_exists=True,
    **matrix_args,
):
    """Generate cobaya yaml configuration for the sampled parameters

    Parameters
    ----------
    summary: pandas.Dataframe
      the summary product from `compute_fisher_matrix`. In case no summary is provided then
      the function will call the `compute_fisher_matrix` function
    sigma_scale: float
      the number of sigma for min/max values of the prior
    proposal_scale: float
      the proposal is equal to `proposal_scale * sigma`
    with_ref: bool
      generate a reference value normally distributed around the central value
      and sigma equal to `sigma * ref_scale` (see below for ref_scale)
    ref_scale: float
      the normal distribution has scale equal to `ref_scale * sigma`
    only_params: list
      generate yaml configuration only for this set of parameters
    filename: str
      the filename to dump the yaml configuration
    error_if_exists: bool
      raise an error if the yaml filename already exists otherwise erase its content
    matrix_args: dict
      the paramaters to pass to the `compute_fisher_matrix` function
    """

    if summary is None:
        summary = compute_fisher_matrix(**matrix_args)

    fn = lambda n: float(_formatter.formatNumber(n))

    yaml_dict = {}
    for latex, fields in summary.to_dict(orient="index").items():
        name = fields.get("param")
        if only_params and name not in only_params:
            continue
        value = float(fields.get("value"))
        sigma = float(fields.get(r"$\sigma$"))
        latex = latex.replace("$", "")
        prior_min = value - sigma_scale * sigma
        prior_max = value + sigma_scale * sigma
        yaml_dict.update({name: {"prior": {"min": fn(prior_min), "max": fn(prior_max)}}})
        if with_ref:
            yaml_dict[name].update(
                {"ref": {"dist": "norm", "loc": value, "scale": fn(sigma * ref_scale)}}
            )
        yaml_dict[name].update({"proposal": fn(sigma * proposal_scale), "latex": latex})

    from cobaya.yaml import yaml_dump, yaml_dump_file

    if filename:
        yaml_dump_file(filename, yaml_dict, error_if_exists=error_if_exists)
    else:
        print(yaml_dump(yaml_dict))

import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
from getdist.plots import loadMCSamples


def compare(test, ref, current, msg=""):
    test.assertIsInstance(ref, type(current), msg=msg)

    if isinstance(ref, dict):
        for k in ref.keys():
            compare(test, ref[k], current[k], msg=msg)
    elif isinstance(ref, list):
        test.assertEqual(len(ref), len(current), "Not same list length")
        for i in range(len(ref)):
            compare(test, ref[i], current[i], msg=msg)
    elif isinstance(ref, np.ndarray):
        if ref.dtype in (float, int, bool, complex):
            np.testing.assert_almost_equal(ref, current, err_msg=msg)
        else:
            test.assertEqual(len(ref), len(current), "Not same number of objets")
            for i in range(len(ref)):
                compare(test, ref[i], current[i], msg=msg)
    elif isinstance(ref, matplotlib.axes._axes.Axes):
        test.assertEqual(len(ref.lines), len(current.lines), "Not same number of Line2D objets")
        for i in range(len(ref.lines)):
            np.testing.assert_almost_equal(ref.lines[i].get_xdata(), current.lines[i].get_xdata())
            np.testing.assert_almost_equal(ref.lines[i].get_ydata(), current.lines[i].get_ydata())
    elif isinstance(ref, matplotlib.figure.Figure):
        compare(test, ref.axes, current.axes, msg=msg)
    elif isinstance(ref, pd.DataFrame):
        pd.testing.assert_frame_equal(ref, current, rtol=0.1)
    else:
        test.assertTrue(False, f"Data type '{type(ref)}' are not compared!")


data_path = os.path.join(os.path.dirname(__file__), "data")
mcmc_samples = {"Unit test": f"{data_path}/chains"}
params = ["a", "b"]
data = {}


def generate_mcmc():
    # Create fake data and log files
    info = {
        "likelihood": {
            "gaussian_mixture": {
                "means": [0.2, 0],
                "covs": [[0.1, 0.05], [0.05, 0.2]],
            }
        },
        "params": {
            "a": {"prior": {"min": -0.5, "max": 3}, "latex": r"\alpha"},
            "b": {
                "prior": {"dist": "norm", "loc": 0, "scale": 1},
                "ref": 0,
                "proposal": 0.5,
                "latex": r"\beta",
            },
        },
        "sampler": {"mcmc": {"seed": 31415}},
        "output": os.path.join(data_path, "chains/mcmc"),
    }
    from cobaya.run import run

    updated_info, sampler = run(info, debug=f"{data_path}/chains/mcmc.log", force=True)


def generate_pickle(data):
    with open(os.path.join(data_path, "reference.pkl"), "wb") as f:
        pickle.dump(data, f)


def get_reference():
    reference_filename = os.path.join(data_path, "reference.pkl")
    if not os.path.exists(reference_filename):
        import requests

        url = "https://portal.nersc.gov/cfs/sobs/users/xgarrido/cobaya_utilities_data/reference.pkl"
        print(f"Downloading reference data from {url}")
        with open(reference_filename, "wb") as f:
            f.write(requests.get(url).content)

    with open(reference_filename, "rb") as f:
        data_ref = pickle.load(f)
    return data_ref


def generate_chains():
    generate_mcmc()

    from cobaya_utilities import tools

    data.update(
        {
            "plot_chains": tools.plot_chains(mcmc_samples, params),
            "print_chains_size": tools.print_chains_size(mcmc_samples, mpi_run=False).data,
            "plot_progress": tools.plot_progress(mcmc_samples),
            "print_results": tools.print_results(
                [loadMCSamples(f"{path}/mcmc") for path in mcmc_samples.values()],
                params=params,
                labels=mcmc_samples,
            ),
        }
    )


cosmo_params = {
    "cosmomc_theta": 0.0104085,
    "logA": 3.044,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "tau": 0.0544,
}

nuisance_params = {
    "a_tSZ": 3.30,
    "a_kSZ": 1.60,
    "a_p": 6.90,
    "beta_p": 2.08,
    "a_c": 4.90,
    "beta_c": 2.20,
    "a_s": 3.10,
    "a_gtt": 8.70,
    "a_gte": 0.0,
    "a_gee": 0.0,
    "a_psee": 0.0,
    "a_pste": 0.0,
    "xi": 0.10,
    "T_d": 9.60,
}

all_params = {**cosmo_params, **nuisance_params}

likelihood_name = "mflike.TTTEEE"
likelihood_config = {
    "input_file": "LAT_simu_sacc_00000.fits",
    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
}


def generate_fisher():
    from cobaya_utilities import fisher

    summary, correlation = fisher.compute_fisher_matrix(
        likelihood_name, likelihood_config, all_params, return_correlation=True
    )
    data.update(
        {
            "compute_fisher_matrix": {"summary": summary, "correlation": correlation},
            "plot_fisher_matrix": fisher.plot_fisher_matrix(),
        }
    )
    fisher.generate_yaml_config(
        summary, filename=os.path.join(data_path, "fisher_ref.yaml"), error_if_exists=False
    )


def main():
    generate_chains()
    generate_fisher()
    generate_pickle(data)


if __name__ == "__main__":
    main()

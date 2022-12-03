import io
import os
import unittest

from utilities import (
    all_params,
    compare,
    data_path,
    get_reference,
    likelihood_config,
    likelihood_name,
)

from cobaya_utilities import fisher


class FisherTest(unittest.TestCase):
    summary, correlation = fisher.compute_fisher_matrix(
        likelihood_name, likelihood_config, all_params, return_correlation=True
    )

    def setUp(self):
        self.ref = get_reference()

    def test_compute_fisher(self):
        compare(
            self,
            self.ref.get("compute_fisher_matrix"),
            {"summary": self.summary, "correlation": self.correlation},
        )

    def test_plot_fisher_matrix(self):
        compare(self, self.ref.get("plot_fisher_matrix"), fisher.plot_fisher_matrix())

    def test_generate_yaml_config(self):
        ref_yaml_file = os.path.join(data_path, "fisher_ref.yaml")
        current_yaml_file = os.path.join(data_path, "fisher_current.yaml")
        fisher.generate_yaml_config(self.summary, filename=current_yaml_file)
        with io.open(ref_yaml_file) as ref, io.open(current_yaml_file) as current:
            self.assertListEqual(list(ref), list(current))

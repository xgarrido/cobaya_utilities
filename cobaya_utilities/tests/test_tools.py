import unittest

from utilities import compare, data_path, get_reference, mcmc_samples, params

from cobaya_utilities import tools


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.ref = get_reference()

    def test_create_symlink(self):
        tools.create_symlink(mcmc_samples)

    def test_plot_chains(self):
        compare(self, self.ref.get("plot_chains"), tools.plot_chains(mcmc_samples, params))

    def test_print_chains_size(self):
        compare(self, self.ref.get("print_chains_size"), tools.print_chains_size(mcmc_samples).data)

    def test_plot_progress(self):
        compare(self, self.ref.get("plot_progress"), tools.plot_progress(mcmc_samples))

    def test_print_results(self):
        from getdist.plots import loadMCSamples

        compare(
            self,
            self.ref.get("print_results"),
            tools.print_results(
                [loadMCSamples(f"{path}/mcmc") for path in mcmc_samples.values()],
                params=params,
                labels=mcmc_samples,
            ),
        )

import glob
import os


def print_chains_size(chain_dirs, chain_names=None, tablefmt="html"):
    chain_dirs = glob.glob(chain_dirs)

    if chain_names is None:
        chain_names = [os.path.basename(d) for d in chain_dirs]

    nchains = {}
    for i, d in enumerate(chain_dirs):
        key = chain_names[i]
        files = sorted(glob.glob(os.path.join(d, "mcmc.?.txt")))
        nchains[key] = [sum(1 for line in open(f)) for f in files]
        nchains[key] += [sum(nchains[key])]

    from tabulate import tabulate

    return tabulate(
        [(k, *v) for k, v in nchains.items()],
        headers=["mcmc {}".format(i) for i in range(1, 5)] + ["total"],
        tablefmt=tablefmt,
    )

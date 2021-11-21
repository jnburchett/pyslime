#!/usr/bin/python
from pyslime import slime
from pyslime import utils as pu
import numpy as np
from matplotlib import pyplot as plt


def parser(options=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Display slime density plot given a distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "slime_dir",
        type=str,
        default=None,
        help="""Path to folder containing the slime trace. \n
            Ex: SDSS_z=0.1_float32_rerouting-to-data/""",
    )
    parser.add_argument(
        "--datafile",
        type=str,
        default="trace.bin",
        help="""name of the trace binary file. Defaults to 'trace.bin'.""",
    )

    parser.add_argument(
        "--xmin",
        type=float,
        default=-0.5,
        help="""lower lim to plot. Defaults to -0.5""",
    )
    parser.add_argument(
        "--xmax", type=float, default=4, help="""upper lim to plot. Defaults to 4""",
    )

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def dens_plot(
    sdss_slime_dir: str,
    sdss_datafile: str = "trace.bin",
    dtype=np.float32,
    standardize=False,
    xmin=-0.5,
    xmax=4,
    nbins=1000,
):
    sdss_slime32 = pu.get_slime(
        sdss_slime_dir, datafile=sdss_datafile, dtype=dtype, standardize=standardize
    )
    # flatten for histogram
    flatsdssslime = sdss_slime32.flatten()

    bins = np.linspace(xmin, xmax, nbins)
    kwargs = {"bins": bins, "histtype": "step", "density": True}
    _ = plt.figure(figsize=(10, 7))
    plt.hist(flatsdssslime, **kwargs)
    plt.show()


def main(args):
    slime_dir = args.slime_dir

    datafile = args.datafile
    xmin = args.xmin
    xmax = args.xmax
    dens_plot(sdss_slime_dir=slime_dir, datafile=datafile, xmin=xmin, xmax=xmax)


if __name__ == "__main__":
    main()

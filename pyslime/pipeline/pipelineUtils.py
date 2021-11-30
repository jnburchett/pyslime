from pyslime import utils as pu
import numpy as np
from astropy.cosmology import Planck15 as cosmo
import scipy.stats
from itertools import product
from pyslime.slime import Slime
import os
import pickle
import glob
import pandas as pd

# from jax import jit, vmap
# import jax.numpy as np


def closest(arr, value):
    """Return index of array element closest to value. This is
    very slow. Try find nearest. 

    Args:
        arr (np.ndarray): array you want to query from
        value (float, np.ndarray): what you want to find in array.

    Returns:
        int: index where value is closest
    """
    if isinstance(value, int) | isinstance(value, float):
        idx = (np.abs(arr - value)).argmin()
    else:
        idx = []
        for val in value:
            idx.append((np.abs(arr - val)).argmin())
    return idx


def find_nearest(array, value):
    """
    Return index of array element closest to value. Will fail if
    the value is larger than anything in the array. Much faster.

    https://stackoverflow.com/questions/32446703/find-closest-vector-from-a-list-of-vectors-python
    Args:
        array (np.ndarray): array you want to query from
        value (float, np.ndarray): what you want to find in array.

    Returns:
        int: index where value is closest
    """
    idx = np.searchsorted(array, value, side="left")
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx.astype(np.int32)


def bprho_idx_to_dist(xidx, yidx, zidx):
    xycoords_rho = np.linspace(-125 / cosmo.h, 125.0 / cosmo.h, 1024)
    zcoords_rho = np.linspace(0.0, 250.0 / cosmo.h, 1024)
    return xycoords_rho[xidx], xycoords_rho[yidx], zcoords_rho[zidx]


def bprho_dist_to_idx(x_dist, y_dist, z_dist, brick_size=1024):
    xycoords_rho = np.linspace(-125 / cosmo.h, 125.0 / cosmo.h, 1024)
    zcoords_rho = np.linspace(0.0, 250.0 / cosmo.h, brick_size)
    x = find_nearest(xycoords_rho, x_dist)
    y = find_nearest(xycoords_rho, y_dist)
    z = find_nearest(zcoords_rho, z_dist)

    return x, y, z


def get_sim_data(bpDensityFile: str) -> np.ndarray:
    """grab the Bolshoi-Planck simulation data

    Args:
        bpDensityFile (str): location of the BPdenisty binary file

    Returns:
        np.ndarray: log10 simulation density cube
    """
    rho_m = np.fromfile(bpDensityFile, dtype=np.float64)
    rho_m = np.reshape(rho_m, (1024, 1024, 1024))
    logrhom = np.log10(rho_m)
    return logrhom


def sample_bins(
    bpslime: Slime,
    logrhom: np.ndarray,
    smrhobins: np.ndarray,
    verbose: bool = True,
    size: int = 2000,
):
    """Sample the slime mold by binning in density. This allows us
    to look at every density regime in slime mold fits and find the
    corresponding density in the simulation. 

    Args:
        bpslime (Slime): SlimeObject that fits to the simulation
        logrhom (np.ndarray): BP simulation density cube
        smrhobins (np.ndarray): how to bin up the slime mold density
        verbose (bool, optional): print? Defaults to True.
        size (int, optional): number of samples to take in each bin 

    Returns:
        List: bpdistribs_sm 
        List: smdistribs_sm
    """

    smdistribs_sm = []
    bpdistribs_sm = []
    for i, dv in enumerate(smrhobins):
        if dv != smrhobins[-1]:
            these = np.where((bpslime.data > dv) & (bpslime.data < smrhobins[i + 1]))
        else:
            continue
        print(dv, len(these[0]))
        try:
            randidxs = np.random.randint(0, len(these[0]), size=size)
            randidxsx = these[0][randidxs]
            randidxsy = these[1][randidxs]
            randidxsz = these[2][randidxs]
        except:
            randidxsx, randidxsy, randidxsz = these

        xdist, ydist, zdist = pu.idx_to_cartesian(
            randidxsx, randidxsy, randidxsz, slime=bpslime
        )
        bpdensvals = np.zeros(len(xdist))
        bpidx_x, bpidx_y, bpidx_z = bprho_dist_to_idx(xdist, ydist, zdist)
        bpdensvals = logrhom[bpidx_x, bpidx_y, bpidx_z]
        bpdistribs_sm.append(bpdensvals)
        smdistribs_sm.append(bpslime.data[randidxsx, randidxsy, randidxsz])

    return bpdistribs_sm, smdistribs_sm


def vmap_sample_bins(
    bpslime: Slime,
    logrhom: np.ndarray,
    smrhobins: np.ndarray,
    verbose: bool = True,
    size: int = 2000,
):
    return sample_bins(
        bpslime=bpslime, logrhom=logrhom, smrhobins=smrhobins, verbose=True, size=size
    )


def distribution_stats(bpdistribs_sm: list, bootstrap: bool = False):
    """Find # Find the mean, median and 1$\sigma$ std for each slime mold density bin

    Args:
        bpdistribs_sm (list): list of arrays containing the bp dens distributions.
        bootstrap (bool, optional): compute boostrapping error estimates. Defaults to False.

    Returns:
        tuple: medvals_bp, meanvals_bp, stdvals_bp, loperc_bp, hiperc_bp
    """
    medvals_bp = np.zeros(len(bpdistribs_sm))
    meanvals_bp = np.zeros(len(bpdistribs_sm))
    stdvals_bp = np.zeros(len(bpdistribs_sm))
    loperc_bp = np.zeros(len(bpdistribs_sm))
    hiperc_bp = np.zeros(len(bpdistribs_sm))
    if bootstrap:
        bsmederrs_bp = np.zeros(len(bpdistribs_sm))
        bsmeanerrs_bp = np.zeros(len(bpdistribs_sm))

    for i in range(len(bpdistribs_sm)):
        medvals_bp[i] = np.median(bpdistribs_sm[i])
        meanvals_bp[i] = np.mean(bpdistribs_sm[i])
        stdvals_bp[i] = np.std(bpdistribs_sm[i][bpdistribs_sm[i] != 0])
        try:
            if bootstrap:
                compute_bootstrap(bpdistribs_sm, bsmederrs_bp, bsmeanerrs_bp, i)

            loperc_bp[i] = np.percentile(bpdistribs_sm[i], 16.0)
            hiperc_bp[i] = np.percentile(bpdistribs_sm[i], 84.0)
        except:
            continue
    if bootstrap:
        return (
            medvals_bp,
            meanvals_bp,
            stdvals_bp,
            loperc_bp,
            hiperc_bp,
            bsmederrs_bp,
            bsmeanerrs_bp,
        )
    return medvals_bp, meanvals_bp, stdvals_bp, loperc_bp, hiperc_bp


def compute_bootstrap(bpdistribs_sm, bsmederrs_bp, bsmeanerrs_bp, i):
    from astropy import stats as astats

    bsmederrs_bp[i] = np.std(
        astats.bootstrap(bpdistribs_sm[i], bootnum=500, bootfunc=np.median)
    )
    bsmeanerrs_bp[i] = np.std(
        astats.bootstrap(bpdistribs_sm[i], bootnum=500, bootfunc=np.mean)
    )


def costfunc(
    u_values: np.ndarray,
    v_values: np.ndarray,
    u_weights: np.ndarray,
    v_weights: np.ndarray,
    denscut: float = 0.5,
) -> float:
    """Compute the cost function between two density distributions. 
    This method uses the Wasserstein distance between distributions U and V as 
    implemented by scipy.

    Args:
        u_values (np.ndarray): bin centers of the U histogram.
        v_values (np.ndarray): bin centers of the V histogram.
        u_weights (np.ndarray): number of objects in each bin for the U hist.
        v_weights (np.ndarray): number of objects in each bin for the V hist.
        denscut (float): upper limit on density to ignore in fitting. 

    Returns:
        float: wasserstein cost
    """
    # do a cut on the data before computing cost
    denscut = v_values > denscut
    v_weights_cut = v_weights[denscut]
    v_values_cut = v_values[denscut]

    if len(v_values_cut) == 0:
        return np.inf

    cost = scipy.stats.wasserstein_distance(
        u_values, v_values_cut, u_weights=u_weights, v_weights=v_weights_cut
    )
    return cost


def objective_function(
    stretch: float,
    shift: float,
    u_values: np.ndarray,
    v_values: np.ndarray,
    u_weights: np.ndarray,
    v_weights: np.ndarray,
    denscut: float,
) -> float:
    """The objective function for the linear transformation of a distribution.
     newdist = stretch*dist + shift or y = mx + b


    Args:
        stretch (float): scale the distribution.
        shift (float): bias the distribution.
        u_values (np.ndarray): bin centers of the U histogram.
        v_values (np.ndarray): bin centers of the V histogram.
        u_weights (np.ndarray): number of objects in each bin for the U hist.
        v_weights (np.ndarray): number of objects in each bin for the V hist.
        denscut (float): upper limit on density to ignore in fitting. 


    Returns:
        float: [description]
    """
    new_vvaleus = stretch * v_values + shift
    return costfunc(u_values, new_vvaleus, u_weights, v_weights, denscut)


def calc_stretch_shift(
    bpslime_dir: str,
    survey_slime_dir: str,
    dens_threshold: float = 0.5,
    bpslime_datafile: str = "trace.bin",
    survey_datafile: str = "trace.bin",
    stretchmin: float = 0.1,
    strectmax: float = 3.0,
    shiftmin: float = -2,
    shiftmax: float = 2,
) -> tuple:
    """The function estimates the optimal linear operator that aligns the two
    empirical distributions. It can funciton on only a portion
    of the distributions. V is the source distribution, U is the target 
    distribution. Uses the wasserstein distance to estimate the a cost function
    and finds the optimal solution. 
    

    .. math::
        y(x)=Ax+b

    where :
        A is the stretch
        b is the shift, or bias

    Args:
        bpslime_dir (str): path to the fit to the bp slime
        survey_slime_dir (str): path to the fit to the survey, sdss or LRG
        dens_threshold (float, optional): ignore densities below this. Defaults to 0.5.
        bpslime_datafile (str, optional): bp binary datafile. Defaults to "trace.bin".
        survey_datafile (str, optional): survey binary datafile. Defaults to "trace.bin".
        stretchmin (float, optional): lower bound on stretch. Defaults to 0.1.
        strectmax (float, optional): upper bound on stretch. Defaults to 3.0.
        shiftmin (float, optional): lower bound on shift. Defaults to -2.
        shiftmax (float, optional): upper bound on shift. Defaults to 2.

    Returns:
        tuple: (stretch, shift)
    """

    # load the bp fit and survey data
    survey_slime = pu.get_slime(
        survey_slime_dir, datafile=survey_datafile, dtype=np.float32, standardize=False
    )
    bpslime = pu.get_slime(
        bpslime_dir, datafile=bpslime_datafile, dtype=np.float32, standardize=False
    )

    flatbpslime = bpslime.data.flatten()
    flatsdssslime = survey_slime.data.flatten()
    bins = np.linspace(-5, 5, 1000)
    uweights, uvalues_edges = np.histogram(flatbpslime, bins=bins)
    vweights, vvalues_edges = np.histogram(flatsdssslime, bins=bins)

    # find the centers of the bins
    uvalues = 0.5 * uvalues_edges[:-1] + 0.5 * uvalues_edges[1:]
    vvalues = 0.5 * vvalues_edges[:-1] + 0.5 * vvalues_edges[1:]

    # since we are fixing the bp (u_values)
    denscut = uvalues > -10
    uweights_cut = uweights[denscut]
    uvalues_cut = uvalues[denscut]

    # take all of the bpdata
    vdenscut = vvalues > dens_threshold
    vweights_cut = vweights[vdenscut]
    vvalues_cut = vvalues[vdenscut]

    stretch, shift = _calc_stretch_shift(
        uvalues_cut,
        vvalues_cut,
        uweights_cut,
        vweights_cut,
        stretchmin=stretchmin,
        strectmax=strectmax,
        shiftmin=shiftmin,
        shiftmax=shiftmax,
        denscut=dens_threshold,
    )

    return stretch, shift


def _calc_stretch_shift(
    uvalues_cut,
    vvalues_cut,
    uweights_cut,
    vweights_cut,
    stretchmin=0.1,
    strectmax=3.0,
    shiftmin=-2,
    shiftmax=2,
    denscut=0.5,
):
    stretch = np.geomspace(stretchmin, strectmax, 110)
    shift = np.linspace(shiftmin, shiftmax, 120)

    costarr = np.zeros((110, 120))
    for p in product(enumerate(stretch), enumerate(shift)):
        [i, stretchval], [j, shiftval] = p
        f = objective_function(
            stretchval,
            shiftval,
            uvalues_cut,
            vvalues_cut,
            uweights_cut,
            vweights_cut,
            denscut=denscut,
        )
        costarr[i, j] = f
    # find the minimum
    idx, jdx = np.where(costarr == costarr.min())
    return stretch[idx], shift[jdx]


def calc_map_bp_slime(bpDensityFile, bpslimedir, bpdatafile, out_pickle_file):

    if os.path.exists(out_pickle_file):
        # dont run this, its VERY slow.
        print(
            f"You already have a mapping pickle file with this name {out_pickle_file}. \
        if you wish to run the mapping again, delete this file"
        )
    else:
        # load data
        logrhom = get_sim_data(bpDensityFile)
        bpslime = pu.get_slime(
            bpslimedir, datafile=bpdatafile, dtype=np.float32, standardize=False
        )
        # Sample the densites from the slime fit to the simulation and get
        # the corresponding density from the simulation.
        mindens = bpslime.data[~np.isinf(bpslime.data)].min() - 0.1
        maxdens = bpslime.data.max() + 0.2
        smrhobins = np.arange(mindens, maxdens, 0.1)
        bpdistribs_sm, smdistribs_sm = vmap_sample_bins(bpslime, logrhom, smrhobins)

        # Find the median and 1sigma std for each slime mold density bin
        (
            medvals_bp,
            meanvals_bp,
            stdvals_bp,
            loperc_bp,
            hiperc_bp,
        ) = distribution_stats(bpdistribs_sm)

        # pickle the results
        packageDict_smBins = {
            "medvals_bp": medvals_bp,
            "meanvals_bp": meanvals_bp,
            "loperc_bp": loperc_bp,
            "hiperc_bp": hiperc_bp,
            "bpdistribs_sm": bpdistribs_sm,
            "smdistribs_sm": smdistribs_sm,
            "smrhobins": smrhobins,
        }
        pickle.dump(packageDict_smBins, open(out_pickle_file, "wb"))

        return None


def _get_hist(data, bins):
    weights, values_edges = np.histogram(data, bins=bins, density=True)
    values = 0.5 * values_edges[:-1] + 0.5 * values_edges[1:]
    return weights, values


def get_datafolders(datadir):
    everything = glob.glob(datadir + "*")
    datafolders = [f for f in everything if ".zip" not in f]
    datafolders = [f for f in datafolders if ".csv" not in f]
    return datafolders


def get_hist(file):
    slime = pu.get_slime(file, dtype=np.float32, standardize=False)
    slimedata = slime.data.ravel()
    bpmin = slimedata[~np.isinf(slimedata)].min()
    bins = np.linspace(bpmin, np.max(slimedata), 10000)
    uweights, uvalues = _get_hist(slimedata, bins=bins)
    return uweights, uvalues


def make_distribution_files(datadir, packagedir):
    """ Since the slime fit files are huge and slow, 
    extract their distributions to make it easer to linear transform
    them.

    Args:
        datafolders (str): path to data folders
        packagedir (str): where to store the distribution numpy files
    """
    datafolders = get_datafolders(datadir)

    for file in datafolders:
        # name = file.replace("=", "_")
        # name = name.replace(".", "_")
        name = file[57:]
        outfilename = packagedir + name + "_dist.npz"
        if not os.path.exists(outfilename):
            print(f"working on {name}")
            w, v = get_hist(file)
            print(f"writing {outfilename}")

            np.savez(outfilename, w, v)
        else:
            print(f"{outfilename} already exists")


def calc_stretch_shift_df(distlist, plot=True):

    bpdata = np.load(distlist[0])
    uweight = bpdata["arr_0"]
    uvalues = bpdata["arr_1"]
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(uvalues, uweight, label=distlist[0][51:])

    lintransform_list = []
    for file in distlist[1:]:
        data = np.load(file)
        w = data["arr_0"]
        v = data["arr_1"]
        name = file[51:]
        stretch, shift = _calc_stretch_shift(
            uvalues,
            v,
            uweight,
            w,
            stretchmin=0.5,
            strectmax=1,
            shiftmin=-1,
            shiftmax=1,
            denscut=-10,
        )
        savedict = {
            "name": name.strip("_dist.npz"),
            "stretch": stretch[0],
            "shift": shift[0],
        }
        lintransform_list.append(savedict)
        if plot:
            plt.plot(v * stretch + shift, w, label=name)
    if plot:
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()
    df = pd.DataFrame(lintransform_list)
    return df


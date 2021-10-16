from pyslime import utils as pu
import numpy as np
from astropy.cosmology import Planck15 as cosmo
import scipy.stats
from itertools import product
from pyslime.slime import Slime


def closest(arr, value):
    # Return index of array element closest to value
    if isinstance(value, int) | isinstance(value, float):
        idx = (np.abs(arr - value)).argmin()
    else:
        idx = []
        for val in value:
            idx.append((np.abs(arr - val)).argmin())
    return idx


def bprho_idx_to_dist(xidx, yidx, zidx):
    xycoords_rho = np.linspace(-125 / cosmo.h, 125.0 / cosmo.h, 1024)
    zcoords_rho = np.linspace(0.0, 250.0 / cosmo.h, 1024)
    return xycoords_rho[xidx], xycoords_rho[yidx], zcoords_rho[zidx]


def bprho_dist_to_idx(x_dist, y_dist, z_dist, brick_size=1024):
    xycoords_rho = np.linspace(-125 / cosmo.h, 125.0 / cosmo.h, 1024)
    zcoords_rho = np.linspace(0.0, 250.0 / cosmo.h, 1024)
    x = int(closest(xycoords_rho, x_dist))
    y = int(closest(xycoords_rho, y_dist))
    z = int(closest(zcoords_rho, z_dist))

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
    bpslime: Slime, logrhom: np.ndarray, smrhobins: np.ndarray, verbose: bool = True
):
    """Sample the slime mold by binning in density. This allows us
    to look at every density regime in slime mold fits and find the
    corresponding density in the simulation. 

    Args:
        bpslime (Slime): SlimeObject that fits to the simulation
        logrhom (np.ndarray): BP simulation density cube
        smrhobins (np.ndarray): how to bin up the slime mold density
        verbose (bool, optional): print? Defaults to True.

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
            randidxs = np.random.randint(0, len(these[0]), size=2000)
            randidxsx = these[0][randidxs]
            randidxsy = these[1][randidxs]
            randidxsz = these[2][randidxs]
        except:
            randidxsx, randidxsy, randidxsz = these

        xdist, ydist, zdist = pu.idx_to_cartesian(
            randidxsx, randidxsy, randidxsz, slime=bpslime
        )
        bpdensvals = np.zeros(len(xdist))
        for j, xd in enumerate(xdist):
            bpidx_x, bpidx_y, bpidx_z = bprho_dist_to_idx(xd, ydist[j], zdist[j])
            bpdensvals[j] = logrhom[bpidx_x, bpidx_y, bpidx_z]
        bpdistribs_sm.append(bpdensvals)
        smdistribs_sm.append(bpslime.data[randidxsx, randidxsy, randidxsz])

    return bpdistribs_sm, smdistribs_sm


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
    denscut: float = 0.5,
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
from pyslime import utils as pu
import numpy as np
from astropy.cosmology import Planck15 as cosmo

from pyslime.slime import Slime

# Return index of array element closest to value


def closest(arr, value):
    if (isinstance(value, int) | isinstance(value, float)):
        idx = (np.abs(arr-value)).argmin()
    else:
        idx = []
        for val in value:
            idx.append((np.abs(arr-val)).argmin())
    return idx


def bprho_idx_to_dist(xidx, yidx, zidx):
    xycoords_rho = np.linspace(-125/cosmo.h, 125./cosmo.h, 1024)
    zcoords_rho = np.linspace(0., 250./cosmo.h, 1024)
    return xycoords_rho[xidx], xycoords_rho[yidx], zcoords_rho[zidx]


def bprho_dist_to_idx(x_dist, y_dist, z_dist, brick_size=1024):
    xycoords_rho = np.linspace(-125/cosmo.h, 125./cosmo.h, 1024)
    zcoords_rho = np.linspace(0., 250./cosmo.h, 1024)
    x = int(closest(xycoords_rho, x_dist))
    y = int(closest(xycoords_rho, y_dist))
    z = int(closest(zcoords_rho, z_dist))

    return x, y, z


def get_sim_data(bpDensityFile: str) -> np.array:
    """grab the Bolshoi-Planck simulation data

    Args:
        bpDensityFile (str): location of the BPdenisty binary file

    Returns:
        np.array: log10 simulation density cube
    """
    rho_m = np.fromfile(bpDensityFile, dtype=np.float64)
    rho_m = np.reshape(rho_m, (1024, 1024, 1024))
    logrhom = np.log10(rho_m)
    return logrhom


def sample_bins(bpslime: Slime, logrhom: np.array,
                smrhobins: np.array, verbose: bool = True):
    """Sample the slime mold by binning in density. This allows us
    to look at every density regime in slime mold fits and find the
    corresponding density in the simulation. 

    Args:
        bpslime (Slime): SlimeObject that fits to the simulation
        logrhom (np.array): BP simulation density cube
        smrhobins (np.array): how to bin up the slime mold density
        verbose (bool, optional): print? Defaults to True.

    Returns:
        List: bpdistribs_sm 
        List: smdistribs_sm
    """

    smdistribs_sm = []
    bpdistribs_sm = []
    for i, dv in enumerate(smrhobins):
        if dv != smrhobins[-1]:
            these = np.where((bpslime.data > dv) & (
                bpslime.data < smrhobins[i+1]))
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
            randidxsx, randidxsy, randidxsz, slime=bpslime)
        bpdensvals = np.zeros(len(xdist))
        for j, xd in enumerate(xdist):
            bpidx_x, bpidx_y, bpidx_z = bprho_dist_to_idx(
                xd, ydist[j], zdist[j])
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
                compute_bootstrap(bpdistribs_sm,
                                  bsmederrs_bp, bsmeanerrs_bp, i)

            loperc_bp[i] = np.percentile(bpdistribs_sm[i], 16.)
            hiperc_bp[i] = np.percentile(bpdistribs_sm[i], 84.)
        except:
            continue
    if bootstrap:
        return medvals_bp, meanvals_bp, stdvals_bp, loperc_bp, hiperc_bp, bsmederrs_bp, bsmeanerrs_bp
    return medvals_bp, meanvals_bp, stdvals_bp, loperc_bp, hiperc_bp


def compute_bootstrap(bpdistribs_sm, bsmederrs_bp, bsmeanerrs_bp, i):
    from astropy import stats as astats
    bsmederrs_bp[i] = np.std(astats.bootstrap(
        bpdistribs_sm[i], bootnum=500, bootfunc=np.median))
    bsmeanerrs_bp[i] = np.std(astats.bootstrap(
        bpdistribs_sm[i], bootnum=500, bootfunc=np.mean))

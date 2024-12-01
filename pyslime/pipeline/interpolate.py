import numpy as np
import pickle


class MapFuncp3:
    def __init__(self, x):
        pass

    pass


def interpolate(mapping_data_pickle_file, mapfunc_pickle_file):
    """Fits a univeriate spline to data to create a function
    to map the slime density to cosmic overdenstiy.

    Args:
        mapping_data_pickle_file (str): location of the mapping binned pickle
        mapfunc_pickle_file ([type]): location to put mapfunc pickle
    """

    smdens, bpdens = slurp_data(mapping_data_pickle_file)

    # constrain where to do the fitting: only past the break
    xbreak = find_xbreak(smdens, bpdens)
    lin_cut = smdens > xbreak - 0.5

    xfit = smdens[lin_cut]
    yfit = bpdens[lin_cut]

    xbreak = find_xbreak(smdens, bpdens)
    y_const = find_y_const(smdens, bpdens, xbreak)
    lin_cut = smdens > xbreak - 0.5
    xfit = smdens[lin_cut]
    yfit = bpdens[lin_cut]

    z = np.polyfit(xfit, yfit, 3)
    pfit = np.poly1d(z)

    outdict = {}
    outdict["xbreak"] = xbreak
    outdict["y_const"] = y_const
    outdict["pfit"] = pfit

    pickle.dump(outdict, open(mapfunc_pickle_file, "wb"))


def slurp_data(mapping_data_pickle_file):
    # read in the mapping pickle file
    smpackage = pickle.load(open(mapping_data_pickle_file, "rb"))
    smrhobins = smpackage["smrhobins"]
    midbins = 0.5 * smrhobins[:-1] + 0.5 * smrhobins[1:]

    # get rid of empty edge bins
    nonan = ~np.isnan(smpackage["medvals_bp"])

    smdens = midbins[nonan]
    bpdens = smpackage["medvals_bp"][nonan]
    return smdens, bpdens


def find_xbreak(smdens, bpdens, plot=False):
    break_cut = (smdens > 0) & (smdens < 2)
    xmiddiff = 0.5 * smdens[break_cut][1:] + 0.5 * smdens[break_cut][:-1]
    diff = np.diff(bpdens[break_cut])
    xbreak = xmiddiff[np.argmax(diff)]
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(xmiddiff, diff)
        plt.axvline(xbreak)
    return xbreak


def find_y_const(smdens, bpdens, xbreak):
    y_const = np.mean(bpdens[smdens < xbreak])
    return y_const


def relu_p3(x, xbreak, y_const, pfit):
    y = np.where(x < xbreak - 0.3, y_const, pfit(x))
    return np.maximum(y_const, y)


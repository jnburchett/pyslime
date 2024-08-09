from astropy.table import Table, vstack, unique, hstack, join
from pyslime import slime
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from pyslime.pipeline import interpolate


def make_subcatalog(
    slimeobj_list,
    lumdist_cutoffs,
    slice_flag,
    xbreak,
    y_const,
    pfit,
    tab,
    diagnostics=False,
):
    tablist = []
    for slimeobj, lumcut, flag in zip(slimeobj_list, lumdist_cutoffs, slice_flag):

        if "NGC" in slimeobj.name:
            subtable = tab[tab["survey"] == "LRG_NGC"]
        elif "SGC" in slimeobj.name:
            subtable = tab[tab["survey"] == "LRG_SGC"]
        elif "SDSS" in slimeobj.name:
            subtable = tab[tab["survey"] == "SDSS"]
        else:
            print("there is no subtable, chck this!!!!")

        t = Table()
        t["ra"] = subtable["RA"]
        t["dec"] = subtable["DEC"]
        t["z"] = subtable["Z"]
        t["mstars"] = subtable["MassStars"]
        t["lumdist"] = subtable["lumDist"]
        t["MCPM_RUN"] = flag

        cut = t[(t["lumdist"] > lumcut[0]) & (t["lumdist"] < lumcut[1])]
        galtrace = slimeobj.sky_to_data(
            cut["ra"], cut["dec"], cut["z"], check_bounds=True
        )
        if diagnostics:
            # diagnostics
            infcount = np.sum(np.isinf(galtrace))
            misscount = sum(np.isclose(galtrace, -642.0))
            print("+" * len(slimeobj.name))
            print(slimeobj.name)
            print(f"lum cuts: {lumcut}")
            print(f"num gals within lumcut {len(cut)}")
            print(f"there are {infcount} -infs")
            print(f"there are {misscount} -642s")
            print(f"inf fraction = {infcount/len(cut)}")
            print(f"642 fraction = {misscount/len(cut)}")
            print("+" * len(slimeobj.name))
            bins = np.linspace(-1, 4, 500)
            s = slimeobj.name
            idx = s.rfind("/")
            fname = s[idx:]
            plt.hist(galtrace, bins=bins, histtype="step", label=fname, density=True)
            plt.legend()
            plt.savefig(f"/Users/mwilde/Desktop/test_plots/{fname}_dist.pdf")

        cut["log10_slimedens"] = galtrace
        cut["log10_overdens"] = interpolate.relu_p3(galtrace, xbreak, y_const, pfit,)

        tablist.append(cut)

    return tablist


def make_hi_lo_z_lists(slime_dirs):
    name_list = []
    slimeobj_list_lo = []
    slimeobj_list_hi = []
    slice_flag_lo = []
    slice_flag_hi = []
    lumdist_cutoffs_lo = []
    lumdist_cutoffs_hi = []
    for s in slime_dirs:
        slime_obj = slime.Slime.get_slime(s, standardize=True)
        survey_idx = s.rfind("/") + 1
        survey = s[survey_idx:]

        slice_dist_idx = s.rfind("=") + 1
        num = int(s[slice_dist_idx])

        numidx = s.rfind("-")
        numidx_end = s.rfind("mpc")
        lowend = int(s[slice_dist_idx:numidx])
        hiend = int(s[numidx + 1 : numidx_end])
        print(survey, [lowend, hiend])
        name_list.append([survey, num, [lowend, hiend]])
        if "SGC" in survey:
            if num == 0:
                slimeobj_list_lo.append(slime_obj)
                slice_flag_lo.append(2)
                lumdist_cutoffs_lo.append([lowend, hiend])

            if num == 9:
                slimeobj_list_lo.append(slime_obj)
                slice_flag_lo.append(4)
                lumdist_cutoffs_lo.append([lowend, hiend])
            if num == 1:
                slimeobj_list_hi.append(slime_obj)
                slice_flag_hi.append(6)
                lumdist_cutoffs_hi.append([lowend, hiend])
            if num == 2:
                slimeobj_list_hi.append(slime_obj)
                slice_flag_hi.append(8)
                lumdist_cutoffs_hi.append([lowend, hiend])
        if "NGC" in survey:
            if num == 0:
                slimeobj_list_lo.append(slime_obj)
                slice_flag_lo.append(1)
                lumdist_cutoffs_lo.append([lowend, hiend])

            if num == 9:
                slimeobj_list_lo.append(slime_obj)
                slice_flag_lo.append(3)
                lumdist_cutoffs_lo.append([lowend, hiend])
            if num == 1:
                slimeobj_list_hi.append(slime_obj)
                slice_flag_hi.append(5)
                lumdist_cutoffs_hi.append([lowend, hiend])
            if num == 2:
                slimeobj_list_hi.append(slime_obj)
                slice_flag_hi.append(7)
                lumdist_cutoffs_hi.append([lowend, hiend])
        if "SDSS" in survey:
            slimeobj_list_lo.append(slime_obj)
            slice_flag_lo.append(0)
            lumdist_cutoffs_lo.append([lowend, hiend])
    return (
        name_list,
        slimeobj_list_lo,
        slimeobj_list_hi,
        slice_flag_lo,
        slice_flag_hi,
        lumdist_cutoffs_lo,
        lumdist_cutoffs_hi,
    )


def load_mapfuncs(pipedatadir):
    # load the spline func used to map
    with open(pipedatadir + "mapfunc_z0p0.pickle", "rb") as f:
        mapdict0 = pickle.load(f)
    xbreak0 = mapdict0["xbreak"]
    y_const0 = mapdict0["y_const"]
    pfit0 = mapdict0["pfit"]

    with open(pipedatadir + "mapfunc_z0p5.pickle", "rb") as f:
        mapdict5 = pickle.load(f)
    xbreak5 = mapdict5["xbreak"]
    y_const5 = mapdict5["y_const"]
    pfit5 = mapdict5["pfit"]
    return xbreak0, y_const0, pfit0, xbreak5, y_const5, pfit5


def make_galaxy_cat_with_slimedens(datadir, pipedatadir):
    slime_dirs = glob.glob(datadir + "/*mpc")
    slime_dirs = [s for s in slime_dirs if ".zip" not in s]
    slime_dirs = [s for s in slime_dirs if ".csv" not in s]
    slime_dirs.sort()
    # print(slime_dirs)

    cat = Table.read(datadir + "galaxy_cat_forSlime.csv", format="ascii.ecsv")
    xbreak0, y_const0, pfit0, xbreak5, y_const5, pfit5 = load_mapfuncs(pipedatadir)

    (
        name_list,
        slimeobj_list_lo,
        slimeobj_list_hi,
        slice_flag_lo,
        slice_flag_hi,
        lumdist_cutoffs_lo,
        lumdist_cutoffs_hi,
    ) = make_hi_lo_z_lists(slime_dirs)
    tablist_lo = make_subcatalog(
        slimeobj_list_lo,
        lumdist_cutoffs_lo,
        slice_flag_lo,
        xbreak0,
        y_const0,
        pfit0,
        cat,
    )
    tablist_hi = make_subcatalog(
        slimeobj_list_hi,
        lumdist_cutoffs_hi,
        slice_flag_hi,
        xbreak5,
        y_const5,
        pfit5,
        cat,
    )

    lo = vstack(tablist_lo)
    hi = vstack(tablist_hi)
    tab = vstack([lo, hi])
    tab.write(datadir + "prevac_catalog.csv", format="ascii.csv")
    return tab


def xmatch(
    table1,
    table2,
    ra1_key,
    dec1_key,
    ra2_key,
    dec2_key,
    units="deg",
    max_sep=1.0 * u.arcsec,
):
    import pandas as pd
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    # convert to astropy
    if not isinstance(table1, Table):
        if isinstance(table1, pd.DataFrame):
            table1 = Table.from_pandas(table1)
        else:
            print("table1 must be pandas or astropy table")

    if not isinstance(table2, Table):
        if isinstance(table2, pd.DataFrame):
            table2 = Table.from_pandas(table2)
        else:
            print("table2 must be pandas or astropy table")

    ra1 = np.array(table1[ra1_key])
    dec1 = np.array(table1[dec1_key])
    ra2 = np.array(table2[ra2_key])
    dec2 = np.array(table2[dec2_key])

    c1 = SkyCoord(ra=ra1, dec=dec1, unit=units)
    c2 = SkyCoord(ra=ra2, dec=dec2, unit=units)

    # find the closest match
    idx, d2d, _ = c1.match_to_catalog_sky(c2, nthneighbor=1)

    sep_constraint = d2d < max_sep
    t1_matches = table1[sep_constraint]
    t2_matches = table2[idx[sep_constraint]]

    comb_tab = hstack([t1_matches, t2_matches])

    # add ang_sep
    comb_tab["ang_sep"] = d2d[sep_constraint].arcsec
    return comb_tab

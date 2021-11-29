import numpy as np
import os
import glob
import pandas as pd

from pyslime.pipeline import pipelineUtils as ppu
from importlib.resources import path

# All the data live here (except the simulation binaries):
datadir = "/Volumes/GoogleDrive/My Drive/SlimeMold/2021-11-23-VACv2/"
dropboxdir = "/Users/mwilde/Dropbox/slime-mold/data/final_data/"
packagedir = "/Users/mwilde/python/pyslime/pyslime/pipeline/data/"
otherdatadir = "/Users/mwilde/python/pyslime/pyslime/data/"

### STEP 1: Compute the mapping from the BP simulation to the SlimeMold
# DM denstiy from the simulations
bpDensityFile_z0p0 = dropboxdir + "BP_0214_densities_1024_0.bin"
bpDensityFile_z0p5 = dropboxdir + "BP_0170_densities_1024_0.bin"

# slime fit to the BP simulations
bpslimedir = datadir + "BP_z=0.0"
bpslime_datafile = "trace.bin"

out_pickle_file_z0p0 = packagedir + "mapping_BP_z0p0_1sigma.pick"

out_pickle_file_z0p5 = packagedir + "mapping_BP_z0p5_1sigma.pick"
if not os.path.exists(out_pickle_file_z0p0):
    ppu.calc_map_bp_slime(
        bpDensityFile_z0p0, bpslimedir, bpslime_datafile, out_pickle_file_z0p0
    )
if not os.path.exists(out_pickle_file_z0p5):
    ppu.calc_map_bp_slime(
        bpDensityFile_z0p5, bpslimedir, bpslime_datafile, out_pickle_file_z0p5
    )

# STEP 2: Interpolate the mapping to create a function
mapfunc_pickle_file_z0p0 = packagedir + "mapfunc_z0p0.pickle"

mapfunc_pickle_file_z0p5 = packagedir + "mapfunc_z0p5.pickle"

if not os.path.exists(mapfunc_pickle_file_z0p0):
    ppu.interpolate(out_pickle_file_z0p0, mapfunc_pickle_file_z0p0)

if not os.path.exists(mapfunc_pickle_file_z0p5):
    ppu.interpolate(out_pickle_file_z0p5, mapfunc_pickle_file_z0p5)

# STEP 3: make the distribtion files since the
# datafiles are huge. Will not run the slow parts if
# the files already exist. see get_distributions.ipynb
ppu.make_distribution_files(datadir, packagedir)


# STEP 4: standardize with linear transforms
# see the standardize_generic.ipynb notebook
# need to manually be putinto low and hiz groups

if not os.path.exists(otherdatadir + "transform_table.csv"):
    distfiles = glob.glob(packagedir + "*dist.npz")
    distfiles.sort()

    lowz = [
        "/Users/mwilde/python/pyslime/pyslime/pipeline/data/BP_z=0.0_dist.npz",
        "/Users/mwilde/python/pyslime/pyslime/pipeline/data/SDSS_z=44-476mpc_dist.npz",
        "/Users/mwilde/python/pyslime/pyslime/pipeline/data/LRG_NGC_z=0-1000mpc_dist.npz",
        "/Users/mwilde/python/pyslime/pyslime/pipeline/data/LRG_NGC_z=900-1600mpc_dist.npz",
        "/Users/mwilde/python/pyslime/pyslime/pipeline/data/LRG_SGC_z=0-1000mpc_dist.npz",
        "/Users/mwilde/python/pyslime/pyslime/pipeline/data/LRG_SGC_z=900-1600mpc_dist.npz",
    ]

    hiz = [f for f in distfiles if f not in lowz]

    dflowz = ppu.calc_stretch_shift_df(lowz, plot=False)
    dfhiz = ppu.calc_stretch_shift_df(hiz, plot=False)
    dfall = pd.concat([dflowz, dfhiz])
    dfall.to_csv(otherdatadir + "transform_table.csv")

# STEP 5: load each set of catalogs in and apply mapfunc
# this happens in 'get_slime_dense_catalog_from-*.ipynb'
# from the dropbox/finalcode folder
#
# Now when we load in a dataset, we can simply
# look up the linear transformation for that dataset

# STEP 5: make_vac_catalog.ipynb
# also MAKE SURE TO FIX DUPLICATES its in this notebook

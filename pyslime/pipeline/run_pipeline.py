import numpy as np
import os

from pyslime.pipeline import pipelineUtils as ppu

# All the data live here (except the simulation binaries):
datadir = "/Volumes/GoogleDrive/My Drive/SlimeMold/2021-11-23-VACv2/"
dropboxdir = "/Users/mwilde/Dropbox/slime-mold/data/final_data/"
packagedir = "/Users/mwilde/python/"

### STEP 1: Compute the mapping from the BP simulation to the SlimeMold
# DM denstiy from the simulations
bpDensityFile_z0p0 = dropboxdir + "BP_0214_densities_1024_0.bin"
bpDensityFile_z0p5 = dropboxdir + "BP_0170_densities_1024_0.bin"

# slime fit to the BP simulations
bpslimedir = datadir + "BP_z=0.0"
bpslime_datafile = "trace.bin"

out_pickle_file_z0p0 = (
    packagedir + "pyslime/pyslime/pipeline/data/mapping_BP_z0p0_1sigma.pick"
)

out_pickle_file_z0p5 = (
    packagedir + "pyslime/pyslime/pipeline/data/mapping_BP_z0p5_1sigma.pick"
)
if not os.path.exists(out_pickle_file_z0p0):
    ppu.calc_map_bp_slime(
        bpDensityFile_z0p0, bpslimedir, bpslime_datafile, out_pickle_file_z0p0
    )
if not os.path.exists(out_pickle_file_z0p5):
    ppu.calc_map_bp_slime(
        bpDensityFile_z0p5, bpslimedir, bpslime_datafile, out_pickle_file_z0p5
    )

# STEP 2: Interpolate the mapping to create a function
mapfunc_pickle_file_z0p0 = (
    packagedir + "pyslime/pyslime/pipeline/data/mapfunc_z0p0.pickle"
)

mapfunc_pickle_file_z0p5 = (
    packagedir + "pyslime/pyslime/pipeline/data/mapfunc_z0p5.pickle"
)

if not os.path.exists(mapfunc_pickle_file_z0p0):
    ppu.interpolate(out_pickle_file_z0p0, mapfunc_pickle_file_z0p0)

if not os.path.exists(mapfunc_pickle_file_z0p5):
    ppu.interpolate(out_pickle_file_z0p5, mapfunc_pickle_file_z0p5)

# STEP 3: linear transform each survey data set

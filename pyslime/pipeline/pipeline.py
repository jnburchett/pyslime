import numpy as np
import os

from pyslime.pipeline import pipelineUtils as ppu

# STEP 1: Compute the mapping from the BP simulation to the SlimeMold
# TODO: do this for both redshifts
bpDensityFile = (
    "/Users/mwilde/Dropbox/slime-mold/data/final_data/BP_0214_densities_1024_0.bin"
)
bpslimedir = "/Volumes/GoogleDrive/My Drive/SlimeMold/2021-10-12/BP_z=0_float32"
bpdatafile = "trace_BP_z=0_float32.bin"
out_pickle_file = (
    "/Users/mwilde/python/pyslime/pyslime/pipeline/data/mapping_BP_z0p0_1sigma.pick"
)
if not os.path.exists(out_pickle_file):
    ppu.calc_map_bp_slime(bpDensityFile, bpslimedir, bpdatafile, out_pickle_file)

# STEP 2: Interpolate the mapping to create a function


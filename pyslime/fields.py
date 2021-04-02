import numpy as np
from astropy.cosmology import Planck15
from pyslime import utils as pu

def spherical_velocities(slvobj):
    xvel = slvobj.data[:,:,:,1]
    yvel = slvobj.data[:, :, :, 2]
    zvel = slvobj.data[:, :, :, 3]

    if slvobj is None:
        griddims = np.array(griddims)
    else:
        griddims = slvobj.griddims

    xidxs = np.arange(griddims[0])
    yidxs = np.arange(griddims[1])
    zidxs = np.arange(griddims[2])

    xx,yy,zz = np.meshgrid(xidxs,yidxs,zidxs,indexing='ij')

    ### Get sky coordinates at each voxel
    print('Converting cube indices to sky coordinates...this will take a while')
    xc,yc,zc = pu.idx_to_cartesian(xx,yy,zz,slime=slvobj)
    ra,dec = pu.cartesian_to_sky(xc,yc,zc)

    ### Make vectors and take dot products
    print('Calculating components of radial vectors...')
    ra = np.radians(ra)
    dec = np.radians(dec)
    radvec_x = np.cos(ra) * np.sin(90. - dec)
    radvec_y = np.sin(ra) * np.sin(90-dec)
    radvec_z = np.cos(90. - dec)

    ### Get orientation vectors from data
    xvecs = np.float32(slvdat.data[xx, yy, zz, 1])
    yvecs = np.float32(slvdat.data[xx, yy, zz, 2])
    zvecs = np.float32(slvdat.data[xx, yy, zz, 3])

    dps = radvec_x * xvecs + radvec_y * yvecs + radvec_z * zvecs


    return rdot, phidot,thetadot

def normalize_velocity_vectors(slvobj):
    xvel = slvobj.data[:,:,:,1]
    yvel = slvobj.data[:, :, :, 2]
    zvel = slvobj.data[:, :, :, 3]

    mag = xvel**2+yvel**2+zvel**2

    ### Make vectors and take dot products
    print('Calculating components of radial vectors...')
    ra = np.radians(ra)
    dec = np.radians(dec)
    radvec_x = np.cos(ra) * np.sin(90. - dec)
    radvec_y = np.sin(ra) * np.sin(90-dec)
    radvec_z = np.cos(90. - dec)

    ### Get orientation vectors from data
    xvecs = np.float32(slvdat.data[xx, yy, zz, 1])
    yvecs = np.float32(slvdat.data[xx, yy, zz, 2])
    zvecs = np.float32(slvdat.data[xx, yy, zz, 3])

    dps = radvec_x * xvecs + radvec_y * yvecs + radvec_z * zvecs


    return rdot, phidot,thetadot

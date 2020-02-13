import numpy as np
from mayavi import mlab

def downsample_cube(slimeobj,zoomfac=0.5,islog2=True,order=1):
    from scipy.ndimage import zoom
    newarr = slimeobj.data.astype(np.float32)
    neginf = np.isneginf(newarr)
    if islog2:
        newarr[neginf] = np.min(newarr[~neginf])
    else:
        newarr[np.isneginf(newarr)] = 0.
    datares = zoom(newarr, (zoomfac, zoomfac, zoomfac),order=order)
    return datares


def vis_cube(datacube, size=(350,350),vmin=None,vmax=None):
    fig = mlab.figure(1, bgcolor=(0, 0, 0), size=size)
    source = mlab.pipeline.scalar_field(datacube)
    if vmin is None:
        vmin = datacube.min()
    if vmax is None:
        vmax = datacube.max()
    vol = mlab.pipeline.volume(source, vmin=vmin, vmax=vmax)
    return fig




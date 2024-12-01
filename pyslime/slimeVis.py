import numpy as np
from astropy.coordinates import SkyCoord

from bokeh.plotting import figure, output_file, show, output_notebook, gridplot
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter, RangeSlider, CustomJSFilter, CustomJS
from bokeh.resources import INLINE
from bokeh.models.tools import HoverTool
from bokeh.models import Arrow
from bokeh.layouts import column, row, Spacer


def downsample_cube(slimeobj, zoomfac=0.5, islog2=True, order=1):
    from scipy.ndimage import zoom
    newarr = slimeobj.data.astype(np.float32)
    neginf = np.isneginf(newarr)
    if islog2:
        newarr[neginf] = np.min(newarr[~neginf])
    else:
        newarr[np.isneginf(newarr)] = 0.
    datares = zoom(newarr, (zoomfac, zoomfac, zoomfac), order=order)
    return datares


def vis_cube(datacube, size=(350, 350), vmin=None, vmax=None):
    from pyslime import utils as pu
    from mayavi import mlab

    datacube = datacube.astype(np.float32)
    fig = mlab.figure(1, bgcolor=(0, 0, 0), size=size)
    source = mlab.pipeline.scalar_field(datacube)
    if vmin is None:
        vmin = datacube.min()
    if vmax is None:
        vmax = datacube.max()
    vol = mlab.pipeline.volume(source, vmin=vmin, vmax=vmax)
    return fig


def add_points(coords, slimeobj):
    from pyslime import utils
    from mayavi import mlab

    cartcoords = utils.transform_to_cartesian(coords[0], coords[1], coords[2])
    cenidxs = slimeobj.cartesian_to_idx(
        cartcoords[0], cartcoords[1], cartcoords[2])
    mlab.points3d(cenidxs[0], cenidxs[1], cenidxs[2], scale_factor=0.3)


def bokeh_this_that(tab, colx, coly, cds=None, tooltip=None, title=None,
                    xlabel=None, ylabel=None, xrange=None, yrange=None,
                    markersize=5, view=None, alpha=0.5, tools=None, **kwargs):
    if cds is None:
        newpd = tab.to_pandas()
        cds = ColumnDataSource(newpd.to_dict(orient='list'))

    # Interactive features of plot
    if tools is None:
        TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
    else:
        TOOLS = tools

    # Instantiate
    fig = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel,
                 tooltips=tooltip, tools=TOOLS, active_drag='box_select',
                 x_range=xrange, y_range=yrange)
    if view is not None:
        fig.square(colx, coly, source=cds, size=markersize,
                   view=view, fill_alpha=alpha, **kwargs)
    else:
        fig.square(colx, coly, source=cds, size=markersize,
                   fill_alpha=alpha, **kwargs)

    return fig


def inspect3d(slimeobj, coords, z, boxsize=[20, 20, 20], windowsize=[900, 900],
              vmaxfac=0.3, otherObjectRaDecRed=None, otherSearchDist=None):
    from astropy.cosmology import Planck15 as cosmo
    from pyslime import utils
    from mayavi import mlab

    if isinstance(coords, SkyCoord):
        radecz = [coords.ra.deg, coords.dec.deg, z]
    else:
        radecz = [coords[0], coords[1], z]
    #import pdb; pdb.set_trace()
    sliced = slimeobj.slice3d([radecz[0], radecz[1], radecz[2]], size=boxsize)
    cencoords = SkyCoord(radecz[0], radecz[1], distance=cosmo.luminosity_distance(radecz[2]),
                         unit=('deg', 'deg', 'Mpc'))
    fig = vis_cube(sliced.data, size=windowsize,
                   vmax=np.max(sliced.data) * vmaxfac)
    cartcoords = utils.transform_to_cartesian(radecz[0], radecz[1], radecz[2])
    cenidxs = sliced.cartesian_to_idx(
        cartcoords[0], cartcoords[1], cartcoords[2])
    mlab.points3d(cenidxs[0], cenidxs[1], cenidxs[2], scale_factor=0.5)
    if otherObjectRaDecRed is not None:
        othercoords = SkyCoord(otherObjectRaDecRed[0], otherObjectRaDecRed[1],
                               distance=cosmo.luminosity_distance(
                                   otherObjectRaDecRed[2]),
                               unit=('deg', 'deg', 'Mpc'))
        seps = cencoords.separation_3d(othercoords)
        if otherSearchDist is None:
            otherSearchDist = np.max(boxsize)/2
        close = np.where(seps.value < otherSearchDist)[0]
        coords2add = [otherObjectRaDecRed[0][close], otherObjectRaDecRed[1][close],
                      otherObjectRaDecRed[2][close]]
        add_points(coords2add, sliced)
    return fig

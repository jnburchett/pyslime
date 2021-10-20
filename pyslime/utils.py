import numpy as np
from astropy.cosmology import Planck15
from pyslime import slime


def load_slime_data(
    datafile, griddims, dtype=np.float16, axes="xyz", with_velocity=False
):
    raw_data = np.fromfile(datafile, dtype=dtype)
    if with_velocity:
        voxels = raw_data.reshape((griddims[2], griddims[1], griddims[0], 4))
    else:
        voxels = raw_data.reshape((griddims[2], griddims[1], griddims[0]))

    if axes is not None:
        # The axes of the loaded data array do not match up to the meta OR 'xyz'
        xidx = axes.index("x")
        yidx = axes.index("y")
        zidx = axes.index("z")
        print(np.shape(voxels), (xidx, yidx, zidx))
        # idxarr = np.array([xidx,yidx,zidx])
        # idxarr = np.roll(idxarr,2)
        # voxels = np.transpose(voxels,[idxarr[0],idxarr[1],idxarr[2]])
        if with_velocity:
            voxels = np.transpose(voxels, [zidx, yidx, xidx, 3])
            # voxels = np.transpose(voxels, [2, 1, 0, 3])
        else:
            voxels = np.transpose(voxels, [zidx, yidx, xidx])
            # voxels = np.transpose(voxels, [2, 1, 0])
    return voxels


def parse_meta_file(metafile, axes="xyz"):
    metadict = {}
    ff = open(metafile, "r")
    filedat = ff.readlines()
    fitpars = {}
    for ll in filedat:
        if "number of agents" in ll:
            numag = parse_meta_line_val(ll, return_units=False, return_str=True)
            fitpars["num_agents"] = numag
        elif "grid resolution" in ll:
            dims = parse_meta_line_val(ll, return_units=False)
            metadict["grid_res"] = reorder_axes(dims, axes)
        elif "grid size" in ll:
            gs, unit = parse_meta_line_val(ll)
            metadict["physical_dims"] = reorder_axes(gs, axes)
            unit = check_unit(unit)
            metadict["physical_unit"] = unit
        elif "grid center" in ll:
            gc, unit = parse_meta_line_val(ll)
            unit = check_unit(unit)
            if unit != metadict["physical_unit"]:
                raise ValueError("Units of grid center do not match those of size")
            metadict["grid_center"] = reorder_axes(gc, axes)
        elif ("move distance" in ll) & ("grid" not in ll):
            md, unit = parse_meta_line_val(ll)
            fitpars["move_dist"] = md
            unit = check_unit(unit)
            fitpars["dist_unit"] = unit
        elif ("sense distance" in ll) & ("grid" not in ll):
            sd, unit = parse_meta_line_val(ll)
            fitpars["sense_dist"] = sd
            unit = check_unit(unit)
            if unit != fitpars["dist_unit"]:
                raise ValueError(
                    "Units of sense distance do not match those of move distance"
                )
        elif "move spread" in ll:
            ms, unit = parse_meta_line_val(ll)
            fitpars["move_spread"] = ms
            fitpars["angle_unit"] = unit
        elif "sense spread" in ll:
            ss, unit = parse_meta_line_val(ll)
            fitpars["sense_spread"] = ss
        elif "persistence coefficient" in ll:
            pc = parse_meta_line_val(ll, return_units=False)
            fitpars["persist_coeff"] = pc
        elif "agent deposit" in ll:
            ad = parse_meta_line_val(ll, return_units=False)
            fitpars["agent_deposit"] = ad
        elif "sampling sharpness" in ll:
            ss = parse_meta_line_val(ll, return_units=False)
            fitpars["sampling sharpness"] = ss

    metadict["fitpars"] = fitpars
    return metadict


def reorder_axes(vals, axes="xyz"):
    """Reorder values to put in 'xyz' order

    Parameters
    ----------
    vals : list
    axes : str, optional
       String with order of axes.  By default will return same order

    Returns
    -------
    newvals : list
       Reordered values
    """

    xidx = axes.index("x")
    yidx = axes.index("y")
    zidx = axes.index("z")
    idxlist = [xidx, yidx, zidx]
    newvals = [vals[i] for i in idxlist]
    return newvals


def parse_meta_line_val(metaline, return_units=True, return_str=False):
    """Extract values from line in metadata file that includes a value

    Parameters
    ----------
    metaline : str
       Single line from metadata file
    return_units : bool
       If True, return units of value
    return_str : bool
       If True, return raw value as string

    Returns
    -------
    val : int or float or str
        Value to be extracted
    units : str
        Units of extracted value
    """

    rhs = metaline.split(":")[-1][:-1]  # Also gets rid of the '/n'
    rhs = rhs.strip()
    if "[" in rhs:
        units = rhs[rhs.rfind("[") + 1 : rhs.rfind("]")]
        valstr = rhs[: rhs.rfind("[")]
    else:
        units = None
        valstr = rhs

    if " x " in rhs:
        val = valstr.split((" x "))
        if return_str is False:
            for i, dd in enumerate(val):
                if "." in dd:
                    val[i] = float(dd)
                else:
                    val[i] = int(dd)
    elif "(" in rhs:
        valstr = valstr[valstr.rfind("(") + 1 : valstr.rfind(")")]
        val = valstr.split(",")
        for i, dd in enumerate(val):
            if "." in dd:
                val[i] = float(dd)
            else:
                val[i] = int(dd)
    else:
        val = valstr
        if return_str is False:
            if "." in val:
                val = float(val)
            else:
                val = int(val)
    if return_units is True:
        return val, units
    else:
        return val


def transform_to_cartesian(ra, dec, redshift, use_lumdist=True, cosmo=None):
    """Transform sky coordinates to Cartesian coordinates using the luminosity
    distance as z-coordinate

    Parameters
    ----------
    ra : float
    dec : float
    redshift : float
    use_lumdist : bool
    cosmo : astropy.cosmology

    Returns
    -------
    x,y,z : float
       Cartesian coordinates

    """
    from astropy import units as u

    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo

    azimuth = ra / 180.0 * np.pi
    polar = (90.0 - dec) / 180.0 * np.pi
    if use_lumdist:
        radial = cosmo.luminosity_distance(redshift).to(u.Mpc).value
    else:
        radial = redshift
    x = radial * np.sin(polar) * np.cos(azimuth)
    z = radial * np.cos(polar)
    y = radial * np.sin(polar) * np.sin(azimuth)
    return x, y, z


def idx_to_cartesian(
    i,
    j,
    k,
    slime=None,
    griddims=[360, 360, 360],
    minvals=[-0.035, -0.035, -0.035],
    maxvals=[0.035, 0.035, 0.035],
):

    if slime is None:
        griddims = np.array(griddims)
        minvals = np.array(minvals)
        maxvals = np.array(maxvals)
    else:
        griddims = slime.griddims
        minvals = slime.mincoords
        maxvals = slime.maxcoords

    # find approximate interval and increment final element to include maxval
    xincrement = maxvals[0] - minvals[0] / griddims[0]
    yincrement = maxvals[1] - minvals[1] / griddims[1]
    zincrement = maxvals[2] - minvals[2] / griddims[2]
    xvals = np.linspace(minvals[0], maxvals[0], griddims[0])
    yvals = np.linspace(minvals[1], maxvals[1], griddims[1])
    zvals = np.linspace(minvals[2], maxvals[2], griddims[2])

    x = xvals[i]
    y = yvals[j]
    z = zvals[k]
    return x, y, z


def check_unit(unit):
    if unit == "mpc":
        unit = "Mpc"
    return unit


def cartesian_to_sky(x, y, z, cosmo=Planck15, return_redshift=True):
    from astropy.cosmology import z_at_value
    from astropy import units as u

    ra = np.arctan2(y, x) * 180.0 / np.pi
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dec = 90.0 - np.arccos(z / dist) / np.pi * 180.0
    if return_redshift:
        try:
            redshift = z_at_value(cosmo.luminosity_distance, dist * u.Mpc)
        except:
            redshift = [
                z_at_value(cosmo.luminosity_distance, dd) for dd in dist * u.Mpc
            ]
        return ra, dec, redshift
    else:
        return ra, dec


def sample_cube(cube, size=100000, velocities=False):
    randx = np.random.randint(0, np.shape(cube)[0], size=size)
    randy = np.random.randint(0, np.shape(cube)[1], size=size)
    randz = np.random.randint(0, np.shape(cube)[2], size=size)
    if velocities:
        randvals = [cube[randx, randy, randz, i] for i in range(4)]
    else:
        randvals = cube[randx, randy, randz]
    return randvals


def pack_data_binary(
    datafile, racol="ra", deccol="dec", distcol="lumdist", masscol="logMass"
):
    tab = np.genfromtxt(datafile, names=True, dtype=None, encoding=None)
    try:
        azimuth = tab[racol] / 180.0 * np.pi
        polar = (90.0 - tab[deccol]) / 180.0 * np.pi
        radius = tab[distcol]
        x = radius * np.sin(polar) * np.cos(azimuth)
        y = radius * np.sin(polar) * np.sin(azimuth)
        z = radius * np.cos(polar)
        mass = tab[masscol]
    except:
        import pdb

        pdb.set_trace()
    # data = np.zeros((len(tab) - 1, 4), dtype=np.float32)

    data = np.array([x, y, z, mass], dtype=np.float32).T
    print("Min/Max X: {} {}".format(np.min(data[:, 0]), np.max(data[:, 0])))
    print("Min/Max Y: {} {}".format(np.min(data[:, 1]), np.max(data[:, 1])))
    print("Min/Max Z: {} {}".format(np.min(data[:, 2]), np.max(data[:, 2])))
    print("Min/Max Mass: {} {}".format(np.min(data[:, 3]), np.max(data[:, 3])))
    print("Sample record: {}".format(data[0, :]))
    print("Number of records: {}".format(len(tab)))
    data.tofile(datafile.split(".")[0] + ".bin")


def get_slime(
    smdir,
    datafile="trace.bin",
    axes="xyz",
    dtype=np.float16,
    standardize=True,
    stretch=None,
    shift=None,
) -> slime:
    """ This function prepares a raw slime fit for production of the catalog.
        Primarily, we want to do log before we standardize.

        Args:
            smdir (str): path to slime directory
            datafile (str, optional): name of the tace binary. Defaults to 'trace.bin'.
            axes (str, optional): order of the axes. Defaults to 'xyz'.
            dtype (np.dtype, optional): np.float16 or 32. Defaults to np.float16.
            standardize (bool, optional): whether to standardize the slime objects data.

        Returns:
            slimeObj: the prepared slime object
        """

    bpslime = slime.Slime.from_dir(smdir, datafile=datafile, axes=axes, dtype=dtype)
    bpslime.data = bpslime.data.astype(np.float32)
    bpslime.data = np.log10(bpslime.data)
    if standardize:
        if stretch is None or shift is None:
            print("WARNING: Must provide a stretch and shift to standardize")
        else:
            bpslime.standardize(stretch=stretch, shift=shift)
    return bpslime

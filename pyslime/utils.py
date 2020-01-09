import numpy as np

def load_slime_data(datafile,griddims,dtype=np.float16, axes=None):
    raw_data = np.fromfile(datafile, dtype=dtype)
    voxels = raw_data.reshape((griddims[2],griddims[1],griddims[0]))


    if axes is not None:
        ### The axes of the loaded data array do not match up to the meta OR 'xyz'
        xidx = axes.index('x')
        yidx = axes.index('y')
        zidx = axes.index('z')
        print(np.shape(voxels))
        #idxarr = np.array([xidx,yidx,zidx])
        #idxarr = np.roll(idxarr,2)
        #voxels = np.transpose(voxels,[idxarr[0],idxarr[1],idxarr[2]])
        voxels = np.transpose(voxels,[2,1,0])
    return voxels

def parse_meta_file(metafile,axes='xyz'):
    metadict = {}
    ff = open(metafile,'r')
    filedat = ff.readlines()
    fitpars = {}
    for ll in filedat:
        if 'number of agents' in ll:
            numag = parse_meta_line_val(ll,return_units=False,return_str=True)
            fitpars['num_agents'] = numag
        elif 'grid resolution' in ll:
            dims = parse_meta_line_val(ll,return_units=False)
            metadict['grid_res'] = reorder_axes(dims,axes)
        elif 'grid size' in ll:
            gs,unit = parse_meta_line_val(ll)
            metadict['physical_dims'] = reorder_axes(gs,axes)
            unit=check_unit(unit)
            metadict['physical_unit'] = unit
        elif 'grid center' in ll:
            gc, unit = parse_meta_line_val(ll)
            unit = check_unit(unit)
            if unit != metadict['physical_unit']:
               raise ValueError('Units of grid center do not match those of size')
            metadict['grid_center'] = reorder_axes(gc,axes)
        elif 'move distance' in ll:
            md, unit = parse_meta_line_val(ll)
            fitpars['move_dist'] = md
            unit = check_unit(unit)
            fitpars['dist_unit'] = unit
        elif 'sense distance' in ll:
            sd, unit = parse_meta_line_val(ll)
            fitpars['sense_dist'] = sd
            unit = check_unit(unit)
            if unit != fitpars['dist_unit']:
               raise ValueError('Units of sense distance do not match those of move distance')
        elif 'move spread' in ll:
            ms, unit = parse_meta_line_val(ll)
            fitpars['move_spread'] = ms
            fitpars['angle_unit'] = unit
        elif 'sense spread' in ll:
            ss, unit = parse_meta_line_val(ll)
            fitpars['sense_spread'] = ss
        elif 'persistence coefficient' in ll:
            pc = parse_meta_line_val(ll,return_units=False)
            fitpars['persist_coeff'] = pc
        elif 'agent deposit' in ll:
            ad = parse_meta_line_val(ll,return_units=False)
            fitpars['agent_deposit'] = ad
        elif 'sampling sharpness' in ll:
            ss = parse_meta_line_val(ll, return_units=False)
            fitpars['sampling sharpness'] = ss

    metadict['fitpars'] = fitpars
    return metadict

def reorder_axes(vals,axes='xyz'):
    '''Reorder values to put in 'xyz' order

    Parameters
    ----------
    vals : list
    axes : str, optional
       String with order of axes.  By default will return same order

    Returns
    -------
    newvals : list
       Reordered values
    '''

    xidx = axes.index('x')
    yidx = axes.index('y')
    zidx = axes.index('z')
    idxlist = [xidx, yidx, zidx]
    newvals = [vals[i] for i in idxlist]
    return newvals


def parse_meta_line_val(metaline,return_units = True, return_str = False):
    '''Extract values from line in metadata file that includes a value

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
    '''

    rhs = metaline.split(':')[-1][:-1] # Also gets rid of the '/n'
    rhs = rhs.strip()
    if '[' in rhs:
        units = rhs[rhs.rfind('[') + 1: rhs.rfind(']')]
        valstr = rhs[:rhs.rfind('[')]
    else:
        units = None
        valstr = rhs

    if ' x ' in rhs:
        val = valstr.split((' x '))
        if return_str is False:
            for i,dd in enumerate(val):
                if '.' in dd:
                    val[i] = float(dd)
                else:
                    val[i] = int(dd)
    elif '(' in rhs:
        valstr = valstr[valstr.rfind('(')+1:valstr.rfind(')')]
        val = valstr.split(',')
        for i, dd in enumerate(val):
            if '.' in dd:
                val[i] = float(dd)
            else:
                val[i] = int(dd)
    else:
        val = valstr
        if return_str is False:
            if '.' in val:
                val = float(val)
            else:
                val = int(val)
    if return_units is True:
        return val,units
    else:
        return val

def transform_to_cartesian(ra,dec,redshift,use_lumdist=True,cosmo=None):
    '''Transform sky coordinates to Cartesian coordinates using the luminosity
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

    '''
    from astropy import units as u
    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo

    azimuth = ra / 180.0 * np.pi
    polar = (90.0-dec) / 180.0 * np.pi
    if use_lumdist:
        radial = cosmo.luminosity_distance(redshift).to(u.Mpc).value
    else:
        radial = redshift
    x = radial * np.sin(polar) * np.cos(azimuth)
    z = radial * np.cos(polar)
    y = radial * np.sin(polar) * np.sin(azimuth)
    return x,y,z

def idx_to_cartesian(i,j,k,brick_size=360,minval = -0.035, maxval = 0.035):

    # find approximate interval and increment final element to include maxval
    interval = (maxval-minval)/brick_size
    increment = interval/brick_size
    vals = np.linspace(minval, maxval+increment, brick_size)
    z = vals[i]
    y = vals[j]
    x = vals[k]
    return x,y,z

def check_unit(unit):
    if unit == 'mpc':
        unit = 'Mpc'
    return unit




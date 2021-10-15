import numpy as np


class Slime(object):
    def __init__(self, type='trace', data=None, griddims=None, fitpars=None,
                 name=None, physicaldims=None, physicalunit='Mpc', gridcenter=None):
        self.data = data
        self.type = type
        self.griddims = griddims
        self.fitpars = fitpars
        self.name = name
        self.physicaldims = np.array(physicaldims)
        self.physicalunit = np.array(physicalunit)
        self.gridcenter = np.array(gridcenter)

        # Calculate min coords to speed up operations later on
        self.calc_extreme_coords()

    def calc_extreme_coords(self, reverse_xz=True):
        try:
            d2cellcenter = self.physicaldims/self.griddims/2.
            halfdist = self.physicaldims/2.
            mincoords = self.gridcenter - halfdist + d2cellcenter
            self.mincoords = mincoords
            maxcoords = self.gridcenter + halfdist - d2cellcenter
            self.maxcoords = maxcoords

            '''
            mincoo = mincoords.copy()
            maxcoo = maxcoords.copy()
            if reverse_xz:
                self.maxcoords[2] = mincoo[2]
                self.mincoords[2] = maxcoo[2]
                #self.maxcoords[1] = mincoo[1]
                #self.mincoords[1] = maxcoo[1]
                self.maxcoords[0] = mincoo[0]
                self.mincoords[0] = maxcoo[0]
                pass'''

        except:
            pass

    @classmethod
    def from_dir(cls, dirname, datafile='trace.bin', metafile='export_metadata.txt',
                 type='trace', axes='xyz', with_velocity=False, dtype=np.float16):
        import pyslime.utils as pu
        if dirname[-1] == '/':
            dirname = dirname[:-1]
        # Grab metadata 'as-is' to get dimensions of datagrid
        rawmetadata = pu.parse_meta_file('{}{}{}'.format(dirname, '/', metafile),
                                         axes='xyz')
        sdata = pu.load_slime_data('{}{}{}'.format(dirname, '/', datafile),
                                   griddims=rawmetadata['grid_res'], axes=axes, dtype=dtype,
                                   with_velocity=with_velocity)
        print(np.shape(sdata))
        # Now grab metadata with proper ordering of axes
        metadata = pu.parse_meta_file('{}{}{}'.format(dirname, '/', metafile),
                                      axes=axes)
        slimeobj = cls(type=type, data=sdata, griddims=metadata['grid_res'],
                       fitpars=metadata['fitpars'], name=dirname,
                       physicaldims=metadata['physical_dims'],
                       physicalunit=metadata['physical_unit'],
                       gridcenter=metadata['grid_center'])
        return slimeobj

    def cartesian_to_idx(self, x, y, z):
        '''Transform coordinates from physical space to corresponding indices
        in model data grid

        Parameters
        ----------
        x,y,z : float or array
            Cartesian coordinates in 'physical' space
        slimeobj : Slime
            Object whose data grid will be indexed

        Returns
        -------
        i,j,k : floats or arrays
            Indices into slimeobj.data
        '''
        scales = self.physicaldims/self.griddims

        if self.mincoords[0] < self.maxcoords[0]:
            i = (x - self.mincoords[0]) / scales[0]
        else:
            i = (self.mincoords[0]-x) / scales[0]
        if self.mincoords[1] < self.maxcoords[1]:
            j = (y - self.mincoords[1]) / scales[1]
        else:
            j = (self.mincoords[1]-y) / scales[1]
        if self.mincoords[2] < self.maxcoords[2]:
            k = (z - self.mincoords[2]) / scales[2]
        else:
            k = (self.mincoords[2]-z) / scales[2]
        i = np.rint(i).astype(int)
        j = np.rint(j).astype(int)
        k = np.rint(k).astype(int)

        return i, j, k

    def sky_to_data(self, ra, dec, redshift, check_bounds=False, veldim=None):
        import pyslime.utils as pu
        cart = pu.transform_to_cartesian(ra, dec, redshift)
        idxs = self.cartesian_to_idx(cart[0], cart[1], cart[2])
        is4d = len(np.shape(self.data)) == 4
        if check_bounds:
            inbounds = ((idxs[0] > 0) & (idxs[1] > 0) & (idxs[2] > 0) &
                        (idxs[0] < self.griddims[0]) & (idxs[1] < self.griddims[1]) &
                        (idxs[2] < self.griddims[2]))
            toreturn = np.zeros_like(idxs[0])-642.
            if (veldim not in [1, 2, 3]) & is4d:
                toreturn[inbounds] = self.data[idxs[0][inbounds], idxs[1][inbounds],
                                               idxs[2][inbounds], 0]
            elif is4d:
                toreturn[inbounds] = self.data[idxs[0][inbounds], idxs[1][inbounds],
                                               idxs[2][inbounds], veldim]
            else:
                toreturn[inbounds] = self.data[idxs[0][inbounds], idxs[1][inbounds],
                                               idxs[2][inbounds]]
        else:
            if (veldim not in [1, 2, 3]) & is4d:
                toreturn = self.data[idxs[0], idxs[1], idxs[2], 0]
            elif is4d:
                toreturn = self.data[idxs[0], idxs[1], idxs[2], veldim]
            else:
                toreturn = self.data[idxs[0], idxs[1], idxs[2]]

        return toreturn

    def standardize(self, mean=None, stddev=None, denscut=None):
        """ Standardize the distribution by subtracting by the mean 
        and divitding by the standard deviation of the distribuiton of
        density. This assumes the density is already in log10 space. Can 
        specify a density cut to ensure no bias due to empty space in 
        the observed catalogs. 

        Args:
            mean (float, optional): specify a mean. Defaults to None.
            stddev (float, optional): specify a std. Defaults to None.
            denscut (float, optional): an upper limit on the density to
                calculate the mean and std. Defaults to -9999.
        """

        randvals = self.random_sample(1000000)
        if denscut is None:
            if mean is None:
                mean = np.mean(randvals[~np.isneginf(randvals)])
            if stddev is None:
                stddev = np.std(randvals[~np.isneginf(randvals)],
                                dtype=np.float32)

        else:
            cut = randvals > denscut
            mean = np.mean(randvals[cut])
            stddev = np.std(randvals[cut])

        print(mean, stddev)

        self.data = (self.data - mean) / stddev

    def random_sample(self, size=10000, velocities=False):
        import pyslime.utils as pu
        randvals = pu.sample_cube(self.data, size, velocities=velocities)
        return randvals

    def slice3d(self, coords, size=[10., 10., 10.], skycoords=True):
        import pyslime.utils as pu
        if skycoords:
            cartcoords = pu.transform_to_cartesian(
                coords[0], coords[1], coords[2])
        else:
            cartcoords = coords
        startcoords = [(cartcoords[i] - size[i]/2.) for i in range(3)]
        endcoords = [(cartcoords[i] + size[i]/2.) for i in range(3)]
        startidxs = self.cartesian_to_idx(
            startcoords[0], startcoords[1], startcoords[2])
        endidxs = self.cartesian_to_idx(
            endcoords[0], endcoords[1], endcoords[2])

        sliceobj = self.copy()
        sliceobj.data = self.data[startidxs[0]:endidxs[0], startidxs[1]:endidxs[1],
                                  startidxs[2]:endidxs[2]]
        sliceobj.gridcenter = cartcoords
        sliceobj.physicaldims = np.array(size)
        sliceobj.mincoords = startcoords
        sliceobj.maxcoords = endcoords
        sliceobj.griddims = np.shape(sliceobj.data)
        return sliceobj

    def copy(self):
        newobj = Slime(data=self.data, griddims=self.griddims, fitpars=self.fitpars,
                       name=self.name, physicaldims=self.physicaldims,
                       physicalunit=self.physicalunit, gridcenter=self.gridcenter,
                       type=self.type)
        return newobj

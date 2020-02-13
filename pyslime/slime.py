import numpy as np

class Slime(object):
    def __init__(self, type = 'trace', data=None, griddims=None, fitpars=None,
                 name=None,physicaldims=None,physicalunit='Mpc',gridcenter=None):
        self.data = data
        self.type = type
        self.griddims = griddims
        self.fitpars = fitpars
        self.name = name
        self.physicaldims = np.array(physicaldims)
        self.physicalunit = np.array(physicalunit)
        self.gridcenter = np.array(gridcenter)

        ### Calculate min coords to speed up operations later on
        self.calc_extreme_coords()


    def calc_extreme_coords(self):
        try:
            mincoords = self.gridcenter - self.physicaldims/2.
            self.mincoords = mincoords
            maxcoords = self.gridcenter + self.physicaldims / 2.
            self.maxcoords = maxcoords
        except:
            pass

    @classmethod
    def from_dir(cls,dirname,datafile='trace.bin',metafile='export_metadata.txt',
                 type='trace',axes='xyz'):
        import pyslime.utils as pu
        if dirname[-1] == '/':
            dirname = dirname[:-1]
        ### Grab metadata 'as-is' to get dimensions of datagrid
        rawmetadata = pu.parse_meta_file('{}{}{}'.format(dirname,'/',metafile),
                                      axes='xyz')
        sdata = pu.load_slime_data('{}{}{}'.format(dirname,'/', datafile),
                                    griddims=rawmetadata['grid_res'],axes=axes)
        print(np.shape(sdata))
        ### Now grab metadata with proper ordering of axes
        metadata = pu.parse_meta_file('{}{}{}'.format(dirname, '/', metafile),
                                      axes=axes)
        slimeobj =  cls(type=type,data=sdata,griddims=metadata['grid_res'],
                        fitpars=metadata['fitpars'],name=dirname,
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

        i = (x - self.mincoords[0]) / scales[0]
        j = (y - self.mincoords[1]) / scales[1]
        k = (z - self.mincoords[2]) / scales[2]
        i = np.rint(i).astype(int)
        j = np.rint(j).astype(int)
        k = np.rint(k).astype(int)
        return i, j, k

    def sky_to_data(self,ra,dec,redshift):
        import pyslime.utils as pu
        cart = pu.transform_to_cartesian(ra, dec, redshift)
        idxs = self.cartesian_to_idx(cart[0], cart[1], cart[2])

        return self.data[idxs[0], idxs[1], idxs[2]]

    def standardize(self,mean=None,stddev=None):
        randvals = self.random_sample(100000)
        if mean is None:
            mean = np.mean(randvals[~np.isneginf(randvals)])
        if stddev is None:
            stddev = np.std(randvals[~np.isneginf(randvals)],dtype=np.float32)
        self.data = (self.data - mean) / stddev

    def random_sample(self,size=10000):
        import pyslime.utils as pu
        randvals = pu.sample_cube(self.data,size)
        return randvals




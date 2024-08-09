import numpy as np


class Slime(object):
    def __init__(
        self,
        type="trace",
        data=None,
        griddims=None,
        fitpars=None,
        name=None,
        physicaldims=None,
        physicalunit="Mpc",
        gridcenter=None,
    ):
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
            d2cellcenter = self.physicaldims / self.griddims / 2.0
            halfdist = self.physicaldims / 2.0
            mincoords = self.gridcenter - halfdist + d2cellcenter
            self.mincoords = mincoords
            maxcoords = self.gridcenter + halfdist - d2cellcenter
            self.maxcoords = maxcoords

            """
            mincoo = mincoords.copy()
            maxcoo = maxcoords.copy()
            if reverse_xz:
                self.maxcoords[2] = mincoo[2]
                self.mincoords[2] = maxcoo[2]
                #self.maxcoords[1] = mincoo[1]
                #self.mincoords[1] = maxcoo[1]
                self.maxcoords[0] = mincoo[0]
                self.mincoords[0] = maxcoo[0]
                pass"""

        except:
            pass

    @classmethod
    def from_dir(
        cls,
        dirname,
        datafile="trace.bin",
        metafile="export_metadata.txt",
        type="trace",
        axes="xyz",
        with_velocity=False,
        dtype=np.float32,
    ):
        import pyslime.utils as pu

        if dirname[-1] == "/":
            dirname = dirname[:-1]
        # Grab metadata 'as-is' to get dimensions of datagrid
        rawmetadata = pu.parse_meta_file(
            "{}{}{}".format(dirname, "/", metafile), axes="xyz"
        )
        sdata = pu.load_slime_data(
            "{}{}{}".format(dirname, "/", datafile),
            griddims=rawmetadata["grid_res"],
            axes=axes,
            dtype=dtype,
            with_velocity=with_velocity,
        )
        # print(np.shape(sdata))
        # Now grab metadata with proper ordering of axes
        metadata = pu.parse_meta_file(
            "{}{}{}".format(dirname, "/", metafile), axes=axes
        )
        slimeobj = cls(
            type=type,
            data=sdata,
            griddims=metadata["grid_res"],
            fitpars=metadata["fitpars"],
            name=dirname,
            physicaldims=metadata["physical_dims"],
            physicalunit=metadata["physical_unit"],
            gridcenter=metadata["grid_center"],
        )
        return slimeobj

    @classmethod
    def get_slime(
        self,
        smdir,
        datafile="trace.bin",
        axes="xyz",
        dtype=np.float32,
        standardize=True,
        custom=False,
        stretch=None,
        shift=None,
    ):
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

        slimeobj = self.from_dir(smdir, datafile=datafile, axes=axes, dtype=dtype)
        slimeobj.data = np.log10(slimeobj.data)
        if standardize:
            if custom:
                if stretch is None or shift is None:
                    print(
                        "WARNING: Must provide a stretch and shift to standardize if custom is True"
                    )
            else:
                slimeobj.standardize(custom=custom, stretch=stretch, shift=shift)
        return slimeobj

    def cartesian_to_idx(self, x, y, z):
        """Transform coordinates from physical space to corresponding indices
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
        """
        scales = self.physicaldims / self.griddims

        if self.mincoords[0] < self.maxcoords[0]:
            i = (x - self.mincoords[0]) / scales[0]
        else:
            i = (self.mincoords[0] - x) / scales[0]
        if self.mincoords[1] < self.maxcoords[1]:
            j = (y - self.mincoords[1]) / scales[1]
        else:
            j = (self.mincoords[1] - y) / scales[1]
        if self.mincoords[2] < self.maxcoords[2]:
            k = (z - self.mincoords[2]) / scales[2]
        else:
            k = (self.mincoords[2] - z) / scales[2]
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
            inbounds = (
                (idxs[0] > 0)
                & (idxs[1] > 0)
                & (idxs[2] > 0)
                & (idxs[0] < self.griddims[0])
                & (idxs[1] < self.griddims[1])
                & (idxs[2] < self.griddims[2])
            )
            toreturn = np.zeros_like(idxs[0]) - 642.0
            if (veldim not in [1, 2, 3]) & is4d:
                toreturn[inbounds] = self.data[
                    idxs[0][inbounds], idxs[1][inbounds], idxs[2][inbounds], 0
                ]
            elif is4d:
                toreturn[inbounds] = self.data[
                    idxs[0][inbounds], idxs[1][inbounds], idxs[2][inbounds], veldim
                ]
            else:
                toreturn[inbounds] = self.data[
                    idxs[0][inbounds], idxs[1][inbounds], idxs[2][inbounds]
                ]
        else:
            if (veldim not in [1, 2, 3]) & is4d:
                toreturn = self.data[idxs[0], idxs[1], idxs[2], 0]
            elif is4d:
                toreturn = self.data[idxs[0], idxs[1], idxs[2], veldim]
            else:
                toreturn = self.data[idxs[0], idxs[1], idxs[2]]

        return toreturn

    def standardize(
        self, custom: bool = False, stretch: float = 1.0, shift: float = 0.0
    ) -> None:
        """Standardize the distribution via a linear transormation. The values 
        of stretch and shift should be chosen based on an inspection of the data.
        The best values are stored in a csv file in pyslime.data and will be
        chosen form there unless custom is set to true

        Args:
            custom (bool, optional): [description]. Defaults to False.
            stretch (float, optional): [description]. Defaults to 1.0.
            shift (float, optional): [description]. Defaults to 0.0.
        """
        if not custom:
            import pandas as pd
            import importlib.resources
            from pyslime import data

            with importlib.resources.path(data, "transform_table.csv") as p:
                df = pd.read_csv(p, index_col=0)

            nameidx = self.name.rfind("/") + 1
            name = self.name[nameidx:]
            stretch = df.loc[df.name == name]["stretch"].values[0]
            shift = df.loc[df.name == name]["shift"].values[0]

        new_distribution = self.data * stretch + shift
        self.data = new_distribution

    def random_sample(self, size=10000, velocities=False):
        import pyslime.utils as pu

        randvals = pu.sample_cube(self.data, size, velocities=velocities)
        return randvals

    def slice3d(self, coords, size=[10.0, 10.0, 10.0], skycoords=True):
        import pyslime.utils as pu

        if skycoords:
            cartcoords = pu.transform_to_cartesian(coords[0], coords[1], coords[2])
        else:
            cartcoords = coords
        startcoords = [(cartcoords[i] - size[i] / 2.0) for i in range(3)]
        endcoords = [(cartcoords[i] + size[i] / 2.0) for i in range(3)]
        startidxs = self.cartesian_to_idx(
            startcoords[0], startcoords[1], startcoords[2]
        )
        endidxs = self.cartesian_to_idx(endcoords[0], endcoords[1], endcoords[2])

        sliceobj = self.copy()
        sliceobj.data = self.data[
            startidxs[0] : endidxs[0],
            startidxs[1] : endidxs[1],
            startidxs[2] : endidxs[2],
        ]
        sliceobj.gridcenter = cartcoords
        sliceobj.physicaldims = np.array(size)
        sliceobj.mincoords = startcoords
        sliceobj.maxcoords = endcoords
        sliceobj.griddims = np.shape(sliceobj.data)
        return sliceobj

    def copy(self):
        newobj = Slime(
            data=self.data,
            griddims=self.griddims,
            fitpars=self.fitpars,
            name=self.name,
            physicaldims=self.physicaldims,
            physicalunit=self.physicalunit,
            gridcenter=self.gridcenter,
            type=self.type,
        )
        return newobj

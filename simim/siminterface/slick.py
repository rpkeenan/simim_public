import os
import requests
import warnings

import numpy as np

from simim.siminterface._rawsiminterface import SimCatalogs, Snapshot
from simim.siminterface._sims import _checksim, _slicksims

class SLICKCatalogs(SimCatalogs):
    """Class to format SLICK group catalogs - this is a stub
    to be filled in later if needed"""

    def __init__(self,
                 sim, 
                 path='auto',
                 snaps='all',
                 updatepath=True,
                 ):
        """Initialize an interface with the SLICK data
        
        Parameters
        ----------
        sim : string
            Name of the simulation you want to download. 
            'SLICK-TNG50' is currently defined
        path : optional, 'auto' or string
            The path for saving/accessing the simulation data. If 'auto', this 
            will be looked up or created in the default SimIM filepaths. Defaults
            to 'auto' and should probably only be changed if you want are making 
            additional copies of the data for some reason.
        snaps : optional, 'all' or list of ints
            The snapshots to use when downloading/formatting the simulation. Defaults
            to 'all' which will use all known snapshots.
        updatepath : optional, bool
            Defaults to True. If True, the path parameter will be saved as the 
            default path to this simulation in future uses.
        """
        
        raise ValueError("__init__ not implemented")

        # Figure out the snapshots needed
        if sim in _slicksims:
            self.allsnaps = # Fill in how many snapshots are present


        # Initialize catalog
        SimCatalogs.__init__(self, sim, path, snaps, updatepath)

        # Initialize fields
        # This is what this looked like for Illustris - names may be 
        # different in SLICK
        self.basic_fields = {
                        'SubhaloPos':[('pos_x','f','Mpc/h',-1),('pos_y','f','Mpc/h',-1),('pos_z','f','Mpc/h',-1)],
                        'SubhaloMass':[('mass','f','Msun/h',-1)],
                        }
        self.dm_fields = {
                     # Maximum value of the spherically-averaged rotation
                     # curve.
                     'SubhaloVmax':[('vmax','f','km/s',0)],
                     }
        self.matter_fields = {
                        # Mass-weighted average metallicity (Mz/Mtot, where
                        # Z = any element above He) of the gas cells bound to
                        # this Subhalo, but restricted to cells within twice
                        # the stellar half mass radius.
                        'SubhaloGasMetallicity':[('met','f','None',0)],
                        # Sum of the individual star formation rates of all gas
                        # cells in this subhalo.
                        'SubhaloSFR':[('sfr','f','Msun/yr',0)],
                        }

        # Identify keys that will require unit conversions
        def power_10(x,h):
            return x*1e10
        def kpc_to_mpc(x,h):
            return x / 1000
        self.transform_keys = {'SubhaloMass':power_10,
                               'SubhaloPos':kpc_to_mpc,
                               }

        # Check whether snapshots have been downloaded
        not_downloaded = []
        for i in self.snaps:
            file_path = os.path.join(self.path,'raw',FILENAME_OF_SNAP_i)
            if not os.path.exists(file_path):
                not_downloaded.append(i)
        if len(not_downloaded) > 0:
            warnings.warn("No data found for snapshots {} - make sure they are in {}".format(not_downloaded, os.path.join(self.path,'raw')))



    # Function to load the data in a format we want:
    def _loader(self, path, snapshot, fields):
        """Loader to get a field from a snapshot halo catalog

        This is promarily meant for internal use by the .format method
        of the SimCatalogs class
        
        Parameters
        ----------
        path : string
            Path to the raw data files
        snapshot : int
            Index-number of the snapshot
        fields : list
            List of field names to load

        Returns
        -------
        subhalos : dict
            Dictionary containing the field names (from fields) as keys and the values
            of those fields for each subhalo
        n_halos : int
            The number of halos found
        """

        raise ValueError("_loader not implemented")

        # First load the raw data = e.g.:
        data_raw = np.load(os.path.join(path,FILENAME_OF_SNAPSHOT))
        # (replace np.load with whatever data reader is necessary for the format of the raw data)

        # Then filter to requested properties:
        subhalos = {}
        for key in fields:
            subhalos[key] = data_raw[key]

        # Also need to count the numbr of halos being returned
        n_halos = len(data_raw)

        return subhalos, n_halos


    def _get_rawsnapfile(self, snapshot):
        """Get path to a snapshot's raw file"""

        raise ValueError("_get_rawsnapfile not implemented")
        
        # This should return the name of a snapshot's raw data,
        # given a snapshot number. e.g.
        return os.path.join('snapshot_{:03d}'.format(snapshot))

    def download(self, redownload=False):
        """Download catalogs - can be implemented if making 
        data available online"""

        raise ValueError("download not implemented")

        if not os.path.exists(os.path.join(self.path,'raw')):
            os.mkdir(os.path.join(self.path,'raw'))

        # Download metadata if not present
        if not os.path.exists(self.meta_path):
            self.download_meta()

        # Add a check for already downloaded files
        if redownload:
            self.download_snaps = np.copy(self.snaps)
        else:
            self.download_snaps = []
            for snap in self.snaps:
                file_path = os.path.join(self.path,'raw',self._get_rawsnapfile(snap))
                if os.path.exists(file_path):
                    warnings.warn("Skipping snapshot {} as it appears to exist already".format(snap))
                else:
                    self.download_snaps.append(snap)

        # Download each snap
        for i in range(len(self.download_snaps)):
            snap = self.download_snaps[i]
            print("downloading item {} of {} ({})".format(i+1,len(self.download_snaps),self._get_rawsnapfile(snap)))
            file_path = os.path.join(self.path,'raw',self._get_rawsnapfile(snap))
            urlretrieve(WEB_PAGE_GOES_HERE+self._get_rawsnapfile(snap),file_path)

    def download_meta(self, redownload=False):
        """Download and generate metadata for the set of snapshots 
        specified when initalizing the class
        
        Note: the metadata saved is dependent on the list of snapshots,
        therefore if you plan to use many snapshots for some applications
        but have for some reason only initialized your SLICKCatalogs
        instance with a few it is probably best to do something like the 
        following:
            >>> x = SLICKCatalogs(...,snaps='all')
            >>> x.download_meta()
            >>> x = SLICKCatalogs(...,snaps=[10,11,12])
            >>> x.download()
        """

        raise ValueError("download_meta not implemented")

        # Check that metadata doesn't already exist
        if not redownload:
            if os.path.exists(self.meta_path):
                warnings.warn("Metadata appears to exist already")
                return

        self.metadata = ...
        self.snap_meta = ...

        np.save(self.meta_path,self.metadata)
        np.save(self.snap_meta_path,self.snap_meta)

import os
import warnings

from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import astropy.units as u
import h5py
import numpy as np

from simim._paths import _SimIMPaths
from simim.siminterface._sims import _checksim


class Snapshot():
    """Class containing information for individual snapshots
    
    This class stores information and does some basic calculations
    about individual snapshots - it's initialized when loading and
    formatting simulation data. Users probably don't need it for
    anything"""

    def __init__(self,index,redshift,metadata):
        """Class containing information for individual snapshots
    
        This class stores information and does some basic calculations
        about individual snapshots - it's initialized when loading and
        formatting simulation data. Users probably don't need it for
        anything. 
        
        Initialization parameters are generally hard coded for the 
        different sims, or otherwise extracted from simulation metadata.
        
        Parameters
        ----------
        index : int
            The numerical index of the snapshot
        redshift : float
            The redshift of the snapshot
        metadata : dict
            Dictionary containing the snapshot metadata. This must contain
            (at a minimum) 'cosmo_h', 'cosmo_omega_matter', 'cosmo_omega_baryon'
            (for defining the cosmology used by the sim)
        """

        self.index = index
        self.redshift = redshift

        cosmo = FlatLambdaCDM(H0=100*metadata['cosmo_h']*u.km/u.s/u.Mpc,
                              Om0=metadata['cosmo_omega_matter'],
                              Ob0=metadata['cosmo_omega_baryon'],
                              Tcmb0=2.7255*u.K)

        self.cosmology = cosmo

    def dif_snap(self,other):
        """Determine the midpoint between two snaps
        
        Method to determine the redshift and time of the midpoint
        between this snapshot and other snapshot. The midpoint is
        defined to be the cosmoligcal age half-way between the ages
        of the two snapshots."""

        time_snap = self.cosmology.age(self.redshift).value
        time_other = self.cosmology.age(other.redshift).value
        time_middle = (time_snap+time_other)/2

        redshift_middle = z_at_value(self.cosmology.age,time_middle*u.Gyr)

        return redshift_middle, time_middle

    def dif_lowerz_snap(self,other):
        """Determine redshift, time, distance where snap ends and 
        a later one begins.
        
        If other is 0 it will treat this as the last (in time) snap
        and assume that it extends to z=0."""

        if other == 0:
            self.redshift_min = 0
            self.time_end = self.cosmology.age(0).value
            self.distance_min = 0
            self.transverse_distance_min = 0

        else:
            if self.redshift < other.redshift:
                raise ValueError("other snapshot has a higher redshift")

            redshift_middle, time_middle = self.dif_snap(other)
            self.redshift_min = redshift_middle
            self.time_end = time_middle
            self.distance_min = self.cosmology.comoving_distance(self.redshift_min).value
            self.transverse_distance_min = self.cosmology.comoving_transverse_distance(self.redshift_min).value

    def dif_higherz_snap(self,other):
        """Determine redshift, time, distance where snap ends and 
        an earlier one begins.
        
        If other is 'max' it will treat this as the first (in time) snap
        and assume that it extends to the actual redshift of the snap."""

        if other == 'max':
            self.redshift_max = self.redshift
            self.time_start = self.cosmology.age(self.redshift).value
            self.distance_max = self.cosmology.comoving_distance(self.redshift).value
            self.transverse_distance_max = self.cosmology.comoving_transverse_distance(self.redshift).value

        else:
            if self.redshift > other.redshift:
                raise ValueError("other snapshot has a lower redshift")

            redshift_middle, time_middle = self.dif_snap(other)
            self.redshift_max = redshift_middle
            self.time_start = time_middle
            self.distance_max = self.cosmology.comoving_distance(self.redshift_max).value
            self.transverse_distance_max = self.cosmology.comoving_transverse_distance(self.redshift_max).value





# Notes about HDF5 file structure:
# file
# -- Atribute: associated metadata for sim
# -- Atribute: associated cosmology for sim
# -- Atribute: keys available
# -- Group: Snap {}
#    -- Atribute: associated metadata for snapshot
#    -- Atribute: redshifts and distances where the sim starts/ends
#    -- Atribute: structured array containing location of mass cuts
#    -- dataset:
#         note: mass units - Msun/h, lenghth units Mpc/h, time units Gyr/h

class SimCatalogs():
    """Generic class for interacting with simulation halo catalogs and
    converting them into SimIM's preferred format.
    
    This class isn't used directly, but should be extended for any 
    simulation that is to be integrated into SimIM. The modules
    illustris.py, and universemachine.py show how to do this for two
    examples.

    To construct a new SimCatalogs subclass, a few steps are necessary.
    1. Define the __init__ method: this method should first create an array
    self.allsnaps, which contains the number-index of every snap in the
    orignial simulation. __init__ should then call
    SimCatalogs.__init__ with the relevant arguments (see Parameters 
    for the SimCatalogs.__init__ method). Then it should construct three 
    dictionaries that define the field names in the unformatted
    halo catalog files and how these names map to fields in the SimIM data
    structure. 

    The mapping between unformatted and formatted fields is done with 
    three dictionaries - self.basic_fields for fields that MUST be included
    in the formatted SimIM catalog (these are pos_x, pos_y, pos_z, and mass
    for subhalo positional coordinates and subhamo mass), self.dm_fields
    for fields describing additional dark matter properties, and self.matter_fields
    for fields describing baryonic properties. The latter dictionaries can 
    be left empty if no additional properties are in the unformatted catalogs
    and/or if none of these other properties will ever be propagated into the
    SimIM formatted data.

    The basic fields dictionary should be structured as follows:
    self.basic_fields = 
        {'[unformatted field name]':[('formatted field name',
                                      'formatted field dtype',
                                      'formatted field units',
                                      'formatted field dependence on hubble constant'
                                      )]}
    The use of lists for the values of each dictionary entry is
    to allow for the possibility that a field in the unformatted
    catalog consists of a tuple of values. For example, illustris
    catalogs store halo positins as a tuple 'SubhaloPos'. To convert
    this to SimIM format (pos_x, pos_y, pos_z) the following dictionary
    entry is used:
        {'SubhaloPos':[('pos_x','f','Mpc/h',-1),
                       ('pos_y','f','Mpc/h',-1),
                       ('pos_z','f','Mpc/h',-1)
                       ]}
    The 'formatted field dtype' should be whatever data format you want
    the final data to be written as, and the 'formatted field units' and 
    'formatted field dependence on hubble constant' should be the units and
    hubble constant dependence of the data in its final format, after any
    transformations have been applied (see below). Note that SimIM generally
    assumes data are saved in 'little h' units, so if they are saved units 
    with no little h, the h dependence should be set to 0.
    
    The dm_fields and matter_fields dictionaries are constructed 
    similarly.

    In addition to the dictionaries specifying keys, a dictionary 
    specifying transformations to apply when formatting the data can
    be providded. This should be called self.transform_keys and should
    have keys that match the names of unformatted keys in the 
    simulation catalog and arguments that specify a function to apply.
    For example, SimIM generally assumes distance units of Mpc/h, while
    Illustris uses kpc/h for some fields. A conversion might look like
        {'SubhaloPos':lambda x: x/1000}

    Finally, the init function should have a check to verify which 
    snapshots have already been downloaded, and which still need to be:
        # Check whether snapshots have been downloaded
        not_downloaded = []
        for i in self.snaps:
            file_path = [path where snapshot would be saved]
            if not os.path.exists(file_path):
                not_downloaded.append(i)
        if len(not_downloaded) > 0:
            warnings.warn("No data exists for snapshots {} - run .download".format(not_downloaded))

    2. The extended class also need methods to download the halo catalog 
    (self.download) and simulation metadata (self.download_meta). These
    functions should save the data in the directory listed in self.path.
    The simulation halo catalogs should be placed in a subdirctory called
    'raw', and the metadata should be placed in self.meta_path and 
    self.snap_meta_path. Look at existing code for examples of how to do
    this.

    3. The extended class needs a self._loader which takes as arguments the
    path to the data, a snapshot number, and a list of fields and returns
    a dictionary containing key-value pairs of the property name and the 
    values for every halo of the property, along with an integer specifying
    the number of halos found.
    """
 
    def __init__(self,
                 sim, 
                 path='auto',
                 snaps='all',
                 updatepath=True,
                 ):
        """Initialize the information needed to downlaod and format a simulation.
        
        Parameters
        ----------
        sim : string
            The string naming the simulation - this string must be in the
            _acceptedsims list from the _sims.py submodule
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

        # Check that we can handle the specified sim
        _checksim(sim)
        self.sim = sim

        # Set up a place to keep the data
        paths = _SimIMPaths()
        if path == 'auto':
            if self.sim in paths.sims:
                self.path = paths.sims[self.sim]
            else:
                paths._newsimpath(self.sim)
                self.path = paths.sims[self.sim]
        else:
            if not os.path.exists(path):
                raise NameError("Specified path not found")
            self.path = os.path.abspath(path)
            if updatepath:
                paths._newsimpath(sim,new_path=path,checkoverwrite=True)

        self.meta_path = os.path.join(self.path, 'meta.npy')
        self.snap_meta_path = os.path.join(self.path, 'meta_snaps.npy')

        if snaps == 'all':
            self.snaps = self.allsnaps
        else:
            self.snaps = snaps

        # Check that snapshots requested exist
        badsnaps = np.setdiff1d(self.snaps,self.allsnaps)
        if len(badsnaps) > 0:
            warnings.warn("Some requested snapshots do not exist: {}. Skipping them.".format(badsnaps))
        self.snaps = np.intersect1d(self.snaps,self.allsnaps)

        # Check what data might already exist
        if not os.path.exists(os.path.join(self.path,'raw')):
            warnings.warn("No data for this simulation has been downloaded - run .download")
        if not os.path.exists(self.meta_path):
            warnings.warn("No meta data for this simulation is available - run .download_meta")
            self.metadata = None
            self.snap_meta = None
            self.h = None
            self.box_edge = None
        else:
            self.metadata = np.load(self.meta_path,allow_pickle='TRUE').item()
            self.snap_meta = np.load(self.snap_meta_path)
            self.h = self.metadata['cosmo_h']
            self.box_edge = self.metadata['box_edge']

        # Initialize the containers for various types of info
            # keys that require a unit conversion:
        self.transform_keys = {}

        # Mapping key in original data type to new data structure
        self.basic_fields = {}
        self.dm_fields = {}
        self.matter_fields = {}

        # File name structure of raw data
        self.raw_fname = ''

        # Initialize a function to get halos - this must be modified when
        # creating the actual class, it's here as a template
        # def _loader(path, snapshot, fields):
        #     subhalos = {'anykey':[]}
        #     n_halos = len(subhalos['anykey'])
        #     return subhalos, n_halos
        # self._loader = _loader

    def clean_raw(self):
        """Remove unformatted data for a simulation
        
        This method permanently deletes the unformatted data for a
        simulation. This should only be done if the data is no longer
        needed - e.g. after the SimIM formatted file has been created
        and validated."""

        # Confirm you really want to delete the stuff you spend two days downloading
        print("This will delete raw files for {}.".format(self.sim))
        answer = input("Are you sure you wish to proceed? y/n: ")
        while answer != 'y':
            if answer == 'n':
                print("Aborting cleanup")
                return
            print("Answer not recognized.\n")

            answer = input("Are you sure you wish to proceed? y/n: ")

        # Remove stuff
        file_path = os.path.join(self.path,'raw')
        os.system("rm -r {}".format(file_path))

    def format(self,remake=False,overwrite=False,basic=False,realtime_clean_raw=False,realtime_clean_raw_check=True):
        """Convert the unformatted simulation data into the uniform format used by 
        SimIM for all halo catalogs.
        
        This method makes a data.hdf5 file containing the formatted data for the snapshots
        pointed to in the SimCatalogs instance. If a file already exists, it will
        be added to, not overwritten (i.e. snapshots already present in data.hdf5 won't be
        reformatted/replaced, but those not present will be added). By setting remake=True
        the whole file can be overwritten. By setting overwrite=True snaps present in 
        both the data.hdf5 file and the snap_catalogs instance will be reformatted and 
        rewritten. Note: if a data.hdf5 file exists but contains no snapshots, the behavior
        will be the same as if remake=True.
        
        Set realtime_clean_raw=True to delete the raw data files as they are processed
        and save space on disk - be careful as this will permanently delete files and may
        require re-downloading if there's a problem formatting.
        
        Set basic=True to save only a limited set of halo properties (position, velocity, mass)
        instead of everything in the raw catalog. This is ignored if appending to an existing
        file, and the properties in that file are used instead.
        
        Parameters
        ----------
        remake : bool, default=False
            If True, a new file will be created (overwriting any data.hdf5 file that previously
            existed) and the snaps listed in the SimCatalogs instance will be written to it.
            If False, only new data will be added.
        overwrite : bool, default=False
            If True, snaps already present in the data.hdf5 file but also listed in the SimCatalogs 
            instance will be written over with newly formatted versions. If False, the versions alreay
            in data.hdf5 will be left untouched
        basic : bool, default=False
            If True, only halo positions and masses will be formatted and saved in data.hdf5, if False,
            all fields will be saved. If a data.hdf5 file already exists this will be ignored and the
            the fields in the existing data file will be matched (unless remake=True)
        realtime_clean_raw : bool, default=False
            If set to True, the unformatted simulation data will be deleted as it is formatted and 
            written to data.hdf5. This saves disk space but will require redownloading data if 
            there are problems with the formatting.
        realtime_clean_raw_check : bool, default=True
            Confirms with the user before starting to delete unformatted snaps when realtime_clean_raw
            is set to True
        """

        # Confirm you really want to delete the stuff you spend two days downloading
        if realtime_clean_raw and realtime_clean_raw_check:
            print("This will delete raw files for {} after they are processed.".format(self.sim))
            answer = input("Are you sure you wish to proceed? y/n: ")
            while answer != 'y':
                if answer == 'n':
                    print("Aborting")
                    return
                print("Answer not recognized.\n")

                answer = input("Are you sure you wish to proceed? y/n: ")

        # Figure out what to do - 1) create new thing altogether (overwriting old one)
        #    2) add things to existing file - only add things not already present
        #    3) add things to existing file - overwrite old things
        pathexists = os.path.exists(os.path.join(self.path,'data.hdf5'))
        
        pathhassnaps = False
        if pathexists:
            with h5py.File(os.path.join(self.path,'data.hdf5'), "r") as file:
                if len(file.keys())>0:
                    pathhassnaps = True

        # Case 1 - create new file, save all snapshots in SimHandler to it
        if not pathexists or remake or not pathhassnaps:
            snaps_to_do = self.snaps
            if basic:
                self.other_fields = {}
            else:
                self.other_fields = {**self.dm_fields,**self.matter_fields}
            self.all_fields = {**self.basic_fields,**self.other_fields}

            # Create the file and give it some basic information
            with h5py.File(os.path.join(self.path,'data.hdf5'), "w", fs_persist=True) as file:
                # The simulation meta data
                for key in self.metadata.keys():
                    file.attrs[key] = self.metadata[key]

                file.attrs['snapshots'] = self.snap_meta

        # Cases 2 and 3 - need to first get the fields to write and figure out what snaps are called for and get current metadata
        # Then sort through the requested snaps and decide whether to overwrite them or not when duplicated
        else:
            with h5py.File(os.path.join(self.path,'data.hdf5'), "r") as file:
                add_snaps = np.array([i for i in self.snaps if 'Snapshot {}'.format(i) not in file.keys()], dtype='int')
                replace_snaps = np.array([i for i in self.snaps if 'Snapshot {}'.format(i) in file.keys()], dtype='int')

                data_fields = [k for k in file[list(file.keys())[0]].keys()]
                self.all_fields = {**self.basic_fields}
                self.other_fields = {}
                other_fields_start = {**self.matter_fields, **self.dm_fields}
                for f in other_fields_start:
                    if other_fields_start[f][0][0] in data_fields:
                        self.other_fields[f] = other_fields_start[f]
                        self.all_fields[f] = other_fields_start[f]
                
                meta_snaps = file.attrs['snapshots']
                meta_numsnaps = file.attrs['number_snaps']

            if overwrite:
                snaps_to_do = np.concatenate((add_snaps,replace_snaps))
            else:
                snaps_to_do = add_snaps


            meta_snaps = meta_snaps[~np.isin(meta_snaps['index'], snaps_to_do)]
            new_snaps = self.snap_meta[np.isin(self.snap_meta['index'], snaps_to_do)]
            meta_snaps = np.concatenate((meta_snaps, new_snaps))
            meta_snaps = np.sort(meta_snaps,order='index')

            with h5py.File(os.path.join(self.path,'data.hdf5'), "a") as file:
                file.attrs['snapshots'] = meta_snaps
                file.attrs['number_snaps'] = len(meta_snaps)

        if len(snaps_to_do) == 0:
            warnings.warn("No new data has been added to file")
            return

        # Now get the data
        for snap in snaps_to_do:
            print("Formatting snap {}".format(snap))

            # Load stuff in from original file formats
            subhalos, n_halos = self._loader(path=os.path.join(self.path,'raw',''), snapshot=snap, fields=self.all_fields.keys())

            # Format values
            if n_halos > 0:
                for key in self.transform_keys.keys():
                    if key in subhalos.keys():
                        subhalos[key] = self.transform_keys[key](subhalos[key],self.h)
                
                ## Old version - kept for now in cause new version causes bugs
                ## new version is generic and much more flexible
                # # Put mass in Msun/h
                # for key in self.mass_e8_keys:
                #     if key in subhalos.keys():
                #         subhalos[key] = subhalos[key] * 1e10
                # for key in self.mass_add_h_keys:
                #     if key in subhalos.keys():
                #         subhalos[key] = subhalos[key] * self.h
                # # Put position in Mpc/h
                # for key in self.pos_kpc_keys:
                #     if key in subhalos.keys():
                #         subhalos[key] = subhalos[key] / 1000
                # # Put time in Gyr/h
                # for key in self.inv_time_keys:
                #     if key in subhalos.keys():
                #         subhalos[key] = subhalos[key] * .978

            # We want to keep a few basic properties together
            dtype_basic = []
            for key in self.basic_fields.keys():
                for subkey in range(len(self.basic_fields[key])):
                    dtype_basic.append(self.basic_fields[key][subkey][:2])

            basic_properties = np.empty(n_halos,dtype=dtype_basic)
            basic_units = {}
            basic_h = {}

            for key in self.basic_fields.keys():
                n_subkeys = len(self.basic_fields[key])
                if n_subkeys > 1:
                    for subkey in range(n_subkeys):
                        new_key = self.basic_fields[key][subkey][0]
                        if n_halos > 0:
                            basic_properties[new_key] = subhalos[key][:,subkey]
                        basic_units[new_key] = self.basic_fields[key][subkey][2]
                        basic_h[new_key] = self.basic_fields[key][subkey][3]
                else:
                    new_key = self.basic_fields[key][0][0]
                    if n_halos > 0:
                        basic_properties[new_key] = subhalos[key]
                    basic_units[new_key] = self.basic_fields[key][0][2]
                    basic_h[new_key] = self.basic_fields[key][0][3]

            # We'll sort everything by mass (descending)
            sorted_inds = np.argsort(basic_properties['mass'])[::-1]
            basic_properties = basic_properties[sorted_inds]
            for key in self.other_fields.keys():
                if n_halos > 0:
                    subhalos[key] = subhalos[key][sorted_inds]
                else:
                    subhalos[key] = np.zeros((0,len(self.other_fields[key])))

            # Now we want to know the indices where various mass cuts can
            # be applied. We'll do it in steps of 0.1 dex
            mass_cuts = np.zeros(141,dtype=[('min_mass','f'),('index','i')])
            mass_cuts['min_mass'] = np.logspace(6,20,141)[::-1]

            for i in range(len(mass_cuts)):
                inds = np.where(basic_properties['mass']>=mass_cuts['min_mass'][i])[0]
                if len(inds) == 0:
                    mass_cuts['index'][i] = 0
                else:
                    mass_cuts['index'][i] = max(inds)+1

            # Now put it in the file
            with h5py.File(os.path.join(self.path,'data.hdf5'), "a") as file:

                if "Snapshot {}".format(snap) in file.keys():
                    del file["Snapshot {}".format(snap)]
                snap_grp = file.create_group("Snapshot {}".format(snap))

                snap_grp.create_dataset('mass_cuts',data=mass_cuts)

                for key in basic_properties.dtype.names:
                    snap_grp.create_dataset(key,data=basic_properties[key])
                    snap_grp[key].attrs['units'] = basic_units[key]
                    snap_grp[key].attrs['h dependence'] = basic_h[key]

                for key in self.other_fields.keys():
                    n_subkeys = len(self.other_fields[key])
                    if n_subkeys > 1:
                        for subkey in range(n_subkeys):
                            new_key = self.other_fields[key][subkey][0]
                            if new_key[:4] != 'none':
                                snap_grp.create_dataset(new_key,data=subhalos[key][:,subkey])
                                snap_grp[new_key].attrs['units'] = self.other_fields[key][subkey][2]
                                snap_grp[new_key].attrs['h dependence'] = self.other_fields[key][subkey][3]
                    else:
                        new_key = self.other_fields[key][0][0]
                        snap_grp.create_dataset(new_key,data=subhalos[key])
                        snap_grp[new_key].attrs['units'] = self.other_fields[key][0][2]
                        snap_grp[new_key].attrs['h dependence'] = self.other_fields[key][0][3]

                snap_meta = self.snap_meta[self.snap_meta['index'] == snap]
                snap_grp.attrs['metadata'] = snap_meta
            
            # Delete raw data if requested
            if realtime_clean_raw:
                file_path = os.path.join(self.path,'raw',self._get_rawsnapfile(snap))
                os.system("rm -r {}".format(file_path))



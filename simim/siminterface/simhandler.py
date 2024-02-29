import warnings
import os

import h5py
import numpy as np

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from simim._paths import _SimIMPaths
from simim._handlers import Handler
# from simim.map import gridder
from simim.siminterface._sims import _checksim

class SnapHandler(Handler):
    """Handler for individual snapshots - see generic Handler
    documentation. 
    
    The simplest way to initialize a SnapHandler instance is
    probably via a SimHandler instance for the simulation 
    containing the snapshot in question. Then the method 
    SimHandler.get_snap will return a Handler instance for 
    the snapshot with only the snapshot index-number 
    specified.
    """

    def __init__(self,path,snap,redshift,cosmo,box_edge):
        """Initialize Handler for a simulation snapshot
        
        Parameters
        ----------
        path : string
            Path to SimIM formatted file containing the snapshot
            (most likely .../[Simulation Name]/data.hdf5)
        snap : int
            Index-number of the snapshot within the whole simulation
        redshift : float
            Redshift at which the snapshot was taken
        cosmo : dict
            Dictionary containing the cosmological parameters for the 
            simulation
        box_edge : float
            The edge length of the simulation box, should be in units 
            of Mpc/h.
        """

        super().__init__(path,objectname='Snapshot',groupname='Snapshot {}'.format(snap))

        self.redshift = redshift
        self.cosmo = cosmo
        self.box_edge = box_edge
        self.box_edge_no_h = box_edge / self.h

        self.extra_props['cosmo'] = self.cosmo
        self.extra_props['redshift'] = self.redshift
        self.extra_props['box_edge'] = self.box_edge
        self.extra_props['box_edge_no_h'] = self.box_edge / self.h

    def grid(self,*property_names,
             in_h_units=False,use_all_inds=False,
             res=None,xlim=None,ylim=None,zlim=None,
             norm=None):
        """Place selected properties into a 3d grid
        
        Uses the properties of the array to construct a position (pos_x,pos_y,pos_z)-
        value (property_names) grid. Only required argument is a valid property name
        or names. Additional arguments can specify the limits and resolution of the grid

        Parameters
        ----------
        property_names : str
            The name or names of properties in the Handler instance
        in_h_units : bool, default=False
            If True, positions and property values fed to the gridder will be
            in units including little h. If False, little h dependence will be 
            removed.
        use_all_inds : bool, default=False
            If True function all halos will be gridded, otherwise only
            active halos will be included.
        res : float, optional
            The resolution for the grid in Mpc (if in_h_units==False) or Mpc/h
            (if in_h_units==True). If no value is specified, it will default to
            1/100th of the box edge length
        xlim, xylim, zlim : tuples, optional
            Tuples containing minimum and maximum values of the grid along the x, 
            y, and z axes, in units of Mpc (if in_h_units==False) or Mpc/h (if 
            in_h_units==True). If no values are specified the defaults are (0, box 
            edge length).
        norm : None, 'cell_volume', float
            Apply a normalization to the gridded values. Default is None, if 'cell_volume'
            is specified each cell will be divided by its volume. If a float is given
            each cell will multiplied by the float

        Returns
        -------
        grid : simim.map.grid instance
            The gridded properties
        """

### Fix this once maps is implemented
        raise ValueError("simim version does not support gridding")

        x = self.return_property('pos_x',in_h_units=in_h_units,use_all_inds=use_all_inds)
        y = self.return_property('pos_y',in_h_units=in_h_units,use_all_inds=use_all_inds)
        z = self.return_property('pos_z',in_h_units=in_h_units,use_all_inds=use_all_inds)
        props = np.array([self.return_property(p,in_h_units=in_h_units,use_all_inds=use_all_inds) for p in property_names]).T
        
        if in_h_units:
            side = self.box_edge
        else:
            side = self.box_edge_no_h

        c = []
        l = []

        for lim in [xlim,ylim,zlim]:
            if lim is None:
                c.append(side/2)
                l.append(side)
            elif len(lim) != 2:
                raise ValueError("xlim, ylim, and zlim must each be None or have length = 2")
            else:
                c.append((lim[0]+lim[1])/2)
                l.append(np.max(lim) - np.min(lim))

        if res is None:
            res = [x/100 for x in l]

        grid = gridder(np.array([x, y, z]).T, props,
                       center_point=c, side_length=l,
                       pixel_size=res)

        if norm is None:
            norm = 1
        elif norm == 'cell_volume':
            norm = 1/np.prod(grid.pixel_size)
        
        grid.grid *= norm

        return grid



class SimHandler():
    """Class to handle I/O for subhalo/galaxy catalogs in SimIM format
    
    This class handles basic operations for accessing and masking simulation
    data. It is a wrapper around Handlers for individual snapshots, and
    in many cases performs operations iteratively over all snapshots in 
    a particular simulation. It is also a convenient wrapper for accessing
    specific snapshots of a given simulation.

    The general philosophy of Handlers is to not load actual properties of 
    a halo into memory until they are requested, and to remove them from 
    memory when they are no longer in use (or at least make it convenient
    to do so).
    """

    def __init__(self,sim,init_snaps=False):
        """Initialize Handler for a specified simulation

        Provides a generic interface for interacting with data from
        any simulation that has been converted to SimIM format and is
        accessible to memory. Note that simulations must be downloaded
        and formatted (see e.g. simim.siminterface.illustris or 
        simim.siminterface.universemachine for code to accomplish this)
        
        Parameters
        ----------
        sim : string
            Name of the simulation to load.
        init_snaps : bool, default=False
            Setting this as True will create persistent Handler instances 
            for every snapshot, rather than doing so when data from a given 
            handler is called for. This is generally not necessary, but is
            used when creating properties for all snapshots but NOT writing
            them to disk.
        """

        # Check that we can handle the specified sim
        _checksim(sim)

        # Set up a place to keep the data
        paths = _SimIMPaths()
        if sim in paths.sims:
            self.path = paths.sims[sim]
        else:
            raise ValueError("Simulation {} not available. Try installing it or updating the path".format(sim))
        self.sim = sim

        # Get the metadata
        self.metadata = {}
        with h5py.File(os.path.join(self.path,'data.hdf5'),'r') as file:
            for key in file.attrs.keys():
                self.metadata[key] = file.attrs[key]
            self.snap_meta = file.attrs['snapshots']
        self.h = self.metadata['cosmo_h']

        # Sort out redshift matching to stuff
        snaps_sorted = np.sort(np.copy(self.snap_meta),order='redshift')
        self.z_bins = [snap['redshift_min'] for snap in snaps_sorted]
        self.z_bins.append(snaps_sorted[-1]['redshift_max'])
        self.z_bin_snaps = [snap['index'] for snap in snaps_sorted]

        # Make the cosmology
        self.cosmo = FlatLambdaCDM(H0=100*self.metadata['cosmo_h']*u.km/u.s/u.Mpc,
                                   Om0=self.metadata['cosmo_omega_matter'],
                                   Ob0=self.metadata['cosmo_omega_baryon'],
                                   Tcmb0=2.7255*u.K)
        self.box_edge = self.metadata['box_edge']
        self.box_edge_no_h = self.box_edge / self.h

        # Get keys and units
        with h5py.File(os.path.join(self.path,'data.hdf5'),'r') as file:
            snaps = [i for i in file.keys()]
            keys = [i for i in file[snaps[0]].keys() if i != 'mass_cuts']

            key_units = {key:file[snaps[0]][key].attrs['units'] for key in keys}
            key_h_dependence = {key:file[snaps[0]][key].attrs['h dependence'] for key in keys}

        self.keys = keys
        self.key_units = key_units
        self.key_h_dependence = key_h_dependence

        self.init_snaps = init_snaps
        if self.init_snaps:
            self.initialize_all_snaps(redo=True)

    def initialize_all_snaps(self,remake=False):
        """Initialize SnapHandlers for each snapshot
        
        Parameters
        ----------
        remake : bool, default=False
            Determines whether snapshots should be re-initialized if 
            this method is called twice
        """

        if self.init_snaps and not remake:
            raise ValueError("Snapshots already initialized")
        
        print('Initializing snapshots, this may take a few seconds')
        self.snap_handlers = {}
        # Set up snapshot handlers
        for i in range(len(self.snap_meta)):

            snap = self.snap_meta['index'][i]
            redshift = self.snap_meta['redshift'][i]

            self.snap_handlers[str(snap)] = SnapHandler(self.path+'/data.hdf5',snap,redshift,self.cosmo,self.box_edge)
        print("Snapshots initialized.")
        self.init_snaps = True

    def number_volumes(self,volume,in_h_units=False):
        """Compute the number of times a specified volume can fit in the simulation box
        
        Parameters
        ----------
        volume : float
            The volume to check in units of Mpc^3
        in_h_units : bool, default = False
            If True the value of volume will be assumed to have units of
            (Mpc/h)^3
        """

        if in_h_units:
            return volume / self.box_edge**3
        else:
            return volume / self.box_edge_no_h**3

    def extract_snap_meta(self,snap):
        """Get the meta-data for a snapshot

        Parameters
        ----------
        snap : int
            Number of snapshot to be extracted

        Returns
        -------
        snap_meta
            The meta data for the requested snapshot
        """

        if snap in self.snap_meta['index']:
            snap_meta = self.snap_meta[self.snap_meta['index'] == snap][0]
            return snap_meta
        else:
            raise ValueError("Snapshot not found")

    def z_to_snap(self,z):
        """Determine the snapshot corresponding to a particular redshift

        Parameters
        ----------
        z : float
            Redshift to search for

        Returns
        -------
        snap_ind
            The index number of the snapshot matching the requested redshift
        """

        if z > np.amax(self.z_bins) or z < 0:
            raise ValueError("z out of range")
        bin = np.digitize(z,self.z_bins)-1
        bin_id = self.z_bin_snaps[bin]

        return bin_id

    def extract_snap_keys(self):
        """Get the fields associated with halos in the simulation

        Parameters
        ----------
        none

        Returns
        -------
        keys
            The fields of each snapshot
        """

        return self.keys

    def get_mass_index(self,mass,snap,in_h_units=False):
        """Find the indices above a specified mass

        Parameters
        ----------
        mass : float
            Minimum mass to access in Msun units
        snap : int
            Number of snapshot to be extracted
        in_h_units : bool (default=False)
            If True, mass will be taken to have units including little h,
            otherwise, it will be assumed to have units with no h dependence.

        Returns
        -------
        index : int
            The index
        """

        if not snap in self.snap_meta['index']:
            raise ValueError("Snapshot not found")

        with h5py.File(os.path.join(self.path,'data.hdf5'),'r') as file:
            mass_cuts = file["Snapshot {}".format(snap)]['mass_cuts']

            if in_h_units:
                mass_check = mass
            else:
                mass_check = mass / self.h
            vals = mass_cuts['min_mass'] - mass_check
            vals = vals[vals>0]
            if len(vals) < 1:
                index = 0
            elif len(vals) == len(mass_cuts):
                index = mass_cuts['index'][-1]
            else:
                index = mass_cuts['index'][len(vals)]

        return index

    def get_snap(self,snap):
        """Return a SnapHandler instance for a specified snapshot
        
        Parameters
        ----------
        snap : int
            Index-number of the desired snapshot

        Returns
        -------
        SnapHandler
            A SnapHandler instance for the requested snapshot
        """

        snap_meta = self.extract_snap_meta(snap)

        if self.init_snaps:
            return self.snap_handlers[str(snap_meta['index'])]
        else:
            snap = snap_meta['index']
            redshift = snap_meta['redshift']
            return SnapHandler(self.path+'/data.hdf5',snap,redshift,self.cosmo,self.box_edge)

    def get_snap_from_z(self,z):
        """Return a SnapHandler instance for the snapshot closest to a requested redshift
        
        Parameters
        ----------
        z : float
            The desired redshift for the snap
        
        Returns
        -------
        SnapHandler
            A SnapHandler instance for the requested snapshot
        """

        return self.get_snap(self.z_to_snap(z))

    def set_property_range(self,property_name=None,pmin=-np.inf,pmax=np.inf,reset=True, in_h_units=False):
        """Restrict property range for all snapshots
        
        This is a wraper around SnapHandler.set_property_range
        that iteratively applies it to all snapshots. Initializing
        handlers for each snapshot is necessary for this to work.

        Parameters
        ----------
        property_name : str
            The name of the field to use
        pmin : float
            The minimum value of the property to bracket the selected range.
        pmax : float
            The maximum value of the property to bracket the selected range.
        reset : bool, optional
            If True, the active indices will be those selected between pmin and
            pmax. If False, the active indices will be that satisfy pmin<=p<=pmax
            and which were previously in the active indices (ie this allows
            selection over multiple properties.)
        in_h_units : bool (default=False)
            If True, pmin and pmax will be taken to have units including little h,
            otherwise, they will be assumed to have units with no h dependence
            (and have the correct dependency applied before setting cuts for parameters
            where the stored catalog values are in h units).
        """

        if not self.init_snaps:
            raise ValueError("This handler instance was not initialized with snapshots available")
        for i in range(len(self.snap_meta)):
            snap = self.snap_meta['index'][i]
            handler = self.get_snap(snap)
            handler.set_property_range(property_name=property_name,pmin=pmin,pmax=pmax,reset=reset,in_h_units=in_h_units)

    def make_property(self, property, rename=None, kw_remap={}, other_kws={}, overwrite=False, use_all_inds=False, write=False, writedtype=None):
        """Use a galprops.prop instance to evaluate a new property over all snapshots

        This is a wraper around SnapHandler.make_property that iteratively 
        applies it to all snapshots. For this to work, either 1) write must
        be set to True (resulting in the new property being saved to disk) or
        2) SnapHandlers must be initialized for each snapshot, in which case
        the property can be stored only in memory. The latter is likely to 
        require a significant allocation of memory and should be used carefully.

        Parameters
        ----------
        property : galprops.property instance
            The galprops.property instance containing the property information
            and generating function
        rename : list, optional
            List of names specifying how to rename the property from the name
            specified in the galprops.prop instance
        kw_remap : dict, optional
            A dictinary remaping kwargs of the property generating function to
            different properties of the lightcone. By default if the function
            calls for kwarg 'x' it will be evaluated on simulation property 'x', but
            passing the dictionary {'x':'y'} will result in the function being 
            evaluated on simulation property 'y'.
        other_kws : dict, optional
            A dictionary of additional keyword arguments passed directly to
            the property.prop_function call
        overwrite : bool, default=False
            Default is False. If a property name is already in use and overwrite
            is False, an error will be raised. Otherwise the property will be
            overwritten.
        use_all_inds : bool, default=False
            If True values will be assigned for all halos, otherwise only
            active halos will be evaluated, and others will be assigned nan.
        write : bool, default=False
            If True values of the new property will be written to the disk storage
            for the simulation snapshots.
        writedtype : None or dtype
            Specifies the data format to write the new property in.

        Returns
        -------
        None
        """

        if not self.init_snaps and not write:
            raise ValueError("This handler instance was not initialized with snapshots available")
        for i in range(len(self.snap_meta)):
            snap = self.snap_meta['index'][i]
            print("\033[1m"+"Assigning props for Snapshot {}.  ".format(snap)+"\033[0m",end='\r')
            handler = self.get_snap(snap)

            handler.make_property(property=property,
                                  rename=rename,
                                  kw_remap=kw_remap,
                                  other_kws=other_kws,
                                  overwrite=overwrite,
                                  use_all_inds=use_all_inds)

            if write:
                if isinstance(rename,str):
                    rename = [rename]
                if rename is None:
                    names = property.names
                elif len(rename) != property.n_props:
                    raise ValueError("Length of rename list doesn't match number of properties")
                else:
                    names = rename

                if i==0:
                    with h5py.File(handler.path,'a') as file:
                        for name in names:
                            if name in file[handler.groupname].keys():
                                if not overwrite:
                                    raise ValueError("Property {} already exists".format(name))
                                elif 'userdefined' not in file[handler.groupname][name].attrs.keys():
                                    raise ValueError("Property {} is not userd-defined and cannot be overwritten".format(name))
                                else:
                                    warnings.warn("Property {} already exists, overwriting".format(name))

                handler.write_property(*names,overwrite=overwrite,dtype=writedtype)
                handler.unload_property(*names)
        print("")

    def delete_property(self,*property_names):
        """Remove a property from the saved file on the disk for all simulation snapshots

        Parameters
        ----------
        property_names : str
            The name of the field to be written, can give multiple

        Returns
        -------
        None
        """

        for i in range(len(self.snap_meta)):
            snap = self.snap_meta['index'][i]
            handler = self.get_snap(snap)

            if i==0:
                with h5py.File(handler.path,'a') as file:
                    for name in property_names:
                        if name in file[handler.groupname].keys():
                            if 'userdefined' not in file[handler.groupname][name].attrs.keys():
                                raise ValueError("Property {} is not userd-defined and cannot be deleted".format(name))
                            elif not file[handler.groupname][name].attrs['userdefined']:
                                raise ValueError("Property {} is not userd-defined and cannot be deleted".format(name))

            handler.delete_property(*property_names)

    def snap_stat(self, stat_function, kwargs, kw_remap={}, other_kws={},
                  give_args_in_h_units=False, use_all_inds=False, snaps=None):
        """Evaluate stat_function over every snapshot and return results

        This is a wraper around SnapHandler.eval_stat that iteratively 
        applies it to all snapshots. 
        
        Parameters
        ----------
        stat_function : function
            Any function which can be applied to data in a snapshot
        kwargs : list
            List containing the arguments that must be passed to stat_function
        kw_remap : dict
            Dictionary mapping between function arguments (listed in kwargs) as 
            the keys and the names of handler properties to feed in as values.
            E.g. to provide handler property 'mass' to stat_function argument 'a'
            one would use kw_remap={'a':'mass'}
        other_kws : dict, optional
            A dictionary of additional keyword arguments passed directly to
            the stat_function call
        use_all_inds : bool, default=False
            If True function will be computed using all halos, otherwise only
            active halos will be evaluated.
        give_args_in_h_units : bool, default=False
            If True, values will be fed to stat_function in units including little h.
            If False, little h dependence will be removed.
        snaps : list, optional
            A list of snapshots on which to evaluate the stat_function. If none is 
            specified all snapshots will be used.
            
        Returns
        -------
        vals : list
            List containing the value(s) returned by stat_function on each snapshot
        redshifts : list
            List containing the redshift of each snapshot        
        """

        vals = []
        redshifts = []

        if snaps is None:
            snaps = np.arange(len(self.snap_meta))
        for i in snaps:
            snap = self.snap_meta['index'][i]
            print("\033[1m"+"Collecting sources from Snapshot {}.  ".format(snap)+"\033[0m",end='\r')
            redshifts.append(self.snap_meta['redshift'][i])

            handler = self.get_snap(snap)
            vals.append(handler.eval_stat(stat_function,kwargs,kw_remap,other_kws=other_kws,use_all_inds=use_all_inds,give_args_in_h_units=give_args_in_h_units))
        print("")
            
        return vals, redshifts

# Wrapper for back compatibility
def simhandler(*args, **kwargs):
    warnings.warn("simhandler is depricated, use SimHandler instead")
    return SimHandler(*args, **kwargs)
def snaphandler(*args, **kwargs):
    warnings.warn("snaphandler is depricated, use SnapHandler instead")
    return SnapHandler(*args, **kwargs)
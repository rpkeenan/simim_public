import os
import warnings
from fnmatch import fnmatch

from urllib.request import urlretrieve
import requests

import numpy as np

from simim.siminterface._rawsiminterface import SimCatalogs, Snapshot
from simim.siminterface._sims import _checksim

class UniversemachineCatalogs(SimCatalogs):
    def __init__(self,
                 sim, path='auto',
                 snaps='all',
                 updatepath=True,
                 ):
        """Initialize an interface with the UniverseMachine Bolshoi/MD catalogs
        
        Parameters
        ----------
        sim : string
            Name of the simulation/UM catalog you want to download/
            format. Options are 'UniverseMachine-BolshoiPlanck',
            'UniverseMachine-SMDPL','UniverseMachine-MDPL2'
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

        if sim == 'UniverseMachine-BolshoiPlanck':
            self.allsnaps = np.arange(178)
        elif sim == 'UniverseMachine-SMDPL':
            self.allsnaps = np.arange(118)
        elif sim == 'UniverseMachine-MDPL2':
            self.allsnaps = np.arange(126)

        super().__init__(sim, path, snaps, updatepath)

        self.web_path = os.path.join(self.path, 'web_paths.npy')
        if self.sim == 'UniverseMachine-BolshoiPlanck':
            self.webpage = 'https://halos.as.arizona.edu/UniverseMachine/DR1/SFR_ASCII/'
            self._loader = self._ascii_loader
        elif self.sim == 'UniverseMachine-SMDPL':
            self.webpage = 'https://halos.as.arizona.edu/UniverseMachine/DR1/SMDPL_SFR/'
            self._loader = self._bin_loader
        elif self.sim == 'UniverseMachine-MDPL2':
            self.webpage = 'https://halos.as.arizona.edu/UniverseMachine/DR1/MDPL2_SFR/'
            self._loader = self._bin_loader

        if os.path.exists(self.web_path):
            self.web_files = np.load(self.web_path)
        else:
            # Get file names and scale factors from the interweb
            try:
                html_text = requests.get(self.webpage).text
                files = [t for t in html_text.split(' ') if fnmatch(t, "*sfr_catalog_*")]
                files = [t.split('"')[1] for t in files]
                self.web_files = np.array(files)
            except:
                raise ValueError("Unable to access file names on {}".format(self.webpage))
            np.save(self.web_path,self.web_files)

        # Identify keys that will require unit conversions
        # For UM these are mostly empty because everythings in units
        # we like already.
        # Note that binary files have different units for halo masses
        # than the text files + different format for position/velocity data
        if self.sim == 'UniverseMachine-BolshoiPlanck':
            self.basic_fields = {
                #X Y Z: halo position (comoving Mpc/h)
                #VX VY VZ: halo velocity (physical peculiar km/s)
                'x':[('pos_x','f','Mpc/h',-1)],
                'y':[('pos_y','f','Mpc/h',-1)],
                'z':[('pos_z','f','Mpc/h',-1)],
                'vx':[('v_x','f','km/s',0)],
                'vy':[('v_y','f','km/s',0)],
                'vz':[('v_z','f','km/s',0)],

                #M: Halo mass (Bryan & Norman 1998 virial mass, Msun)
                'm':[('mass','f','Msun/h',-1)],
                }

        else:
            self.basic_fields = {
                #X Y Z: halo position (comoving Mpc/h)
                #VX VY VZ: halo velocity (physical peculiar km/s)
                'pos':[('pos_x','f','Mpc/h',-1), ('pos_y','f','Mpc/h',-1), ('pos_z','f','Mpc/h',-1), 
                       ('v_x','f','km/s',0), ('v_y','f','km/s',0), ('v_z','f','km/s',0)],

                #M: Halo mass (Bryan & Norman 1998 virial mass, Msun)
                'm':[('mass','f','Msun/h',-1)],
                }

        # Format: 'name_in_parent':[('p1_name_in_simim','p1_dtype','p1_unit',p1_little_h_exponent), ... ('pn_name_in_simim','pn_dtype','pn_unint',pn_little_h_exponent)]
        self.dm_fields = {
            #ID: Unique halo ID
            'id':[('halo_id','i','None',0)],

            #DescID: ID of descendant halo (or -1 at z=0).
            'descid':[('descendent_id','i','None',0)],

            #UPID: -1 for central halos, otherwise, ID of largest parent halo
            'upid':[('parent_id','i','None',0)],

            #V: Halo vmax (physical km/s)
            'v':[('vmax','f','km/s',0)],

            #MP: Halo peak historical mass (BN98 vir, Msun/h)
            'mp':[('mass_peak','f','Msun/h',-1)],

            #VMP: Halo vmax at the time when peak mass was reached.
            'vmp':[('vmax_peak','f','km/s',0)],

            #R: Halo radius (BN98 vir, comoving kpc/h)
            'r':[('r','f','kpc/h',-1)],

            #Rank1: halo rank in Delta_vmax (see UniverseMachine paper)
            'rank1':[('Rank1','f','None',0)],
            }

        self.matter_fields = {
            #SM: True stellar mass (Msun)
            'sm':[('m_stars','f','Msun/h',-1)],

            #ICL: True intracluster stellar mass (Msun)
            'icl':[('icl','f','Msun/h',-1)],

            #SFR: True star formation rate (Msun/yr)
            'sfr':[('sfr','f','Msun/yr',0)],

            #Obs_SM: observed stellar mass, including random & systematic errors (Msun)
            'obs_sm':[('m_stars_obs','f','Msun/h',-1)],

            #Obs_SFR: observed SFR, including random & systematic errors (Msun/yr)
            'obs_sfr':[('sfr_obs','f','Msun/yr',0)],

            #Obs_UV: Observed UV Magnitude (M_1500 AB)
            'obs_uv':[('phot_uv_obs','f','mag',0)],
            }
        
        # Fields in ascii but not binary files: smhm, obs_sssfr
        if self.sim == 'UniverseMachine-BolshoiPlanck':
            #SSFR: observed SSFR
            self.matter_fields['obs_ssfr'] = [('ssfr_obs','f','?',0)]
            #SMHM: SM/HM ratio
            self.matter_fields['smhm'] = [('stellar_to_halo_mass_ratio','f','None',0)]
        
        # Fields in binary but not ascii files: lvmp, A_UV, empty
        else:
            #A_UV: UV attenuation (mag)
            self.matter_fields['a_uv'] = [('uv_attenuation','f','mag',0)]



        # Identify keys that will require unit conversions
        def apply_h(x,h):
            return x * h
        def kpc_to_mpc(x,h):
            return x / 1000
        
        self.transform_keys = {'sm':apply_h,
                               'icl':apply_h,
                               'obs_sm':apply_h,
                               'r':kpc_to_mpc}
        if self.sim == 'UniverseMachine-BolshoiPlanck':
            self.transform_keys['m'] = apply_h
            self.transform_keys['mp'] = apply_h

        # Check whether snapshots have been downloaded
        not_downloaded = []
        for snap in self.snaps:
            file_path = os.path.join(self.path,'raw',self.web_files[snap])
            if not os.path.exists(file_path):
                not_downloaded.append(snap)
        if len(not_downloaded) > 0:
            warnings.warn("No data exists for snapshots {} - run .download".format(not_downloaded))

    # Functions to load the data in a format we want:
    # Fields to ignore: flags, uparent_dist, rank2, ra, rarank
    def _ascii_loader(self, path, snapshot, fields):
        """Loader to get a field from a snapshot halo catalog - for ascii formatted
        UM data

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

        dtype_raw = np.dtype(dtype=[('id', 'i'), ('descid','i'),('upid','i'),('flags','i'),('uparent_dist','f'),
                                    ('x','f'),('y','f'),('z','f'),('vx','f'),('vy','f'),('vz','f'),
                                    ('m','f'),('v','f'),('mp','f'),('vmp','f'),('r','f'),
                                    ('rank1','f'),('rank2','f'),('ra','f'),('rarank','f'),
                                    ('sm','f'),('icl','f'),('sfr','f'),('obs_sm','f'),('obs_sfr','f'),('obs_ssfr','f'),('smhm','f'),('obs_uv','f')])
        data_raw = np.genfromtxt(os.path.join(path,self.web_files[snapshot]), dtype=dtype_raw, comments='#')
        subhaols = {}
        for key in fields:
            subhaols[key] = data_raw[key]
        n_halos = len(data_raw)

        return subhaols, n_halos

    def _bin_loader(self, path, snapshot, fields):
        """Loader to get a field from a snapshot halo catalog - for binary formatted
        UM data

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
        
        dtype_raw = np.dtype(dtype=[('id', 'i8'),('descid','i8'),('upid','i8'),
                                    ('flags', 'i4'), ('uparent_dist', 'f4'),
                                    ('pos', 'f4', (6)), ('vmp', 'f4'), ('lvmp', 'f4'),
                                    ('mp', 'f4'), ('m', 'f4'), ('v', 'f4'), ('r', 'f4'),
                                    ('rank1', 'f4'), ('rank2', 'f4'), ('ra', 'f4'),
                                    ('rarank', 'f4'), ('a_uv', 'f4'), ('sm', 'f4'),
                                    ('icl', 'f4'), ('sfr', 'i4'), ('obs_sm', 'f4'),
                                    ('obs_sfr', 'f4'), ('obs_uv', 'f4'), ('empty', 'f4')],
                             align=True)

        flag = 16
        data_raw = np.fromfile(os.path.join(path,self.web_files[snapshot]), dtype=dtype_raw)
        data_raw = data_raw[(data_raw['flags'] & flag) != 16]

        subhalos = {}
        for key in fields:
            subhalos[key] = data_raw[key]
        subhalos['pos'][:,0:3] % self.box_edge
        n_halos = len(data_raw)

        return subhalos, n_halos

    def _get_rawsnapfile(self, snapshot):
        """Get path to a snapshot's raw file"""
        return self.web_files[snapshot]

    def download_meta(self, redownload=False):
        """Download and generate metadata for the set of snapshots 
        specified when initalizing the class
        
        Note: the metadata saved is dependent on the list of snapshots,
        therefore if you plan to use many snapshots for some applications
        but have for some reason only initialized your IllustrisCatalogs
        instance with a few it is probably best to do something like the 
        following:
            >>> x = UniversemachineCatalogs(...,snaps='all')
            >>> x.download_meta()
            >>> x = UniversemachineCatalogs(...,snaps=[10,11,12])
            >>> x.download()
        """
        
        # Check that metadata doesn't already exist
        if not redownload:
            if os.path.exists(self.meta_path):
                warnings.warn("Metadata appears to exist already")
                return

        self.metadata = {'name':self.sim,
                         'number_snaps':len(self.web_files)}
        # Assign the correct data for different simulation boxes
        if self.sim in ['UniverseMachine-BolshoiPlanck','UniverseMachine-SMDPL','UniverseMachine-MDPL2']:
            self.metadata['cosmo_name'] = 'Planck'
            self.metadata['cosmo_omega_matter'] = 0.307
            self.metadata['cosmo_omega_lambda'] = 0.693
            self.metadata['cosmo_omega_baryon'] = 0.048
            self.metadata['cosmo_h'] = 0.678

        if self.sim == 'UniverseMachine-BolshoiPlanck':
            self.metadata['box_edge'] = 250
        elif self.sim == 'UniverseMachine-SMDPL':
            self.metadata['box_edge'] = 400
        elif self.sim == 'UniverseMachine-MDPL2':
            self.metadata['box_edge'] = 1000

        self.box_edge = self.metadata['box_edge']
        self.h = self.metadata['cosmo_h']

        # Scale factors, redshifts, and sort files (read from file names)
        a = np.array([float(t.split('_')[-1][:-4]) for t in self.web_files])
        redshifts = 1/a-1
        order = np.argsort(a)
        a = a[order]
        redshifts = redshifts[order]
        self.web_files = self.web_files[order]

        # Snapshots for different simulation boxes
        numbers = np.arange(len(self.web_files))

        snap_meta_classes = []
        for i in range(self.metadata['number_snaps']):
            snap_meta_classes.append(Snapshot(i,redshifts[i],self.metadata))

        snap_meta_classes = [snap_meta_classes[i] for i in numbers if i in self.snaps]

        snap_meta_classes[0].dif_higherz_snap('max')
        snap_meta_classes[-1].dif_lowerz_snap(0)
        for i in range(len(snap_meta_classes)-1):
            snap_meta_classes[i].dif_lowerz_snap(snap_meta_classes[i+1])
            snap_meta_classes[i+1].dif_higherz_snap(snap_meta_classes[i])

        snap_meta_dtype = [('index','i'),
                           ('redshift','f'),
                           ('redshift_min','f'),('redshift_max','f'),
                           ('time_start','f'),('time_end','f'),
                           ('distance_min','f'),('distance_max','f'),
                           ('transverse_distance_min','f'),('transverse_distance_max','f')]
        snap_meta = np.zeros(len(snap_meta_classes),
                             dtype = snap_meta_dtype)

        for i in range(len(snap_meta_classes)):
            snap_meta[i]['index'] = snap_meta_classes[i].index
            snap_meta[i]['redshift'] = snap_meta_classes[i].redshift
            snap_meta[i]['redshift_min'] = snap_meta_classes[i].redshift_min
            snap_meta[i]['redshift_max'] = snap_meta_classes[i].redshift_max
            snap_meta[i]['time_start'] = snap_meta_classes[i].time_start
            snap_meta[i]['time_end'] = snap_meta_classes[i].time_end
            snap_meta[i]['distance_min'] = snap_meta_classes[i].distance_min * self.h
            snap_meta[i]['distance_max'] = snap_meta_classes[i].distance_max * self.h
            snap_meta[i]['transverse_distance_min'] = snap_meta_classes[i].transverse_distance_min * self.h
            snap_meta[i]['transverse_distance_max'] = snap_meta_classes[i].transverse_distance_max * self.h
        self.snap_meta = snap_meta

        np.save(self.meta_path,self.metadata)
        np.save(self.snap_meta_path,self.snap_meta)

    def download(self, redownload=False):
        """Download UniverseMachine catalogs"""

        if not os.path.exists(os.path.join(self.path,'raw')):
            os.mkdir(os.path.join(self.path,'raw'))

        # Download metadata if not present
        self.download_meta(redownload=False)

        # Add a check for already downloaded files
        if redownload:
            self.download_snaps = np.copy(self.snaps)
        else:
            self.download_snaps = []
            for snap in self.snaps:
                file_path = os.path.join(self.path,'raw',self.web_files[snap])
                if os.path.exists(file_path):
                    warnings.warn("Skipping snapshot {} as it appears to exist already".format(snap))
                else:
                    self.download_snaps.append(snap)

        # Download each snap
        for i in range(len(self.download_snaps)):
            snap = self.download_snaps[i]
            print("downloading item {} of {} ({})".format(i+1,len(self.download_snaps),self.web_files[snap]))
            file_path = os.path.join(self.path,'raw',self.web_files[snap])
            urlretrieve(self.webpage+self.web_files[snap],file_path)

# Wrapper for back compatibility
def universemachine_catalogs(*args, **kwargs):
    warnings.warn("universemachine_catalogs is depricated, use UniversemachineCatalogs instead")
    return UniversemachineCatalogs(*args, **kwargs)
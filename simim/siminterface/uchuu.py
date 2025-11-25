import os
import warnings
from fnmatch import fnmatch

from urllib.request import urlretrieve
import requests

import h5py
import numpy as np

from simim.siminterface._rawsiminterface import SimCatalogs, Snapshot

class UchuuCatalogs(SimCatalogs):
    def __init__(self,
                 sim, path='auto',
                 snaps='all',
                 updatepath=True,
                 ):
        """Initialize an interface with the UniverseMachine Uchuu catalogs
        
        Parameters
        ----------
        sim : string
            Name of the simulation/UM catalog you want to download/
            format. Only option currenlty is 'UniverseMachine-uchuu',
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

        if sim == 'UniverseMachine-uchuu':
            self.allsnaps = np.arange(46)

        super().__init__(sim, path, snaps, updatepath)

        self.web_path = os.path.join(self.path, 'web_paths.npy')
        if self.sim == 'UniverseMachine-uchuu':
            self.webpage = 'https://skun.iaa.csic.es/SUsimulations/UchuuDR2/Uchuu_UM/'

        if os.path.exists(self.web_path):
            self.web_files = np.load(self.web_path)
        else:
            # Get file names and scale factors from the interweb
            try:
                html_text = requests.get(self.webpage+'SFR/').text
                web_files = [t for t in html_text.split(' ') if fnmatch(t, "*Uchuu_UM_*data1.h5*")]
                web_files = [t.split('"')[1] for t in web_files]
                self.web_files = np.array(web_files)
            except:
                raise ValueError("Unable to access file names on {}".format(self.webpage))

            # Order files with highest redshift first
            redshifts = np.array([float(t.split('_')[2].strip('z').replace('p','.')) for t in self.web_files])
            order = np.argsort(-redshifts)
            self.web_files = self.web_files[order]

            np.save(self.web_path,self.web_files)

        # Identify keys that will require unit conversions
        # For UM these are mostly empty because everything's in units
        # we like already.
        self.datafile_indices = {k:1 for k in ['id','upid','Mvir','sm','icl','sfr','obs_sm','obs_sfr','obs_uv']} | \
                                {k:2 for k in ['x','y','z']} | \
                                {k:3 for k in ['desc_id','vx','vy','vz','Mpeak','Vmax_Mpeak','vmax','A_UV']} | \
                                {'SFH':4, 'ICHL':5, 'SM_main_progenitor':6, 'ICL_main_progenitor':7, 'M_main_progenitor':8} | \
                                {'SFR_main_progenitor':9, 'V@Mpeak':10, 'A_first_infall':11, 'A_last_infall':12}

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
            'Mvir':[('mass','f','Msun/h',-1)],
            }

        # Format: 'name_in_parent':[('p1_name_in_simim','p1_dtype','p1_unit',p1_little_h_exponent), ... ('pn_name_in_simim','pn_dtype','pn_unint',pn_little_h_exponent)]
        self.dm_fields = {
            #ID: ID of the galaxy
            'id':[('halo_id','i','None',0)],

            #DescID: ID of descendant halo (or -1) at z=0.
            'desc_id':[('descendent_id','i','None',0)],

            #UPID: -1 for central halos, otherwise, ID of largest parent halo
            'upid':[('parent_id','i','None',0)],

            #V: Halo vmax (physical km/s)
            'vmax':[('vmax','f','km/s',0)],

            #MP: Halo peak historical mass (BN98 vir, Msun/h)
            'Mpeak':[('mass_peak','f','Msun/h',-1)],

            #VMP: Halo vmax at the time when peak mass was reached.
            'Vmax_Mpeak':[('vmax_peak','f','km/s',0)],

            # #M_main_progenitor: the main progenitor's halo mass history
            # 'M_main_progenitor':[('m_history','f','Msun/h',-1)],

            # #V@Mpeak: main progenitor's Vmax_Mpeak history
            # 'V@Mpeak':[('vmax_peak_history','f','km/s',0)],

            # #A_first_infall: scale factor at which the galaxy first passed through a larger halo
            # 'A_first_infall':[('z_first_infall','f','km/s',0)],

            # #A_last_infall: scale factor at which the galaxy last passed through a larger halo
            # 'A_last_infall':[('z_last_infall','f','km/s',0)],
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

            #A_UV: UV attenuation (mag)
            'A_UV':[('uv_attenuation','f','mag',0)],

            # #SFH: Summed star formation for all parent halos in each previous snapshot
            # 'SFH':[('sfh','f','Msun/yr',0)],

            # #ICLH: star formation for the present-day population of the ICL
            # 'ICLH':[('icl_sfh','f','Msun/yr',0)],

            # #SM_main_progenitor: stellar mass history of main progenitor
            # 'SM_main_progenitor':[('m_stars_history','f','Msun/h',-1)],

            # #ICL_main_progenitor: ICL stellar mass history of main progenitor
            # 'ICL_main_progenitor':[('icl_history','f','Msun/h',-1)],

            # #SFR_main_progenitor: main progenitor star formation rate history (excluding mergers)
            # 'SFR_main_progenitor':[('sfh_main_progenitor','f','Msun/yr',0)],
            }

        # Identify keys that will require unit conversions
        def apply_h(x,h):
            return x * h
        def scale_to_z(x,h):
            return 1/x-1

        self.transform_keys = {'sm':apply_h,
                               'icl':apply_h,
                               'obs_sm':apply_h,
                            #    'M_main_progenitor':apply_h,
                            #    'SM_main_progenitor':apply_h,
                            #    'ICL_main_progenitor':apply_h,
                            #    'A_first_infall':scale_to_z,
                            #    'A_last_infall':scale_to_z,
                               }

        # Check whether snapshots have been downloaded
        not_downloaded = []
        for snap in self.snaps:
            check = True
            for df in [1,2,3]:
                file_path = os.path.join(self.path,'raw',self.web_files[snap].replace('data1',f'data{df}'))
                if not os.path.exists(file_path):
                    check = False
            if not check:
                not_downloaded.append(snap)

        if len(not_downloaded) > 0:
            warnings.warn("Data missing for snapshots {} - run .download".format(not_downloaded))

    # Functions to load the data in a format we want:
    # Fields to ignore: flags, uparent_dist, rank2, ra, rarank
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

        # Check that we know where to find fields
        for f in fields:
            if f not in self.datafile_indices.keys():
                raise ValueError(f"Field '{f}' not recognized")
            
        subhalos = {}
        lens = []
        for df in np.unique([v for v in self.datafile_indices.values()]):
            fields_df = [f for f in fields if self.datafile_indices[f]==df]

            # load stuff here...
            if len(fields_df) > 0:
                file = path + self.web_files[snapshot].replace('data1',f'data{df}')

                with h5py.File(file, "r") as file:
                    for f in fields_df:
                        subhalos[f] = file[f][:]
                        lens.append(len(subhalos[f]))

        # Make sure we have the same anount of data for each field:
        if np.any(np.array(lens) != lens[0]):
            raise ValueError("Not all fields return the same number of halos")
        n_halos = lens[0]

        return subhalos, n_halos

    def _get_rawsnapfiles(self, snapshot):
        """Get path to a snapshot's raw file"""
        rawsnapfiles = [self.web_files[snapshot].replace('data1',f'data{i}') for i in range(1,12)]
        return rawsnapfiles

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
                warnings.warn("Metadata appears to exist already: {}".format(self.meta_path))
                return

        self.metadata = {'name':self.sim,
                         'number_snaps':len(self.web_files)}
        # Assign the correct data for different simulation boxes
        if self.sim in ['UniverseMachine-uchuu']:
            self.metadata['cosmo_name'] = 'Planck'
            self.metadata['cosmo_omega_matter'] = 0.3089
            self.metadata['cosmo_omega_lambda'] = 0.6911
            self.metadata['cosmo_omega_baryon'] = 0.0486
            self.metadata['cosmo_h'] = 0.6775
            self.metadata['cosmo_sigma_8'] = 0.8159
            self.metadata['cosmo_ns'] = 0.9667

        if self.sim == 'UniverseMachine-uchuu':
            self.metadata['box_edge'] = 2000

        self.box_edge = self.metadata['box_edge']
        self.h = self.metadata['cosmo_h']

        # redshifts, and sort files (read from file names)
        redshifts = np.array([float(t.split('_')[2].strip('z').replace('p','.')) for t in self.web_files])
        order = np.argsort(-redshifts)
        redshifts = redshifts[order]

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

    def download(self, redownload=False, datafiles=[1,2,3]):
        """Download UniverseMachine-uchuu catalogs
        
        Parameters
        ----------
        redownload : bool datafiles : optional, list of ints
            The datafiles to include - 1: sfr, stellar mass, halo mass, etc. 2:
            position, 3: additional galaxy properties, 4-12: star formation
            history and related. See https://skun.iaa.csic.es/SUsimulations/UchuuDR2/Uchuu_UM/Readme_Uchuu_UM_data_structure.txt
            for details. Defaults to [1, 2, 3]. Note that only data in
            files 1, 2, 3 is currently used/supported
        """

        if not os.path.exists(os.path.join(self.path,'raw')):
            os.mkdir(os.path.join(self.path,'raw'))

        # Download metadata if not present
        if not os.path.exists(self.meta_path):
            self.download_meta()

        # Add a check for already downloaded files
        if redownload:
            self.download_snaps = [(s,datafiles) for s in self.snaps]
        else:
            self.download_snaps = [(s,[]) for s in self.snaps]
            for i,snap in enumerate(self.snaps):
                file_path = os.path.join(self.path,'raw',self.web_files[snap])
                
                for df in datafiles:
                    if os.path.exists(file_path.replace('data1',f'data{df}')):
                        warnings.warn(f"Skipping snapshot {snap} / datafile {df} as it appears to exist already")
                    else:
                        self.download_snaps[i][1].append(df)
            self.download_snaps = [ds for ds in self.download_snaps if len(ds[1])>0]
        
        # Download each snap
        for i in range(len(self.download_snaps)):
            snap_id, snap_df = self.download_snaps[i]
            file_path = os.path.join(self.path,'raw',self.web_files[snap_id])
            for df in snap_df:
                if df <= 3:
                    webpage = self.webpage + 'SFR/'
                if df > 3:
                    webpage = self.webpage + 'SFH/'
                print("downloading snapshot {} of {} ({})".format(i+1,len(self.download_snaps),self.web_files[snap_id].replace('data1',f'data{df}')))
                urlretrieve(webpage+self.web_files[snap_id].replace('data1',f'data{df}'),file_path.replace('data1',f'data{df}'))


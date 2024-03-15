import os
import numpy as np

from scipy.interpolate import RegularGridInterpolator

from simim._paths import _SimIMPaths
from simim.siminterface._sims import _checksim
from simim.siminterface import SimHandler
from simim.galprops.galprops_am import am_dfcat, modified_schechter_gen




class g13irlf_base():
    """SFRs based on abundance matching Gruppioni et al. IR luminosity functions"""

    def __init__(self,
                 sim, comp=False, remake=False, haloprop='auto', halomassmin=1e10, lirmin=10**8.5,
                 lf_alpha0=1.2, lf_sigma0=0.5, lf_lstar0=10**9.95, lf_phistar0=10**-2.25,
                 lir_to_sfr=1e-10):
        """
        Parameters
        ----------
        sim : string
            Name of simulation - should correspond to a SimIM formatted halo
            catalog, used to perform the abundance matching.
        comp : bool, default is False
            If set to True, will carry out abundance matching to determine the
            halo-LIR connection if the basic files cannot be loaded
        haloprop : 'vmax', 'mass', or 'auto'
            Halo property to abundance match to IR LF - vmax for peak rotational
            velocity, mass for halo mass. auto will check if vmax is present in
            the catalog and use this, or use mass otherwise. Ignored if
            comp==False
        halomassmin : float
            Minimum halo mass to consider when abundance matching in units of
            Msun, defaults to 10^10 Msun. Ignored if comp==False
        lirmin : float
            Minimum IR luminosity to consider when abundance matching
        remake : bool
            Rerun the abundance matching when initializing. Ignored if
            comp==False
        lf_alpha0, lf_sigma0, lf_lstar0, lf_phistar0 : float
            Redshift 0 LF parameters passed to G13 scaling
        lir_to_sfr : float, optional
            Sets the conversion between SFR and IR luminosity (default is 1e-10,
            correct for Chambrier IMF)
        """

        paths = _SimIMPaths()

        _checksim(sim)
        self.sim = sim
        self.name = 'g13_'+sim

        if self.name not in paths.props.keys():
            paths._newproppath(self.name)
        
        self.path = paths.props[self.name]

        self.init_lir_function = False
        self._load_lir_function(comp=comp, remake=remake, haloprop=haloprop, halomassmin=halomassmin, lirmin=lirmin)

        # Save pars
        self.lf_alpha0 = lf_alpha0
        self.lf_sigma0 = lf_sigma0
        self.lf_lstar0 = lf_lstar0
        self.lf_phistar0 = lf_phistar0

        self.lir_to_sfr = lir_to_sfr

    def _load_lir_function(self, comp, remake, haloprop, halomassmin, lirmin):
        """Load or compute halo mass-ir luminosity relation"""

        pathexists = os.path.exists(os.path.join(self.path,'lir.npy'))
        if remake or (comp and not pathexists):
            self._make_lir_function(haloprop=haloprop,halomassmin=halomassmin,lirmin=lirmin)
            pathexists = True
        
        if pathexists:
            # Load spline stuff
            halopropgrid = np.load(os.path.join(self.path,'mass_axis.npy'))
            zgrid = np.load(os.path.join(self.path,'redshift_axis.npy'))
            lirgrid = np.load(os.path.join(self.path,'lir.npy'))

            self.interpolator = RegularGridInterpolator((halopropgrid,zgrid), lirgrid, method='linear', bounds_error=False)
            self.init_lir_function = True
    
    def _make_lir_function(self, haloprop, halomassmin, lirmin):
        """Compute halo mass-ir luminosity relation by abundance matching"""

        handler = SimHandler(self.sim)
        if haloprop not in ['vmax','mass','auto']:
            raise ValueError("haloprop used for abundance matching must be one of 'vmax', 'mass', 'auto'")
        if haloprop == 'auto':
            snap0 = handler.get_snap(0)
            if snap0.has_property('vmax'):
                haloprop = 'vmax'
            elif snap0.has_property('mass'):
                haloprop = 'mass'
            else:
                raise ValueError("no suitable property for abundance matching found")

        if haloprop == 'vmax':
            halopropgrid = np.linspace(0,5000,5000)
        if haloprop == 'mass':
            halopropgrid = np.logspace(np.log10(halomassmin), np.log10(halomassmin)+10,5000)

        zgrid = handler.snap_meta['redshift']
        irgrid = np.zeros((len(zgrid),len(halopropgrid)))
        df = modified_schechter_gen()

        # i is counter, idx is snap number
        for i,idx in enumerate(handler.snap_meta['index']):
            print("\033[1m"+"Perroming Abundance Match for Snapshot {}.  ".format(idx)+"\033[0m",end='\r')
            
            snap = handler.get_snap(idx)
            
            phistar, lstar, alpha, sigma = self.get_irlf_pars(snap.redshift)
            df_pars = {'phi0':phistar,'x0':lstar,'alpha':alpha,'sigma':sigma,'xmin':lirmin}
            df_freeze = df(phi0=phistar, x0=lstar, alpha=alpha, sigma=sigma, xmin=lirmin)
            df_total = df.lum_function_int(lirmin, phi0=phistar, x0=lstar, alpha=alpha, sigma=sigma, xmin=lirmin) * snap.box_edge_no_h**3

            snap.set_property_range('mass',halomassmin,np.inf)
            snap_haloprop = snap.return_property(haloprop,use_all_inds=False)
            snap_total = len(snap_haloprop)

            if snap_total > df_total:
                missing_pmass_low_p1 = 0
                missing_pmass_low_p2 = 1 - df_total / snap_total
            else:
                missing_pmass_low_p1 = 1 - snap_total / df_total
                missing_pmass_low_p2 = 0

            if len(snap_haloprop) > 0:
                lirofhalo, _ = am_dfcat(snap_haloprop,df_freeze,missing_pmass_low_p1=missing_pmass_low_p1,missing_pmass_low_p2=missing_pmass_low_p2)
                irgrid[i] = lirofhalo(halopropgrid)

        np.save(os.path.join(self.path,'mass_axis.npy'), halopropgrid)
        np.save(os.path.join(self.path,'redshift_axis.npy'), zgrid)
        np.save(os.path.join(self.path,'lir.npy'), irgrid.T)

    def get_irlf_pars(self, z, alpha0=1.2, sigma0=0.5, lstar0=10**9.95, phistar0=10**-2.25):
        """Use the redshift scaling from Gruppioni et al. 2013 to get IRLF parameters"""
        if alpha0 is None:
            alpha0 = self.lf_alpha0
            sigma0 = self.lf_sigma0
            lstar0 = self.lf_lstar0
            phistar0 = self.lf_phistar0


        if np.asarray(z).ndim > 0:
            alpha = alpha0 * np.ones(len(z))
            sigma = sigma0 * np.ones(len(z))
            z = np.asarray(z).astype(float)
        else:
            alpha = alpha0
            sigma = sigma0
            z = float(z)

        lstar = np.piecewise(z, [z<2, z>=2],
                            [lambda z: lstar0 * (1+z)**3.55,
                            lambda z: lstar0 * (1+2)**3.55 * ((1+z)/(1+2))**1.62])

        phistar = np.piecewise(z, [z<1.1, z>=1.1],
                            [lambda z: phistar0 * (1+z)**-0.57,
                                lambda z: phistar0 * (1+1.1)**-0.57 * ((1+z)/(1+1.1))**-3.92])
        
        return phistar, lstar, alpha, sigma

    def lir(self, haloprop, redshift):
        """Compute IR luminosity given haloprop (either mass or vmax) and redshift"""
        if self.init_lir_function:
            return self.interpolator((haloprop, redshift))
        else:
            raise ValueError("LIR function not initialized")
        
    def sfr(self, haloprop, redshift, lir_to_sfr=None):
        """Compute SFR given haloprop (either mass or vmax) and redshift"""
        if lir_to_sfr is None:
            lir_to_sfr = self.lir_to_sfr

        if self.init_lir_function:
            return self.interpolator((haloprop, redshift)) * lir_to_sfr
        else:
            raise ValueError("SFR function not initialized")
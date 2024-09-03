import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf

from simim._paths import _SimIMPaths
from simim.siminterface._sims import _checksim
from simim.galprops.log10normal import log10normal
from simim.galprops.galprops_am import am_dfcat, double_schechter_gen
from simim.siminterface import SimHandler

from astropy.cosmology import FlatLambdaCDM

# SMF parameters
# Kelvin et al. 2014, Moutard et al. 2016, Davidzon et al. 2017, 
# Garzian et al. 2015... Garzian uses Salpeter IMF, we rescale by 
# 0.61 to get Chambrier IMF. I've checked that the parameters for 
# Kelvin, Moutard, and Garzian match what's in the SIDES param file
# the Davidzon pars don't seem to match the SIDES code but I can't 
# tell where the disconnect is
cosmo_obs = FlatLambdaCDM(H0=70, Om0=.3)

pars_kelvin = {'mstar': 10**10.64, 'alpha1':-0.43, 'phistar1':4.18e-3, 'alpha2':-1.50, 'phistar2':0.74e-3, 'zmin':0.025, 'zmax':0.06}
pars_moutard1 = {'mstar': 10**10.83, 'alpha1':-0.95, 'phistar1':10**-2.63, 'alpha2':-1.82, 'phistar2':10**-4.01, 'zmin':0.2, 'zmax':0.5}
pars_moutard2 = {'mstar': 10**10.76, 'alpha1':-0.57, 'phistar1':10**-2.66, 'alpha2':-1.49, 'phistar2':10**-3.24, 'zmin':0.5, 'zmax':0.8}
pars_moutard3 = {'mstar': 10**10.68, 'alpha1':-0.33, 'phistar1':10**-2.57, 'alpha2':-1.49, 'phistar2':10**-3.24, 'zmin':0.8, 'zmax':1.1}
pars_moutard4 = {'mstar': 10**10.66, 'alpha1': 0.19, 'phistar1':10**-2.88, 'alpha2':-1.49, 'phistar2':10**-3.24, 'zmin':1.1, 'zmax':1.5}
pars_davidzon5 = {'mstar': 10**10.51, 'alpha2':-1.28, 'phistar2':0.969e-3, 'alpha1':0.82, 'phistar1':0.64e-3, 'zmin':1.5, 'zmax':2.0}
pars_davidzon6 = {'mstar': 10**10.60, 'alpha2':-1.57, 'phistar2':0.295e-3, 'alpha1':0.07, 'phistar1':0.45e-3, 'zmin':2.0, 'zmax':2.5}
pars_davidzon7 = {'mstar': 10**10.59, 'alpha2':-1.67, 'phistar2':0.228e-3, 'alpha1':-0.08, 'phistar1':0.21e-3, 'zmin':2.5, 'zmax':3.0}
pars_davidzon8 = {'mstar': 10**10.83, 'alpha2':-1.76, 'phistar2':0.090e-3, 'alpha1':0, 'phistar1':0, 'zmin':3.0, 'zmax':3.5}
pars_garzian1 = {'mstar':0.61 * 10**10.96, 'alpha2':-1.63, 'phistar2':10**-3.94, 'alpha1':0, 'phistar1':0, 'zmin':3.5, 'zmax':4.5}
pars_garzian2 = {'mstar':0.61 * 10**10.78, 'alpha2':-1.63, 'phistar2':10**-4.18, 'alpha1':0, 'phistar1':0, 'zmin':4.5, 'zmax':5.5}
pars_garzian3 = {'mstar':0.61 * 10**10.49, 'alpha2':-1.55, 'phistar2':10**-4.16, 'alpha1':0, 'phistar1':0, 'zmin':5.5, 'zmax':6.5}
pars_garzian4 = {'mstar':0.61 * 10**10.69, 'alpha2':-1.88, 'phistar2':10**-5.24, 'alpha1':0, 'phistar1':0, 'zmin':6.5, 'zmax':7.5}

bins_use = [pars_kelvin, pars_moutard1, pars_moutard2, pars_moutard3, pars_moutard4, pars_davidzon5, pars_davidzon6, pars_davidzon7, pars_davidzon8, pars_garzian1, pars_garzian2, pars_garzian3, pars_garzian4]

# Extra Davidzon results not used in SIDES
pars_davidzon1 = {'mstar': 10**10.78, 'alpha1':-1.38, 'phistar1':1.187e-3, 'alpha2':-0.43, 'phistar2':1.92e-3, 'zmin':0.2, 'zmax':0.5}
pars_davidzon2 = {'mstar': 10**10.77, 'alpha1':-1.36, 'phistar1':1.070e-3, 'alpha2':0.03, 'phistar2':1.68e-3, 'zmin':0.5, 'zmax':0.8}
pars_davidzon3 = {'mstar': 10**10.56, 'alpha1':-1.31, 'phistar1':1.428e-3, 'alpha2':0.51, 'phistar2':2.19e-3, 'zmin':0.8, 'zmax':1.1}
pars_davidzon4 = {'mstar': 10**10.62, 'alpha1':-1.28, 'phistar1':1.069e-3, 'alpha2':0.29, 'phistar2':1.21e-3, 'zmin':1.1, 'zmax':1.5}
pars_davidzon9 = {'mstar': 10**11.10, 'alpha2':-1.98, 'phistar2':0.016e-3, 'alpha1':0, 'phistar1':0, 'zmin':3.5, 'zmax':4.5}
pars_davidzon10 = {'mstar': 10**11.30, 'alpha2':-2.11, 'phistar2':0.003e-3, 'alpha1':0, 'phistar1':0, 'zmin':4.5, 'zmax':5.5}

# Unclear if davidzon bin 9 or garzian bin 1 should be used - using garzian
zvals = np.array([(p['zmin']+p['zmax'])/2 for p in bins_use])
a1vals = np.array([p['alpha1'] for p in bins_use])
a2vals = np.array([p['alpha2'] for p in bins_use])
p1vals = np.array([p['phistar1'] for p in bins_use])
p2vals = np.array([p['phistar2'] for p in bins_use])
msvals = np.array([p['mstar'] for p in bins_use])

class bethermin17_base():
    """Code to implement the Bethermin et al. 2017 (SIDES) prescription for SFR
    and stellar mass. Note that this implementation does not include scatter
    when performing abundance matching, this will produce result in a different
    relation between halo and galaxy properties, but the same stellar mass
    function and star formation rate distribution function.
    
    When initializing a simulation must be provided as the base for abundance
    matching. This should match the underlying simulation being used for model
    construction (or at least have a similar cosmology).
    
    Abundance matching results are cached for later use.
    """

    def __init__(self,
                 sim,
                 sm_comp=False,sm_remake=False,sm_haloprop='auto',sm_halomassmin=1e10,
                 sffrac_fq00=0.1017, sffrac_gamma=-1.039, sffrac_logmt0=10.53, sffrac_alpha1=0.2232, sffrac_alpha2=0.0913, sffrac_sigsf0=0.8488, sffrac_beta1=0.0418, sffrac_beta2=-0.0159,
                 sfms_m0=0.5, sfms_m1=0.36, sfms_a0=1.5, sfms_a1=0.3, sfms_a2=2.5, sfms_correztlowz=True,
                 sfms_sigma=0.3, sfms_msoffset=0.87, sfms_sboffset=5.3, 
                 sf_sflimit=1e4
                 ):
        """
        Parameters
        ----------
        sim : string
            Name of simulation - should correspond to a SimIM formatted halo
            catalog, used to perform the abundance matching.
        sm_comp : bool, default is False
            If set to True, will carry out abundance matching to determine the
            halo-stellar mass connection if the basic files cannot be loaded
        sm_haloprop : 'vmax', 'mass', or 'auto'
            Halo property to abundance match to stellar mass function - vmax for
            peak rotational velocity, mass for halo mass. auto will check if
            vmax is present in the catalog and use this, or use mass otherwise.
            Ignored if sm_comp==False
        sm_halomassmin : float
            Minimum halo mass to consider when abundance matching in units of
            Msun, defaults to 10^10 Msun. Ignored if sm_comp==False
        sm_remake : bool
            Rerun the abundance matching when initializing. Ignored if
            sm_comp==False
        sffrac_fq00, sffrac_gamma, sffrac_logmt0, sffrac_alpha1, sffrac_alpha2, sffrac_sigsf0, sffrac_beta1, sffrac_beta2 : float
            Parameters controling the fraction of galaxies considered star
            forming - Equation 2 of Bethermin 17. Defaults are the values from
            the paper
        sfms_m0, sfms_m1, sfms_a0, sfms_a1, sfms_a2 : float
            Parameters controlling the star forming main sequence
            parameterization - Equation 6 of Bethermin 17. Defaults are the
            values from the paper
        sfms_correztlowz : bool
            Toggle whether the main-sequence fudge factor is applied at z<1 (see
            Bethermin 17 Appendix B). Default is True, as in the Bethermin paper
        sfms_sigma, sfms_msoffset, sfms_sboffset, sf_sflimit : float
            Parameters controlling the assignment of SFRs, see Bethermin 17
            section 2.5. Defaults match the paper
        """

        paths = _SimIMPaths()

        _checksim(sim)
        self.sim = sim
        self.name = 'b17_'+sim

        if self.name not in paths.props.keys():
            paths._newproppath(self.name)

        self.path = paths.props[self.name]

        self.init_stellarmass_function = False
        self._load_stellarmass_function(comp=sm_comp, remake=sm_remake, haloprop=sm_haloprop, halomassmin=sm_halomassmin)

        self.sfr = self.b17_sfr

        # Params for star forming fraction
        self.sffrac_fq00=sffrac_fq00
        self.sffrac_gamma=sffrac_gamma
        self.sffrac_logmt0=sffrac_logmt0
        self.sffrac_alpha1=sffrac_alpha1
        self.sffrac_alpha2=sffrac_alpha2
        self.sffrac_sigsf0=sffrac_sigsf0
        self.sffrac_beta1=sffrac_beta1
        self.sffrac_beta2=sffrac_beta2

        # Params for the SFMS
        self.sfms_m0=sfms_m0
        self.sfms_m1=sfms_m1
        self.sfms_a0=sfms_a0
        self.sfms_a1=sfms_a1
        self.sfms_a2=sfms_a2
        self.sfms_correztlowz=sfms_correztlowz

        # Params for SFR assignment
        self.sfms_sigma=sfms_sigma
        self.sfms_msoffset=sfms_msoffset
        self.sfms_sboffset=sfms_sboffset
        self.sf_sflimit=sf_sflimit

    def _load_stellarmass_function(self, comp, remake, haloprop, halomassmin):
        """Load or compute halo mass-stelar mass function"""
    
        pathexists = os.path.exists(os.path.join(self.path,'stellarmass.npy'))
        if remake or (comp and not pathexists):
            self._make_stellarmass_function(haloprop=haloprop,halomassmin=halomassmin)
            pathexists = True
        
        if pathexists:
            # Load spline stuff
            halopropgrid = np.load(os.path.join(self.path,'mass_axis.npy'))
            zgrid = np.load(os.path.join(self.path,'redshift_axis.npy'))
            mstarsgrid = np.load(os.path.join(self.path,'stellarmass.npy'))

            self.interpolator = RegularGridInterpolator((halopropgrid,zgrid), mstarsgrid, method='linear', bounds_error=False)
            self.init_stellarmass_function = True
    
    def _make_stellarmass_function(self, haloprop, halomassmin):
        """Compute halo mass-stellar mass relation by abundance matching"""

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
        smgrid = np.zeros((len(zgrid),len(halopropgrid)))

        # i is counter, idx is snap number
        for i,idx in enumerate(handler.snap_meta['index']):
            print("\033[1m"+"Perfoming Abundance Match for Snapshot {}.  ".format(idx)+"\033[0m",end='\r')
            
            snap = handler.get_snap(idx)
            
            ms, a1, p1, a2, p2 = self.b17_smfpars(zgrid[i], handler.cosmo)
            smfpars = {'x0':ms,'phi1':p1,'alpha1':a1,'phi2':p2,'alpha2':a2,'xmin':1e5}
            smf = double_schechter_gen()
            smf_freeze = smf(**smfpars)

            smf_total = smf.lum_function_int(smfpars['xmin'], **smfpars) * snap.box_edge_no_h**3

            snap.set_property_range('mass',halomassmin,np.inf)
            snap_haloprop = snap.return_property(haloprop,use_all_inds=False)
            snap_total = len(snap_haloprop)

            if snap_total > smf_total:
                missing_pmass_low_p1 = 0
                missing_pmass_low_p2 = 1 - smf_total / snap_total
            else:
                missing_pmass_low_p1 = 1 - snap_total / smf_total
                missing_pmass_low_p2 = 0

            if len(snap_haloprop) > 0:
                mstarofhalo, _ = am_dfcat(snap_haloprop,smf_freeze,missing_pmass_low_p1=missing_pmass_low_p1,missing_pmass_low_p2=missing_pmass_low_p2)
                smgrid[i] = mstarofhalo(halopropgrid)
        print()

        np.save(os.path.join(self.path,'mass_axis.npy'), halopropgrid)
        np.save(os.path.join(self.path,'redshift_axis.npy'), zgrid)
        np.save(os.path.join(self.path,'stellarmass.npy'), smgrid.T)

    def b17_smfpars(self, z, cosmo):
        """Update parameters to a chosen cosmology"""

        dlold = cosmo_obs.luminosity_distance(zvals).value
        dlnew = cosmo.luminosity_distance(zvals).value
        vold = cosmo_obs.differential_comoving_volume(zvals).value
        vnew = cosmo.differential_comoving_volume(zvals).value

        a1 = np.interp(1+z, 1+zvals, a1vals)
        a2 = np.interp(1+z, 1+zvals, a2vals)
        p1 = np.interp(1+z, 1+zvals, p1vals * (vnew/vold)**-1)
        p2 = 10**np.interp(1+z, 1+zvals, np.log10(p2vals * (vnew/vold)**-1))
        ms = 10**np.interp(1+z, 1+zvals, np.log10(msvals * (dlnew/dlold)**2))

        return ms, a1, p1, a2, p2
    
    def b17_sffrac(self, m_stars, z):
        """Compute star forming fraction given stellar mass and redshift"""

        fq0 = self.sffrac_fq00 * (1+z)**self.sffrac_gamma
        logmt = self.sffrac_logmt0 + self.sffrac_alpha1*z + self.sffrac_alpha2*z**2
        sigsf = self.sffrac_sigsf0 + self.sffrac_beta1*z + self.sffrac_beta2*z**2
        fsf = (1-fq0) * 0.5 * (1-erf((np.log10(m_stars)-logmt)/sigsf))
        return fsf

    def b17_sfms(self, m_stars, z):
        """Compute main sequence given stellar mass and redshift"""

        term2 = np.log10(m_stars/1e9) - self.sfms_m1 - self.sfms_a2*np.log10(1+z)
        term2[term2<0] = 0
        logms = np.log10(m_stars/1e9) - self.sfms_m0 + self.sfms_a0*np.log10(1+z) - self.sfms_a1*term2**2

        if self.sfms_correztlowz:
            if np.asarray(z).ndim > 0:
                logms[z<0.5] -= 0.1*(0.5-z[z<0.5])/(0.5-0.22)
            elif z<0.5:
                logms -= 0.1*(0.5-z)/(0.5-0.22)
        
        return 10**logms

    def b17_sbfrac(self, m_stars, z):
        """Compute starburts fraction given stellar mass and redshift"""

        fsb = 0.015 + 0.015*z
        if np.asarray(z).ndim > 0:
            fsb[z>1] = 0.03
        elif z>1:
            fsb = 0.03
        return fsb
    
    def b17_sfr(self, m_stars, z, rng=np.random.default_rng()):
        """Compute SFR given stellar mass and redshift"""

        sfr = self.sfms_msoffset * self.b17_sfms(m_stars, z)
        sfr = log10normal(sfr, self.sfms_sigma, preserve_linear_mean=False, rng=rng)

        # Assign starbursts and quenched galaxies - these two probabilities
        # can be treated independently
        sbfrac = self.b17_sbfrac(m_stars, z)
        sfr[rng.uniform(0,1,len(sfr)) < sbfrac] *= self.sfms_sboffset / self.sfms_msoffset

        sffrac = self.b17_sffrac(m_stars, z)
        sfr[rng.uniform(0,1,len(sfr)) > sffrac] = 0

        # No SF above a limit
        sfr[sfr>self.sf_sflimit] = self.sf_sflimit

        return sfr

    def stellarmass(self, haloprop, redshift):
        """Compute stellar mass given haloprop (either mass or vmax) and redshift"""
        if self.init_stellarmass_function:
            return self.interpolator((haloprop, redshift))
        else:
            raise ValueError("stellar mass not initialized")
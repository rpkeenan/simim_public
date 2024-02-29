import os
import warnings
from importlib_resources import files

import numpy as np
from scipy.interpolate import RectBivariateSpline

from simim._paths import _SimIMPaths
from simim.galprops.log10normal import log10normal
from simim._pltsetup import *

class behroozi13_base():
    """Class implementing the best fitting SFR(Mhalo) relation
    from Behroozi et al. 2013"""

    def __init__(self,warn=False): # warn toggles showing a warning about masses or redshifts outside the default range
        """Initialize class - if running for the first time, this will set up
        and cache some files for interpolating within the value grid provided
        in the public data release from https://www.peterbehroozi.com/uploads/6/5/4/8/6548418/sfh_z0_z8.tar.gz
        
        Parameters
        ----------
        warn : bool
            Toggles whether a warning will be shown for masses or redshifts 
            parameter range covered in the underlying data outside the default 
            range
        """

        self.warn = warn

        # Check paths
        paths = _SimIMPaths()
        if paths.root is not None:
            if 'behroozi13' not in paths.props:
                paths._newproppath('behroozi13')
            self.path = paths.props['behroozi13']

            # Make required files if needed
            if not os.path.exists(os.path.join(self.path,'sfr.npy')):
                self._make_spline_files()

            # Load spline stuff
            self.grid_mass = np.load(os.path.join(self.path,'mass_axis.npy'))
            self.grid_redshift = np.load(os.path.join(self.path,'redshift_axis.npy'))
            grid_sfr = np.load(os.path.join(self.path,'sfr.npy')).T
            grid_stellar = np.load(os.path.join(self.path,'stellarmass.npy')).T

            self.sfr_function = RectBivariateSpline(self.grid_redshift,self.grid_mass,grid_sfr,kx=1,ky=1)
            self.stellarmass_function = RectBivariateSpline(self.grid_redshift,self.grid_mass,grid_stellar,kx=1,ky=1)

        else:
            self.path = None

    def _make_spline_files(self):
        """Generate the needed files from the distribution data from
        P. Behroozi's website.
        """

        simim_path = files('simim')
        path = simim_path.joinpath('resources','behroozi13_release.dat')
        data = np.loadtxt(path,
                          dtype=[('redshift','f'),('mass','f'),('sfr','f'),('stellarmass','f')])

        # Put everything in the right units (1+z --> z),
        # log10(mass) --> mass
        data['redshift'] -= 1
        data['mass'] = np.power(10,data['mass'])
        data['stellarmass'] = np.power(10,data['stellarmass'])

        # Get unique values of halo mass and redshift
        redshift = np.unique(data['redshift'])
        mass = np.unique(data['mass'])

        # Make fields to put everything in
        sfr = np.empty((len(mass),len(redshift)))
        stellarmass = np.empty((len(mass),len(redshift)))

        # And put things where they belong
        for i in range(len(mass)):
            for j in range(len(redshift)):
                inds1 = np.where(data['mass'] == mass[i])
                inds2 = np.where(data['redshift'] == redshift[j])
                ind = np.intersect1d(inds1,inds2)

                sfr[i,j] = data['sfr'][ind]
                stellarmass[i,j] = data['stellarmass'][ind]

        # Figure out where we stop getting values (limits)
        for j in range(len(redshift)):
            sfr_at_limit = sfr[:,j][~np.isclose(sfr[:,j],-1000.0)][-1]
            sfr[:,j][np.isclose(sfr[:,j],-1000.0)] = sfr_at_limit

        sfr = np.power(10,sfr)

        np.save(os.path.join(self.path,'sfr.npy'),sfr)
        np.save(os.path.join(self.path,'stellarmass.npy'),stellarmass)
        np.save(os.path.join(self.path,'mass_axis.npy'),mass)
        np.save(os.path.join(self.path,'redshift_axis.npy'),redshift)

    def plot_grid(self,prop='sfr'):
        """Plot the grid.

        Parameters
        ----------
        prop : 'sfr' or 'stellar'
            The property to show
        """

        mass = np.logspace(np.log10(np.amin(self.grid_mass)),np.log10(np.amax(self.grid_mass)),1000)
        redshift = np.linspace(np.amin(self.grid_redshift),np.amax(self.grid_redshift),1000)

        x,y = np.meshgrid(redshift,mass)

        if prop == 'sfr':
            val = self.sfr_function.ev(x.flatten(),y.flatten()).reshape((1000,1000))
        elif prop == 'stellar':
            val = self.stellarmass_function.ev(x.flatten(),y.flatten()).reshape((1000,1000))
        else:
            raise ValueError("val not recognized")

        fig,ax = plt.subplots()
        ax.set(xlabel='Redshift',ylabel=r'Halo Mass [M$_\odot$]',yscale='log')

        map = ax.pcolor(x,y,val)
        cbar = fig.colorbar(map)
        if prop == 'sfr':
            cbar.set_label(r'SFR [M$_\odot$/yr]')
        if prop == 'stellar':
            cbar.set_label(r'M$_*$ [M$_\odot$]')

        plt.show()

    def sfr(self, redshift, mass, scatter=True, sigma_scatter=0.3, rng=np.random.default_rng()):
        """Function to assign star formation rates based on the model
        from Behroozi et al. 2013.

        Parameters
        ----------
        redshift : float or array
            The redshift(s) of the halo(s)
        mass : float or array
            The mass(es) of the halo(s) in Msun
        scatter : bool
            If True, a lognormal scatter will be added around the mean SFR
            assigned. Default is True
        sigma_scatter : float
            The scatter, in dex, to add around the mean. Default is 0.3
        rng : optional, numpy.random.Generator object
            Can be used to specify an rng, useful if you want to control the seed

        Returns
        -------
        sfr : float or array
            The assigned SFRs in Msun/yr
        """

        if self.path is None:
            return ValueError("model setup not complete - make sure setupsimim has been run")

        if np.any(redshift > np.amax(self.grid_redshift)) and self.warn:
            warnings.warn('redshift exceeds maximum in Behroozi model range')
        if np.any(mass > np.amax(self.grid_mass)) and self.warn:
            warnings.warn('mass exceeds maximum in Behroozi model range')

        # Get SFRs and return them
        sfrval = self.sfr_function.ev(redshift,mass)

        if scatter:
            sfrval = log10normal(sfrval, sigma_scatter, preserve_linear_mean=True)

        return sfrval

    def stellarmass(self, redshift, mass, scatter=True, sigma_scatter=0.3, rng=np.random.default_rng()):
        """Function to assign stellar masses based on the model
        from Behroozi et al. 2013.

        Parameters
        ----------
        redshift : float or array
            The redshift(s) of the halo(s)
        mass : float or array
            The mass(es) of the halo(s) in Msun
        scatter : bool
            If True, a lognormal scatter will be added around the mean SFR
            assigned. Default is True
        sigma_scatter : float
            The scatter, in dex, to add around the mean. Default is 0.3
        rng : optional, numpy.random.Generator object
            Can be used to specify an rng, useful if you want to control the seed

        Returns
        -------
        stellar : float or array
            The assigned stellar mass in Msun
        """

        if self.path is None:
            return ValueError("model setup not complete - make sure setupsimim has been run")

        if np.any(redshift > np.amax(self.grid_redshift)) and self.warn:
            warnings.warn('redshift exceeds maximum in Behroozi model range')
        if np.any(mass > np.amax(self.grid_mass)) and self.warn:
            warnings.warn('mass exceeds maximum in Behroozi model range')

        # Get masses and return them
        stellarmassval = self.stellarmass_function.ev(redshift,mass)

        if scatter:
            stellarmassval = log10normal(stellarmassval, sigma_scatter, preserve_linear_mean=True)

        return stellarmassval

# Aliases to make the required functions easy to access
base = behroozi13_base()

def sfr(redshift, mass, scatter=True, sigma_scatter=0.3, rng=np.random.default_rng()):
        """Function to assign star formation rates based on the model
        from Behroozi et al. 2013.

        Parameters
        ----------
        redshift : float or array
            The redshift(s) of the halo(s)
        mass : float or array
            The mass(es) of the halo(s) in Msun
        scatter : bool
            If True, a lognormal scatter will be added around the mean SFR
            assigned. Default is True
        sigma_scatter : float
            The scatter, in dex, to add around the mean. Default is 0.3
        rng : optional, numpy.random.Generator object
            Can be used to specify an rng, useful if you want to control the seed

        Returns
        -------
        sfr : float or array
            The assigned SFRs in Msun/yr
        """
        
        return base.sfr(redshift=redshift, mass=mass, scatter=scatter, sigma_scatter=sigma_scatter, rng=rng)
def stellarmass(redshift, mass, scatter=True, sigma_scatter=0.3, rng=np.random.default_rng()):
        """Function to assign stellar masses based on the model
        from Behroozi et al. 2013.

        Parameters
        ----------
        redshift : float or array
            The redshift(s) of the halo(s)
        mass : float or array
            The mass(es) of the halo(s) in Msun
        scatter : bool
            If True, a lognormal scatter will be added around the mean SFR
            assigned. Default is True
        sigma_scatter : float
            The scatter, in dex, to add around the mean. Default is 0.3
        rng : optional, numpy.random.Generator object
            Can be used to specify an rng, useful if you want to control the seed

        Returns
        -------
        stellar : float or array
            The assigned stellar mass in Msun
        """
        return base.stellarmass(redshift=redshift, mass=mass, scatter=scatter, sigma_scatter=sigma_scatter, rng=rng)

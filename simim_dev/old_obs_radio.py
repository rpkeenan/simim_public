'''Code for simulating radio observations

1. radio_map - class for describing a single dish radio map
2. interferometer_obs - class for describing an interferometric observation
3. radio_analysis - class for conducting intensity mapping analysis

Updated 7.2.19
Version 6.0

Version 5.0 --- create objects to hold radio observation simulations
Version 5.1 --- create a class to handle radio observation data analysis
Version 6.0 --- version 6.0 integrates observation functions
'''

from copy import deepcopy
import types

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]
import matplotlib.patches as mpatches

from numpy import abs, amax, amin, arange, array, concatenate, conj, copy, cos, diag_indices, load, log10, meshgrid, ndarray, ones, pi, prod, real, rint, save, savetxt, sin, sort, sqrt, std, sum, triu_indices, unique, where, zeros
from numpy import mean, inf, std, nan
from numpy.random import normal, rand, randint, uniform
from numpy.fft import fftn, fftshift, ifftshift, ifftn
import numpy.lib.recfunctions

from .obs_beam import mk_beam, convolve_beam
from .obs_galaxies import galaxy_catalog, completeness_100
from .ps_make import mk_grid, ft_grid, av_grid, mk_mask
from .obs_noise import noise_from_T_N, visibility_noise_from_T_N
from .lc_props import d_to_z, L_to_I, L_to_T, nu_to_z, X, Y, z_to_nu
from .paths import cosmo

c = 2.9979e8
k = 1.3806e-23

################################################################################
########################## SECTION 0: Helper Routines ##########################
################################################################################

'''KGRID:
Object to contain kgrids and suplemental information
'''

class kgrid(object):
    def __init__(self, grid, units, desctription=''):
        self.grid = grid
        self.units = units
        self.desctription = desctription



'''POWER_SPECTRUM:
Object to contain power spectra and suplemental information
'''

class power_spectrum(object):
    """(name, spectrum, axis, units, counts, description)
    Power spectrum container"""
    def __init__(self, name, spectrum, axis, units, counts=nan, description=''):
        self.name = name
        self.spectrum = spectrum
        self.axis = axis
        self.units = units
        self.description = description
        self.counts = counts

################################################################################
######################### SECTION 1: Radio Observations ########################
################################################################################

'''RADIO_MAP:
A class to contain all of the pointing and telescope data for a radio observation.

Required Initialization Arguments:
f_min, f_max - the minimum, maximum frequencies of the observation
df - the size of frequency channels (Hz)
ra, dec - the central RA, Declination of the map (rad)
ra_size, dec_size - the linear sizes of the map in the RA and Declination
 directions (rad)
dtheta - the size of a pixel of the map (rad)
fwhm - the full width at half maximum of the telescope beam at frequency fwhm_nu
 (rad)
fwhm_nu - the frequency for which the FWHM is specified (Hz)
Tr - the noise temperature of the observation (K), or a function for calculating
 the noise as a function of position in the data cube (form f(ra, dec, nu))
t_obs - the total observing time (s)
n_det - the number of detectors used
units - the units used in the map - 'K' or 'Jy sr-1' ('K' is default)

Object properties:
All input arguments are stored.

In addition the following data is useful for describing the data cube:
voxel - an array [dtheta, dtheta, df] describing the shape of voxels
center - an array [ra, dec, (f_max+f_min)/2] describing the center of the data
 cube
dims - an array [ra_size, dec_size, f_max-f_min] describing the extent of the
 data cube
map_shape - the dimensions of the data cube
ax - the axes of the data cube (rad, rad, Hz)

The following data is useful for describing the beam and data weights:
beam - array matched in shape to the data cube, containing an image of the beam
window - the fourier transform of the beam into u, v, nu space
mask - set of cells to be included in analysis (all cells for a single dish map)
weights - set of cell weights for analysis (set to 1 for current implementation)

The following data is useful in accounting for the noise:
t_voxel - the time spent observing each voxel
vox_noise - the rms noise in each voxel
noise_debias_map - an array of the same shape as the data cube, containing the
 rms noise of each voxel, deconvolved with the primary beam
noise_map - an array of the same shape as the data cube, containing a
 randomized realization of the noise in each voxel

Object methods:
redraw_noise - generates a new realization of noise_map
'''

class radio_map(object):
    def __init__(self, f_min, f_max, df, ra, dec, ra_size, dec_size, dtheta, fwhm, fwhm_nu,
                 Tr, t_obs, n_det,
                 units='K'):

        # Information about type of observation
        self.type = 'radio_map'

        # Pointing and frequency channel parameters
        self.f_min = f_min
        self.f_max = f_max
        self.f_mid = (f_min+f_max)/2
        self.df = df
        self.ra = ra
        self.dec = dec
        self.ra_size = ra_size
        self.dec_size = dec_size
        self.dtheta = dtheta

        # Beam parameters
        self.fwhm = fwhm
        self.fwhm_nu = fwhm_nu

        # Check units input
        if units in ['K', 'T', 'Temp', 'Temperature', 'temp', 'temperature']:
            self.units = 'K'
        elif units in ['Jy', 'Jy per sr', 'Jy per str', 'Jy sr-1', 'Jy str-1', 'Janskys']:
            self.units = 'Jy sr-1'
        else:
            raise InputError('units', 'units not recognized')

        # Set up map
        self.voxel = array([dtheta, dtheta, df])
        self.center = [ra,ra,(f_max+f_min)/2]
        self.dims = [ra_size, dec_size, f_max-f_min]

        # Grid nothing to get the shape of the map and its axes
        empty_map, self.ax = mk_grid(zeros((1,3)), fit_dims=False,
                                          voxel_length=self.voxel, dims=self.dims, center=self.center)
        self.map_shape = empty_map.shape

        # Make a beam
        self.beam = mk_beam(self.map_shape, self.dtheta, self.fwhm, nu=self.ax[2].flatten(), nu_FWHM=self.fwhm_nu, mode='3D Beam')
        self.window = fftshift(fftn(ifftshift(self.beam, axes=(0,1)), axes=(0,1)),axes=(0,1))

        # Mask includes everything and weights are equal - revisit later
        self.mask = ones(self.map_shape)
        self.weights = self.mask

        # Make sure the noise has the right form - if it's given as a function
        # evaluate it
        if type(Tr) == types.FunctionType:
            self.Trfunc = Tr
            xx,yy,zz = meshgrid(self.ax[0], self.ax[1], self.ax[2], indexing='ij')
            self.Tr = Tr(xx,yy,zz)
        else:
            self.Tr = ones(self.map_shape) * Tr

        # Compute the observing time in each pixel (and by extension voxel) -
        # t_pixel = t_total*n_detectors / n_pixels
        self.t_obs = t_obs
        self.n_det = n_det
        self.n_point = (ra_size/dtheta) * (dec_size/dtheta) / n_det #number of cells / number of detectors
        self.t_voxel = t_obs / self.n_point

        # Realize noise
        if self.units == 'K':
            self.vox_noise = noise_from_T_N(self.Tr, self.t_voxel, self.df) * 1e6
        elif self.units == 'Jy sr-1':
            self.vox_noise = noise_from_T_N(self.Tr, self.t_voxel, self.df) * 2*k*self.ax[2].flatten()**2/c**2 * 1e26

        self.noise_map = normal(loc=0, scale=self.vox_noise, size=self.map_shape)
        self.noise_debias_map = ones(self.map_shape) * self.vox_noise
        self.noise_debias_map = abs(self.noise_debias_map / self.window)

    def noise_redraw(self):
        self.noise_map = normal(loc=0, scale=self.vox_noise, size=self.map_shape)



'''INTERFEROMETER_OBS:
A class to contain all of the pointing and telescope data for an interferometric
observation.

Note that this class is meant to match the key properties of RADIO_MAP so that
the two can be treated similarly in RADIO_ANALYSIS. Instead of specifying the
ra_size and dec_size and dtheta however, these are set internally by the uv
plane coverage.

Required Initialization Arguments:
f_min, f_max - the minimum, maximum frequencies of the observation
df - the size of frequency channels (Hz)
ra, dec - the central RA, Declination of the map (rad)
upos, vpos - the locations in the u*lamda, v*lamda plane sampled by the
 observation (m)
fwhm - the full width at half maximum of the telescope beam at frequency fwhm_nu
 (rad)
fwhm_nu - the frequency for which the FWHM is specified (Hz)
Tr - the noise temperature of the observation (K), or a function for calculating
 the noise as a function of position in the data cube (form f(ra, dec, nu))
t_obs - the total observing time (s)
units - the units used in the map - 'K' or 'Jy sr-1' ('K' is default)

Optional Arguments:
constant_uv_with_nu (= False) - determines whether to treat the uv coverage as
 constant with changing nu across the data cube
map_descale (= 1) - reduces the size of the uv space, at the expense of creating
 artifacts in maps due to aliasing
samples_across_res (= 3) - sets the number of samples across a resolution element
 to use

Object properties:
All input arguments are stored and additionally, an ra_size, dec_size, and
dtheta are computed (in rad).

In addition the following data is useful for describing the data cube:
voxel - an array [dtheta, dtheta, df] describing the shape of voxels
center - an array [ra, dec, (f_max+f_min)/2] describing the center of the data
 cube
dims - an array [ra_size, dec_size, f_max-f_min] describing the extent of the
 data cube (or more properly of its inverse fourier transform)
map_shape - the dimensions of the data cube
ax - the axes of the data cube in real space (rad, rad, Hz)

The following data is useful for describing the beam and data weights:
beam - array matched in shape to the data cube, containing an image of the beam
mask - set of cells to be included in analysis
weights - set of cell weights for analysis (based on number of samples counted)

The following data is useful in accounting for the noise:
vox_noise - the rms noise in each uvnu cell
noise_debias_map - an array of the same shape as the data cube, containing the
 rms noise of each uvnu cell
noise_map - an array of the same shape as the data cube, containing a
 randomized realization of the noise in each visibility

Object methods:
redraw_noise - generates a new realization of noise_map
image_beam - generates images of the primary beam and dirty beam
image_noise - generates an image of the realized noise
'''

class interferometer_obs(object):
    def __init__(self, f_min, f_max, df, ra, dec, upos, vpos, fwhm, fwhm_nu,
                 Tr, t_obs, units='K',
                 constant_uv_with_nu=False, map_descale=1, samples_across_res=3):

        # Information about type of observation
        self.type = 'interferometer_obs'

        # Pointing and frequency channel parameters
        self.f_min = f_min
        self.f_max = f_max
        self.f_mid = (f_min+f_max)/2
        self.df = df
        self.ra = ra
        self.dec = dec
        self.upos = array(upos)
        self.vpos = array(vpos)
        self.samples_across_res = samples_across_res
        self.map_descale = map_descale

        nu = arange(f_min+df/2, f_max+df/2, df)

        # Check units input
        if units in ['K', 'T', 'Temp', 'Temperature', 'temp', 'temperature']:
            self.units = 'K'
        elif units in ['Jy', 'Jy per sr', 'Jy per str', 'Jy str-1', 'Janskys']:
            self.units = 'Jy sr-1'
        else:
            raise InputError('units', 'units not recognized')

        # Compute beam pixel size
        uvpos_min = amin(abs(concatenate((upos[upos!=0], vpos[vpos!=0]))))
        uvpos_max = amax(abs(concatenate((upos[upos!=0], vpos[vpos!=0]))))

        uv_min = uvpos_min / (c/f_min)
        uv_max = uvpos_max / (c/f_max)

        self.dtheta = 1 / (samples_across_res*uv_max)
        n = int(round((1/uv_min) / self.dtheta / self.map_descale))
        if n % 2 == 0:
            n += 1
        self.n_across = n

        self.ra_size = n*self.dtheta
        self.dec_size = self.ra_size

        # Set up map
        self.voxel = array([self.dtheta, self.dtheta, df])
        self.center = [ra,ra,self.f_mid]
        self.dims = [self.ra_size, self.dec_size, f_max-f_min]

        # Grid nothing to get map shape and an axes
        empty_map, self.ax = mk_grid(zeros((1,3)), fit_dims=False,
                                          voxel_length=self.voxel, dims=self.dims, center=self.center)
        self.map_shape = empty_map.shape

        # Beam parameters
        self.fwhm = fwhm
        self.fwhm_nu = fwhm_nu

        # Make a primary beam
        if constant_uv_with_nu == False:
            self.beam = mk_beam(self.map_shape, self.dtheta, self.fwhm, nu, self.fwhm_nu, mode='3D Beam')
        else: # if we're assuming constant uv coverage we might as well also assume the beam doesn't evolve in other ways
            self.beam = mk_beam(self.map_shape, self.dtheta, self.fwhm*self.fwhm_nu/self.f_mid, mode='2D Gauss')
            self.beam = self.beam * ones((1,1,self.map_shape[2]))

        # Set up UV mask
        n_chan = len(nu)
        l_uv = (len(self.upos))
        uv = zeros((l_uv * n_chan, 3))
        for i in range(n_chan):
            # If we have fixed a constant nu then use f_mid and we'll have constant
            # uv columns of filled values along nu/eta. Otherwise allow uv to varry
            # with nu
            if constant_uv_with_nu == False:
                uv[l_uv*i:l_uv*(i+1),0] = self.upos / (c/nu[i])
                uv[l_uv*i:l_uv*(i+1),1] = self.vpos / (c/nu[i])
                uv[l_uv*i:l_uv*(i+1),2] = nu[i]
            else:
                uv[l_uv*i:l_uv*(i+1),0] = self.upos / (c/self.f_mid)
                uv[l_uv*i:l_uv*(i+1),1] = self.vpos / (c/self.f_mid)
                uv[l_uv*i:l_uv*(i+1),2] = nu[i]

        # Set the weights of the data points - grid the uv coverage and set
        # weiths to be the number of counts in each cell, set mask to denote
        # which cells have data
        self.weights, self.uv_axes = mk_grid(uv, fit_dims=False,
                                            voxel_length=[1/self.ra_size, 1/self.dec_size, df],
                                            dims = [1/self.dtheta, 1/self.dtheta, f_max-f_min],
                                            center = [0, 0, self.f_mid])
        self.mask = zeros(self.weights.shape,dtype='bool')
        for i in range(len(self.mask[0,0,:])):
            self.mask[:,:,i][sum(abs(self.weights),axis=2) > 1e-6] = True

        # Make sure the noise has the right form - if it's given as a function
        # evaluate it
        if type(Tr) == types.FunctionType:
            self.Trfunc = Tr
            xx,yy,zz = meshgrid(self.uv_axes[0], self.uv_axes[1], self.uv_axes[2], indexing='ij')
            self.Tr = Tr(xx,yy,zz)
        else:
            self.Tr = ones(self.map_shape) * Tr

        self.t_obs = t_obs
        self.n_det = 1

        # Realize noise
        self.t_voxel = t_obs
        if self.units == 'K':
            self.vox_noise = visibility_noise_from_T_N(self.Tr, self.t_voxel, self.df, self.fwhm**2) * 1e6
        elif self.units == 'Jy sr-1':
            self.vox_noise = visibility_noise_from_T_N(self.Tr, self.t_voxel, self.df, self.fwhm**2) * 2*k*nu**2/c**2 * 1e26

        # draw noise and then make sure it is hermitian ((u,v) and (-u,-v) are not independent data)
        self.noise_map = normal(loc=0, scale=self.vox_noise/sqrt(2), size=self.map_shape) + 1j*normal(loc=0, scale=self.vox_noise/sqrt(2), size=self.map_shape)
        self.noise_map[triu_indices(self.n_across,1)] = conj(self.noise_map[::-1,::-1][triu_indices(self.n_across, 1)])
        self.noise_map[diag_indices(int((self.n_across-1)/2))] = conj(self.noise_map[::-1,::-1][diag_indices(int((self.n_across-1)/2))])
        # reduce noise by number of measurements in cell
        self.noise_map[self.weights > .1] /= sqrt(self.weights[self.weights > .1])

        self.noise_debias_map = zeros(self.map_shape)
        self.noise_debias_map[self.weights > .1] = self.vox_noise[self.weights > .1] / sqrt(self.weights[self.weights > .1])

    def noise_redraw(self):
        # draw noise and then make sure it is hermitian ((u,v) and (-u,-v) are not independent data)
        self.noise_map = normal(loc=0, scale=self.vox_noise/sqrt(2), size=self.map_shape) + 1j*normal(loc=0, scale=self.vox_noise/sqrt(2), size=self.map_shape)
        self.noise_map[triu_indices(self.n_across,1)] = conj(self.noise_map[::-1,::-1][triu_indices(self.n_across, 1)])
        self.noise_map[diag_indices(int((self.n_across-1)/2))] = conj(self.noise_map[::-1,::-1][diag_indices(int((self.n_across-1)/2))])
        # reduce noise by number of measurements in cell
        self.noise_map[self.weights > .1] /= sqrt(self.weights[self.weights > .1])

    def image_beam(self):
        image = fftshift(ifftn(ifftshift(self.weights), axes=(0,1))).real
        for i in range(len(image[0,0,:])):
            image[:,:,i] /= sum(image[:,:,i].flatten())
        for i in range(len(image[0,0,:])):
            fig = plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(121)
            ax1.set(title='Dirty Beam')
            ax1.imshow(image[:,:,i], origin='lower',
                       extent=(-180*60/pi*self.ra_size/2, 180*60/pi*self.ra_size/2, -180*60/pi*self.dec_size/2, 180*60/pi*self.dec_size/2))
            ax2 = fig.add_subplot(122)
            ax2.set(title='Primary Beam')
            ax2.imshow(self.beam[:,:,i], origin='lower',
                       extent=(-180*60/pi*self.ra_size/2, 180*60/pi*self.ra_size/2, -180*60/pi*self.dec_size/2, 180*60/pi*self.dec_size/2))
            plt.show()

    def image_noise(self):
        conv = copy(self.noise_map)
        conv[self.weights<.1] = 0
        image = fftshift(ifftn(ifftshift(conv), axes=(0,1))).real
        for i in range(len(image[0,0,:])):
            plt.imshow(image[:,:,i], origin='lower')
            plt.show()



################################################################################
########################### SECTION 2: Radio Analysis ##########################
################################################################################

'''RADIO_ANALYSIS:
A class for performing a variety of analysis tasks for single dish or
interferometric intensity mapping data

Main modes:
auto power - the auto spectra are automatically computed on initialization
cross power - cross power can be computed by matching to a galaxy catalog
 using the MATCHED_CATALOG method
stacking analysis - galaxy stacking can be performed using the GALAXY_STACK
 method (currently only availablf for single dish mode)

Required Initialization Arguments:
lc - a LIGHT_CONE object from which to draw galaxies
map - a RADIO_MAP or INTERFEROMETER_OBS object from which to draw observation
 parameters
f_target - the rest frequency of the line targeted in the analysis (in Hz)

Optional Arguments:
use_lines (= 'all') - an array containing the indices in the LIGHT_CONE line
 lists of the lines to be included in the analysis - by default all lines are
 used
mask_ki_0 (= True) - exclude the eta=0 modes from analysis
dlogk (= .1) - the bin spacing for constructing 1d power spectra
verbose_on (= True) - set to false to reduce output to user

Object Properties:

Object Methods:
change_lines - alter the set of lines included in use_lines and redo relevant
 analysis steps
change_noise - redraw noise in the data cube and redo relevant analysis
noise_grid - perform the noise autopower analysis
line_grids - perform the individual line autopower analysis
combined_line_grids - perform the total line autopower analysis
observed_grids - perform the autopower analysis for the full observation
matched_catalog - generate a galaxy catalog and perform cross power analysis
noise_cross - perform the noise cross power analysis
line_cross - perform the individual line cross power analysis
combined_line_cross - perform the total line cross power analysis
observed_cross - perform the cross power analysis for the full observation
cross_sensitivity_mc - monte carlo analysis of cross noise
instrument_noise_mc - monte carlo analysis of instrument contribution to cross
 noise
galaxy_noise_mc - monte carlo analysis of galaxy contribution to cross noise
galaxy_stack - stacking analysis of galaxies
plot_pow_spec - plot everything up
plot_image_contributions - image a frequency channel to see how different lines
 contribue
export_pow_spec - save power spectra to other format
'''

class radio_analysis(object):

    # Wrapper to simplify outputing info to the user
    def verbose(self, message):
        if self.verbose_on:
            print(message)

    def __init__(self, lc, map, f_target, use_lines='all', mask_ki_0=True, dlogk=.1, verbose_on=True):

        self.verbose_on = verbose_on

        self.map = map
        self.lc = lc

        if map.units == 'K':
            self.grid_units = 'uK'
            self.power_units = 'uK^2 Mpc^3'
            self.cross_units = 'uK Mpc^3'

        elif map.units == 'Jy sr-1':
            self.grid_units = 'Jy sr^-1'
            self.power_units = '(Jy sr^-1)^2 Mpc^3'
            self.cross_units = 'Jy sr^-1 Mpc^3'

        else:
            raise InputError('map', 'map.units not recognized')


        # Compute some basic parameters
        self.f_target = f_target
        self.z_target = (nu_to_z(map.f_min, self.f_target) + nu_to_z(map.f_max, self.f_target))/2
        self.x = X(self.z_target)
        self.y = Y(self.z_target, self.f_target)

        self.V_vox = prod(map.voxel) * self.x**2 * self.y
        if map.type == 'radio_map':
            self.V_no_inst = prod(map.dims) * self.x**2 * self.y
            self.V_eff = self.V_no_inst
        if map.type == 'interferometer_obs':
            self.V_no_inst = pi*(amax([amax(lc.ra),amax(lc.dec)]))**2 * map.dims[2] * self.x**2 * self.y
            self.V_eff = sum((map.beam/amax(map.beam))**2) * self.V_vox
            self.A_eff = sum((map.beam[:,:,0]/amax(map.beam[:,:,0]))**2) * prod(map.voxel[:2]) * self.x**2
            self.V_vox /= self.V_eff/self.V_no_inst

        self.physical_vox = map.voxel * array([self.x,self.x,self.y])
        self.physical_dims = map.dims * array([self.x,self.x,self.y])

        # Handle the noise
        self.verbose("Analyze map: computing noise")

        if self.map.type == 'radio_map':
            self.noise_debias = self.map.noise_debias_map ** 2 * self.V_vox

            noise_map = convolve_beam(self.map.noise_map, self.map.beam, pad=0, deconvolve=True)
            noise_k_grid, self.k_axes = ft_grid(noise_map, voxel_length=self.physical_vox)
            self.noise_map_grid = kgrid(self.map.noise_map, self.grid_units, 'Map of noise')

        if self.map.type == 'interferometer_obs':
            self.noise_debias = self.map.noise_debias_map**2 / self.A_eff * self.map.voxel[2]/self.y
            noise_k_grid, kz_axis = ft_grid(self.map.noise_map, ax=2, voxel_length=self.map.voxel[2])
            self.k_axes = [2*pi/self.x * self.map.uv_axes[0], 2*pi/self.x * self.map.uv_axes[1], kz_axis[0]/self.y]

        self.noise_k_grid = kgrid(noise_k_grid, self.grid_units, 'k_grid for noise power')
        # Power Spectrum Parameters
        self.mask_ki_0 = mask_ki_0
        self.dlogk = dlogk

        self.no_inst_mask = None
        if mask_ki_0:
            self.mask = self.map.mask * mk_mask(self.k_axes, row_dims=[0,1,2], row_inds=[(0,0),(0,0),(0,0)])
            self.no_inst_mask = mk_mask(self.k_axes, row_dims=[0,1,2], row_inds=[(0,0),(0,0),(0,0)])
        else:
            self.mask = self.map.mask

        # Handle noise power spectrum
        noise_spectrum, k_1d_obs, k_1d_count_obs = av_grid(abs(noise_k_grid*conj(noise_k_grid))/self.V_eff - self.noise_debias, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_obs /= 2
        self.noise_spectrum = power_spectrum('noise', noise_spectrum, k_1d_obs, self.power_units, k_1d_count_obs, 'Noise power spectrum')
        noise_debias_1d,discard = av_grid(self.noise_debias, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk)
        self.sensitivity = power_spectrum('sensitivity', noise_debias_1d/sqrt(k_1d_count_obs), k_1d_obs, self.power_units, k_1d_count_obs, 'Noise debias term - sensitivity')

        # Handle lines
        self.line_grids()

        # Handle lines together
        if use_lines == 'all':
            self.use_lines = arange(len(self.lc.line_names))
        else:
            self.use_lines = use_lines

        self.combined_line_grids(use_lines=self.use_lines)

        # Handle whole thing
        self.observed_grids()

        self.verbose("Analyze map: complete \n")

        # Set flags for other analysis
        self.galaxy_analysis_flag = False
        self.galaxy_analysis_sensitivity_flag = False

    def change_lines(self,use_lines='all'):
        if use_lines == 'all':
            self.use_lines = arange(len(self.lc.line_names))
        else:
            self.use_lines = use_lines

        self.combined_line_grids(use_lines=use_lines)
        if self.galaxy_analysis_flag:
            self.combined_line_cross(use_lines=use_lines)

        # Handle whole thing
        self.observed_grids()
        if self.galaxy_analysis_flag:
            self.observed_cross()

        self.verbose("Analyze map: complete \n")

    def change_noise(self,redraw=True):
        # Handle noise
        self.noise_grid(redraw=redraw)

        if self.galaxy_analysis_flag:
            print('redraw')
            self.noise_cross()

        # Handle whole thing
        self.observed_grids()
        if self.galaxy_analysis_flag:
            self.observed_cross()

        self.verbose("Analyze map: complete \n")

    def noise_grid(self, redraw=False):
        self.verbose("Analyze map: computing noise")

        if redraw:
            self.map.noise_redraw()

        if self.map.type == 'radio_map':
            noise_map = convolve_beam(self.map.noise_map, self.map.beam, pad=0, deconvolve=True)
            noise_k_grid, self.k_axes = ft_grid(noise_map, voxel_length=self.physical_vox)
            self.noise_map_grid = kgrid(self.map.noise_map, self.grid_units, 'Map of noise')

        if self.map.type == 'interferometer_obs':
            noise_k_grid, kz_axis = ft_grid(self.map.noise_map, ax=2, voxel_length=self.map.voxel[2])
            self.k_axes = [2*pi/self.x * self.map.uv_axes[0], 2*pi/self.x * self.map.uv_axes[1], kz_axis[0]/self.y]

        self.noise_k_grid = kgrid(noise_k_grid, self.grid_units, 'k_grid for noise power')

        noise_spectrum, k_1d_obs, k_1d_count_obs = av_grid(abs(noise_k_grid*conj(noise_k_grid))/self.V_eff - self.noise_debias, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_obs /= 2
        self.noise_spectrum = power_spectrum('noise', noise_spectrum, k_1d_obs, self.power_units, k_1d_count_obs, 'Noise power spectrum')



    def line_grids(self):
        self.verbose("Analyze map: computing line power spectra")

        self.true_line_k_grids = []
        self.convolved_line_k_grids = []
        self.line_map_grids = []
        self.true_line_spectra = []
        self.convolved_line_spectra = []

        for i in range(len(self.lc.line_names)):
            self.verbose("   computing line "+str(i+1)+" of "+str(len(self.lc.line_names)))

            # Handle underlying line power
            if self.map.units == 'K':
                line_temp = L_to_T(self.lc.line_luminosities[i], self.lc.line_rest_frequencies[i], self.lc.z, self.map.dtheta**2, self.map.df) * 1e6
            if self.map.units == 'Jy sr-1':
                line_temp = L_to_I(self.lc.line_luminosities[i], self.lc.z, self.map.dtheta**2, self.map.df) * 1e26

            grid, axes = mk_grid(array([self.lc.ra, self.lc.dec, self.lc.line_frequencies[i]]).T, line_temp,
                                 fit_dims=False, voxel_length=self.map.voxel,
                                 dims=self.map.dims, center=self.map.center)

            line_k_grid, k_axes = ft_grid(grid, voxel_length=self.physical_vox)
            self.true_line_k_grids.append(kgrid(line_k_grid, self.grid_units, 'k_grids for '+self.lc.line_names[i]+', no beam convolution or noise added'))

            true_line_spectrum, k_1d_true, k_1d_count_true = av_grid(abs(line_k_grid*conj(line_k_grid))/self.V_no_inst, self.k_axes, mask=self.no_inst_mask, dlogk=self.dlogk, return_count=True)
            k_1d_count_true /= 2
            self.true_line_spectra.append(power_spectrum(self.lc.line_names[i], true_line_spectrum, k_1d_true, self.power_units, k_1d_count_true, self.lc.line_names[i]+' power spectrum no beam effects'))

            # Convolve with instrument beam
            if self.map.type == 'radio_map':
                grid = convolve_beam(grid, self.map.beam, pad=0)
                self.line_map_grids.append(kgrid(grid, self.grid_units, 'Map of line '+self.lc.line_names[i]))

            if self.map.type == 'interferometer_obs':
                grid = grid*(self.map.beam/amax(self.map.beam))

            if self.map.type == 'radio_map':
                grid = convolve_beam(grid, self.map.beam, pad=0, deconvolve=True)
                line_k_grid, k_axes_tmp = ft_grid(grid, voxel_length=self.physical_vox)


            else:
                line_k_grid, k_axes_tmp = ft_grid(grid, voxel_length=self.physical_vox)

            self.convolved_line_k_grids.append(kgrid(line_k_grid, self.grid_units, 'k_grids for '+self.lc.line_names[i]+', convolved with beam before reduction'))

            convolved_line_spectrum, k_1d_obs, k_1d_count_obs = av_grid(abs(line_k_grid*conj(line_k_grid))/self.V_eff, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
            k_1d_count_obs /= 2
            self.convolved_line_spectra.append(power_spectrum(self.lc.line_names[i], convolved_line_spectrum, k_1d_obs, self.power_units, k_1d_count_obs, self.lc.line_names[i]+'beam convolved power spectrum'))



    def combined_line_grids(self,use_lines='all'):
        self.verbose("Analyze map: combining grids")

        if use_lines == 'all':
            use_lines = arange(len(self.lc.line_names))

        sum_true_line_k_grid = 0
        sum_convolved_line_k_grid = 0
        description = ''
        for i in use_lines:
            sum_true_line_k_grid += self.true_line_k_grids[i].grid
            sum_convolved_line_k_grid += self.convolved_line_k_grids[i].grid
            description = description + self.lc.line_names[i] + ', '

        self.summed_true_line_k_grids = kgrid(sum_true_line_k_grid, self.grid_units, 'k_grid for ' + description + 'no beam convolution or noise added')
        self.summed_convolved_line_k_grids = kgrid(sum_convolved_line_k_grid, self.grid_units, 'k_grid for ' + description + 'convolved with beam before reduction')

        summed_true_line_spectra, k_1d_true, k_1d_count_true = av_grid(abs(sum_true_line_k_grid*conj(sum_true_line_k_grid))/self.V_no_inst, self.k_axes, mask=self.no_inst_mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_true /= 2
        self.summed_true_line_spectra = power_spectrum('all lines', summed_true_line_spectra, k_1d_true, self.power_units, k_1d_count_true, 'Power spectrum for ' + description + 'without observational effects')

        summed_convolved_line_spectra, k_1d_obs, k_1d_count_obs = av_grid(abs(sum_convolved_line_k_grid*conj(sum_convolved_line_k_grid))/self.V_eff, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_obs /= 2
        self.summed_convolved_line_spectra = power_spectrum('all lines', summed_convolved_line_spectra, k_1d_obs, self.power_units, k_1d_count_obs, 'Power spectrum for ' + description + 'with beam effects')



    def observed_grids(self):
        self.verbose("Analyze map: computing observation power spectra")

        # Combine lines and noise
        observation_k_grid = self.summed_convolved_line_k_grids.grid + self.noise_k_grid.grid
        self.observation_k_grid = kgrid(observation_k_grid, self.grid_units, 'k_grid of noise and line power, convolved with beam before reduction')

        observation_spectrum, k_1d_obs, k_1d_count_obs = av_grid(abs(observation_k_grid*conj(observation_k_grid))/self.V_eff - self.noise_debias, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_obs /= 2
        self.observation_spectrum = power_spectrum('observation', observation_spectrum, k_1d_obs, self.power_units, k_1d_count_obs, 'Power spectrum for observation')









    def matched_catalog(self,z_err=0,completeness_function=completeness_100,shape='rect'):
        z_max = nu_to_z(self.map.f_min, self.f_target)
        z_min = nu_to_z(self.map.f_max, self.f_target)
        self.galaxies = galaxy_catalog(self.lc, self.map.ra, self.map.dec, self.map.ra_size, self.map.dec_size, z_min, z_max, z_err, completeness_function, shape)
        self.galaxies.freq_eff = z_to_nu(self.galaxies.obs_z, self.f_target)

        gal_grid, axes = mk_grid(array([self.lc.ra, self.lc.dec, self.galaxies.freq_eff]).T[self.galaxies.include],
                                 fit_dims=False, voxel_length=self.map.voxel,
                                 dims=self.map.dims, center=self.map.center)
        self.gal_counts_in_cell = gal_grid
        gal_grid = gal_grid / mean(gal_grid)
        gal_grid = gal_grid - 1

        gal_k_grid, k_axes = ft_grid(gal_grid, voxel_length=self.physical_vox)
        self.galaxy_k_grid = kgrid(gal_k_grid, 'Unitless', 'k_grid for galaxy cross correlation')

        galaxy_spectrum_true, k_1d_true, k_1d_count_true = av_grid(abs(gal_k_grid*conj(gal_k_grid))/self.V_no_inst, self.k_axes, mask=self.no_inst_mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_true /= 2
        self.galaxy_true_spectrum = power_spectrum('galaxies', galaxy_spectrum_true, k_1d_true, 'Unitless', k_1d_count_true, 'Power spectrum of galaxies in matched catalog, no masking applied')

        galaxy_spectrum_obs, k_1d_obs, k_1d_count_obs = av_grid(abs(gal_k_grid*conj(gal_k_grid))/self.V_eff, self.k_axes, mask=self.mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_obs /= 2
        self.galaxy_obs_spectrum = power_spectrum('galaxies', galaxy_spectrum_obs, k_1d_obs, 'Unitless', k_1d_count_obs, 'Power spectrum of galaxies in matched catalog, masked to match radio uv coverage')

        self.noise_cross()
        self.line_cross()
        self.combined_line_cross(use_lines=self.use_lines)
        self.observed_cross()

        self.cross_sensitivity = power_spectrum('Cross Sensitivity', zeros(len(self.sensitivity.spectrum)), self.sensitivity.axis, self.cross_units, zeros(len(self.sensitivity.spectrum)), 'Cross power sensitivity')
        self.cross_sensitivity_gal = power_spectrum('Cross Sensitivity - Galaxies', zeros(len(self.sensitivity.spectrum)), self.sensitivity.axis, self.cross_units, zeros(len(self.sensitivity.spectrum)), 'Cross power sensitivity - random galaxies')
        self.cross_sensitivity_inst = power_spectrum('Cross Sensitivity - Instrumental', zeros(len(self.sensitivity.spectrum)), self.sensitivity.axis, self.cross_units, zeros(len(self.sensitivity.spectrum)), 'Cross power sensitivity - insturment noise')

        self.verbose("Analyze map: cross power analysis complete \n")

        self.galaxy_analysis_flag = True



    def noise_cross(self):
        self.verbose("Analyze map: computing noise cross galaxies")

        noise_cross_spectrum, k_1d_obs, k_1d_count_obs = av_grid(real(self.galaxy_k_grid.grid*conj(self.noise_k_grid.grid))/self.V_eff, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_obs /= 2
        self.noise_cross_spectrum = power_spectrum('noise cross galaxies', noise_cross_spectrum, k_1d_obs, self.cross_units, k_1d_count_obs, 'Noise-galaxy cross power spectrum')



    def line_cross(self):
        self.verbose("Analyze map: computing line cross power spectra")

        self.true_line_cross_spectra = []
        self.convolved_line_cross_spectra = []

        for i in range(len(self.lc.line_names)):
            self.verbose("   computing line "+str(i+1)+" of "+str(len(self.lc.line_names)))

            true_line_cross_spectrum, k_1d_true, k_1d_count_true = av_grid(real(self.galaxy_k_grid.grid*conj(self.true_line_k_grids[i].grid))/self.V_no_inst, self.k_axes, mask=self.no_inst_mask, dlogk=self.dlogk, return_count=True)
            k_1d_count_true /= 2
            self.true_line_cross_spectra.append(power_spectrum(self.lc.line_names[i]+' cross galaxies', true_line_cross_spectrum, k_1d_true, self.cross_units, k_1d_count_true, self.lc.line_names[i]+' cross sepctrum, no beam effects'))

            convolved_line_cross_spectrum, k_1d_obs, k_1d_count_obs = av_grid(real(self.galaxy_k_grid.grid*conj(self.convolved_line_k_grids[i].grid))/self.V_eff, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
            k_1d_count_obs /= 2
            self.convolved_line_cross_spectra.append(power_spectrum(self.lc.line_names[i]+' cross galaxies', convolved_line_cross_spectrum, k_1d_obs, self.cross_units, k_1d_count_obs, self.lc.line_names[i]+' beam convolved cross sepctrum'))

    def combined_line_cross(self,use_lines='all'):
        self.verbose("Analyze map: combining line cross power spectra")

        if use_lines == 'all':
            use_lines = arange(len(self.lc.line_names))

        description = ''
        for i in use_lines:
            description = description + self.lc.line_names[i] + ', '

        summed_true_line_cross_spectra, k_1d_true, k_1d_count_true = av_grid(real(self.galaxy_k_grid.grid*conj(self.summed_true_line_k_grids.grid))/self.V_no_inst, self.k_axes, mask=self.no_inst_mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_true /= 2
        self.summed_true_line_cross_spectra = power_spectrum('all lines cross galaxies', summed_true_line_cross_spectra, k_1d_true, self.cross_units, k_1d_count_true, 'Cross power spectrum for ' + description + 'without observational effects')

        summed_convolved_line_cross_spectra, k_1d_obs, k_1d_count_obs = av_grid(real(self.galaxy_k_grid.grid*conj(self.summed_convolved_line_k_grids.grid))/self.V_eff, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_obs /= 2
        self.summed_convolved_line_cross_spectra = power_spectrum('all lines cross galaxies', summed_convolved_line_cross_spectra, k_1d_obs, self.cross_units, k_1d_count_obs, 'Cross power spectrum for ' + description + 'with beam effects')



    def observed_cross(self):
        self.verbose("Analyze map: computing observation cross power spectra")

        # Combine lines and noise
        observation_cross_spectrum, k_1d_obs, k_1d_count_obs = av_grid(real(self.galaxy_k_grid.grid*conj(self.observation_k_grid.grid))/self.V_eff, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk, return_count=True)
        k_1d_count_obs /= 2
        self.observation_cross_spectrum = power_spectrum('observation cross galaxies', observation_cross_spectrum, k_1d_obs, self.cross_units, k_1d_count_obs, 'Power spectrum for observation')



    def cross_sensitivity_mc(self, n_realizations=100):
        verbose_on = self.verbose_on
        self.verbose_on = False
        self.instrument_noise_mc(n_realizations)
        self.galaxy_noise_mc(n_realizations)

        self.cross_sensitivity.spectrum = sqrt(self.cross_sensitivity_inst.spectrum**2 + self.cross_sensitivity_gal.spectrum**2)
        self.cross_sensitivity.counts = self.cross_sensitivity_inst.counts

        self.galaxy_analysis_sensitivity_flag = True
        self.verbose_on = verbose_on


    def instrument_noise_mc(self, n_realizations=100):

        noise_grid_save = deepcopy(self.noise_k_grid)
        noise_spec_save = deepcopy(self.noise_spectrum)
        noise_cross_save = deepcopy(self.noise_cross_spectrum)

        rand_spectra = zeros((n_realizations, len(self.observation_spectrum.spectrum)))
        for i in range(n_realizations):
            self.noise_grid(redraw=True)
            self.noise_cross()
            rand_spectra[i,:] = self.noise_cross_spectrum.spectrum
            print(i)
        rand_spectra = sort(abs(rand_spectra), axis=0)
        ind = abs(((arange(n_realizations) + 1) / n_realizations) - .6827).argmin()
        self.cross_sensitivity_inst.spectrum = rand_spectra[ind]
        self.cross_sensitivity_inst.counts = self.noise_cross_spectrum.counts

        self.noise_k_grid = noise_grid_save
        self.noise_spectrum = noise_spec_save
        self.noise_cross_spectrum = noise_cross_save

    def galaxy_noise_mc(self, n_realizations=100):

        # Galaxy noise MC
        n = sum(self.galaxies.include)
        rand_spectra = zeros((n_realizations, len(self.observation_spectrum.axis)))
        size = array([self.galaxies.ra_size, self.galaxies.dec_size, (cosmo.comoving_distance(self.galaxies.z_max)-cosmo.comoving_distance(self.galaxies.z_min)).value])
        offset = array([-self.galaxies.ra, -self.galaxies.dec, cosmo.comoving_distance(self.galaxies.z_min).value])
        for i in range(n_realizations):
            pos = rand(n,3)
            pos = pos * size + offset
            pos[:,2] = z_to_nu(d_to_z(pos[:,2]),self.f_target)

            rand_grid, axes = mk_grid(pos,
                                     fit_dims=False, voxel_length=self.map.voxel,
                                     dims=self.map.dims, center=self.map.center)
            rand_grid = rand_grid / mean(rand_grid)
            rand_grid = rand_grid - 1

            rand_k_grid, k_axes = ft_grid(rand_grid, voxel_length=self.physical_vox)
            rand_spectrum, k_1d_obs = av_grid(real(rand_k_grid*conj(self.summed_convolved_line_k_grids.grid))/self.V_eff, self.k_axes, weights=self.map.weights, mask=self.mask, dlogk=self.dlogk)
            rand_spectra[i,:] = rand_spectrum
            print(i)

        rand_spectra = sort(abs(rand_spectra), axis=0)
        ind = abs(((arange(n_realizations) + 1) / n_realizations) - .6827).argmin()
        self.cross_sensitivity_gal = power_spectrum('Galaxy cross noise MC',rand_spectra[ind],k_1d_obs,self.cross_units,self.galaxy_obs_spectrum.counts,"Observation cross random galaxies from MC simulation")



    def galaxy_stack(self, n_realizations=1000):
        observation_map_grid = 0
        for i in self.use_lines:
            observation_map_grid += self.line_map_grids[i].grid
        observation_map_grid += self.noise_map_grid.grid

        bin_size = int(round(2*self.map.fwhm/self.map.dtheta))
        observation_map_grid = rebin(observation_map_grid, bin_size)

        gal_counts_in_cell_rebin = rebin(self.gal_counts_in_cell, bin_size)
        shape = array(gal_counts_in_cell_rebin.shape)

        mask = zeros(shape)
        mask[gal_counts_in_cell_rebin > 0] = 1
        gal_stack = sum(observation_map_grid*mask,axis=(0,1))


        n_gal = (sum(gal_counts_in_cell_rebin,axis=(0,1))+.1).astype('int')

        noise_stack = zeros((shape[2],n_realizations))
        for i in range(n_realizations):
            mask *= 0
            ind1 = randint(0,shape[0],sum(n_gal))
            ind2 = randint(0,shape[1],sum(n_gal))
            ind3 = concatenate([ones(n_gal[j],dtype='int')*j for j in range(len(n_gal))])
            inds = unique(array([ind1,ind2,ind3]).T,axis=0)
            mask[tuple(inds.T)] = 1
            noise_stack[:,i] = sum(observation_map_grid*mask,axis=(0,1))

        noise_sd = std(noise_stack, axis=1)
        plt.plot(self.map.ax[2].flatten(),gal_stack)
        plt.plot(self.map.ax[2].flatten(),noise_sd)
        plt.show()
        plt.hist(noise_stack[0,:])
        plt.hist(noise_stack[1,:])
        plt.hist(noise_stack[-1,:])
        #plt.hist([gal_stack],density=True)
        plt.show()






    def plot_pow_spec(self, delta=False, true_lines=True, cross=False):
        h = cosmo.H0.value/100

        if not cross:
            sensitivity = self.sensitivity
            noise = self.noise_spectrum
            obs = self.observation_spectrum
            sum = self.summed_convolved_line_spectra
            lines = self.convolved_line_spectra
            true_lines = self.true_line_spectra
            unit_power = '$^2$'
            power_sub = ''
        else:
            if self.galaxy_analysis_flag == False:
                raise InputError('radio_analysis','no galaxy cross analysis performed')

            sensitivity = self.cross_sensitivity
            sensitivity_g = self.cross_sensitivity_gal
            sensitivity_i = self.cross_sensitivity_inst
            noise = self.noise_cross_spectrum
            obs = self.observation_cross_spectrum
            sum = self.summed_convolved_line_cross_spectra
            lines = self.convolved_line_cross_spectra
            true_lines = self.true_line_cross_spectra
            unit_power = ''
            power_sub = '$_X$'

        if self.map.type == 'radio_map':
            xmax = 1.1*2*pi/self.x/self.map.fwhm/h
            use_obs = where(sum.axis/h <= xmax)
            use_true = where(true_lines[0].axis/h <= xmax)
        else:
            use_obs = where(sum.axis/h <= 1000)
            use_true = where(true_lines[0].axis/h <= 1000)

        if delta:
            if self.map.units == 'Jy sr-1':
                ylabel = '$\Delta$'+power_sub+'$^2(k)\ [(Jy\ sr^{-1})$'+unit_power+'$]$'
            else:
                ylabel = '$\Delta$'+power_sub+'$^2(k)\ [\mu K$'+unit_power+'$]$'
            obs_pre = obs.axis[use_obs]**3/2/pi**2
            true_pre = true_lines[0].axis[use_true]**3/2/pi**2
        else:
            if self.map.units == 'Jy sr-1':
                ylabel = '$P$'+power_sub+'$(k)\ [(Jy\ sr^{-1})$'+unit_power+'$\ h^{-3} Mpc^3]$'
            else:
                ylabel = '$P$'+power_sub+'$(k)\ [\mu K$'+unit_power+'$\ h^{-3} Mpc^3]$'
            obs_pre = h**3
            true_pre = h**3


        fig = plt.figure(figsize=(8,8))
        pplot = fig.add_subplot(111)
        pplot.set(title='Observed Power Spectrum',
                  xlabel='k [$h Mpc^{-1}$]', xscale='log',
                  ylabel=ylabel, yscale='log')
        pplot.grid()

        pplot.fill_between(sum.axis[use_obs]/h, obs_pre*(sum.spectrum-sensitivity.spectrum)[use_obs], obs_pre*(sum.spectrum+sensitivity.spectrum)[use_obs], color='k', alpha=.2)

        pplot.plot(obs.axis[use_obs]/h, obs_pre*obs.spectrum[use_obs], 'k', label='Realized Measurement')
        pplot.plot(sum.axis[use_obs]/h, obs_pre*sum.spectrum[use_obs], 'k--', label='Underlying Power')

        for i in self.use_lines:
            pplot.plot(lines[i].axis[use_obs]/h, obs_pre*lines[i].spectrum[use_obs], '--', c=colors[i%10], label=self.lc.line_names[i])
            if true_lines:
                pplot.plot(true_lines[i].axis[use_true]/h, true_pre*true_lines[i].spectrum[use_true], ':', c=colors[i%10])

        pplot.plot(noise.axis[use_obs]/h, obs_pre*noise.spectrum[use_obs], 'ro', label='Realized Noise')
        pplot.plot(noise.axis[use_obs]/h, -obs_pre*noise.spectrum[use_obs], 'ro', fillstyle='none')
        pplot.plot(noise.axis[use_obs]/h, obs_pre*sensitivity.spectrum[use_obs], 'r:', label='Sensitivity')
        if cross:
            pplot.plot(noise.axis[use_obs]/h, obs_pre*sensitivity_i.spectrum[use_obs], ':', c='tomato', label='Instrument Noise Sensitivity')
            pplot.plot(noise.axis[use_obs]/h, obs_pre*sensitivity_g.spectrum[use_obs], ':', c='maroon', label='Confusion Sensitivity')

        pplot.legend()
        plt.show()


    def plot_image_contributions(self, channel_ind=0,scale='lin'):
        if scale == 'log':
            def scale_f(x):
                return log10(x)
        else:
            def scale_f(x):
                return x

        fig = plt.figure(figsize=(8,8))
        image = fig.add_subplot(111)

        back = zeros(self.line_map_grids[0].grid[:,:,channel_ind].shape)
        #image.imshow(back,cmap='Greys_r')
        colors = zeros((back.shape[0],back.shape[1],3))

        vmax = 0
        for i in self.use_lines:
            vmax = amax([vmax, amax(self.line_map_grids[i].grid[:,:,channel_ind])])

        colors[...,0] = scale_f(self.line_map_grids[self.use_lines[0]].grid[:,:,channel_ind])/scale_f(vmax)
        labels = [mpatches.Patch(color=(1,0,0), label=self.lc.line_names[self.use_lines[0]])]

        if len(self.use_lines) > 1:
            colors[...,2] = scale_f(self.line_map_grids[self.use_lines[1]].grid[:,:,channel_ind])/scale_f(vmax)
            labels.append(mpatches.Patch(color=(0,0,1), label=self.lc.line_names[self.use_lines[1]]))
        if len(self.use_lines) > 2:
            colors[...,1] = scale_f(self.line_map_grids[self.use_lines[2]].grid[:,:,channel_ind])/scale_f(vmax)
            labels.append(mpatches.Patch(color=(0,1,0), label=self.lc.line_names[self.use_lines[2]]))

            #colors = cmaps[i%5](colors)
            #colors[...,-1] = log10(self.line_map_grids[i].grid[:,:,channel_ind])/log10(vmax)
            #print(colors)
            #print(colors.shape)
        image.imshow(colors)
        image.legend(handles=labels)

        plt.show()


    def export_pow_spec(self, filename, format='csv', delta=False, cross=False):
        h = cosmo.H0.value/100
        if delta:
            obs_pre = self.observation_spectrum.axis**3/2/pi**2
        else:
            obs_pre = h**3

        if not cross:
            data = zeros(len(self.observation_spectrum.axis),dtype=[('k','f'),('sensitivity','f'),('realized noise','f'),('observed spectrum','f'),('noiseless spectrum','f')])
            data['k'] = self.observation_spectrum.axis/h
            data['sensitivity'] = self.sensitivity.spectrum*obs_pre
            data['realized noise'] = self.noise_spectrum.spectrum*obs_pre
            data['observed spectrum'] = self.observation_spectrum.spectrum*obs_pre
            data['noiseless spectrum'] = self.summed_convolved_line_spectra.spectrum*obs_pre
            header = 'k,\t sensitivity,\t realized noise,\t observed spectrum,\t noiseless spectrum,\t '

            lines = self.convolved_line_spectra

        else:
            if self.galaxy_analysis_flag == False:
                raise InputError('radio_analysis','no galaxy cross analysis performed')

            data = zeros(len(self.observation_spectrum.axis),dtype=[('k','f'),('realized cross noise','f'),('observed cross spectrum','f'),('noiseless cross spectrum','f')])
            data['k'] = self.observation_spectrum.axis/h
            data['realized cross noise'] = self.noise_cross_spectrum.spectrum*obs_pre
            data['observed cross spectrum'] = self.observation_cross_spectrum.spectrum*obs_pre
            data['noiseless cross spectrum'] = self.summed_convolved_line_cross_spectra.spectrum*obs_pre
            header = 'k,\t realized cross noise,\t observed cross spectrum,\t noiseless cross spectrum,\t '

            lines = self.convolved_line_cross_spectra

        for i in self.use_lines:
            data = numpy.lib.recfunctions.append_fields(data,[lines[i].name],data=[lines[i].spectrum*obs_pre],usemask=False)
            header = header + lines[i].name + ',\t'

        if format == 'csv':
            savetxt(filename, data, delimiter=',', header=header)
        elif format == 'npy':
            save(filename, data)
        elif format == 'return':
            return data
        else:
            raise InputError('format', 'format not recognized, supported formats are \'csv\', \'npy\', \'return\'')
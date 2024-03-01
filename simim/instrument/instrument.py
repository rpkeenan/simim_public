import numpy as np
import warnings
from simim._mplsetup import *
from simim.instrument.noise import zero_noise, white_noise
from simim.instrument.psf import gauss_psf, gauss_psf_freq_dependent
from simim.instrument.spec_response import gauss_response, boxcar_response
from simim.instrument.helpers import _check_unit, _check_grid_detector_compatibility
from simim.map.gridder import _grid, grid_from_axes_and_function

class Detector():

    def __init__(self,spatial_response,spectral_response,noise_function=None,
                 name=None,
                 spatial_unit='rad',spatial_kwargs={},
                 spectral_unit='Hz',spectral_kwargs={},nominal_frequency=None,
                 flux_unit='Jy',noise_kwargs={}):

        """An object describing a detector in terms of spatial and spectral response plus
        (optional) noise.

        Spatial and spectral response are assumed to be independent, although spatial response is
        evaluated as functions of all three map dimensions (so a spectrally varrying PSF is possible). 
        When extracting detector response to a map, the map will be convolved with the spatial response
        and then averaged along the spectral response dimension, weighted by the spectral response.

        A few default responses are available and can be specified by giving a string rather
        than a function when initializing. 
        
        Parameters
        ----------
        spatial_response : function, or one of 'gauss', 'spec-gauss'
            function that takes vectors of x offset, y offset (in spatial_unit), and frequency 
            (in spectral_unit) and returns the spatial response of the detector. Additional
            function arguments can be as a dictionary of keyword arguments using spatial_kwargs.
            If 'gauss' a two dimensional gaussian will be used. The center (default zero) and
            width can be controlled by specifying the 'cx', 'cy', 'fwhmx', 'fwhmy' in spatial_kwargs.
            If 'spec-gauss' a two dimensional gaussian that varries as a function of frequency
            will be used. The center (default zero) can be controlled by specifying the 'cx' and 'cy' 
            in spatial_kwargs. The width at a specific frequency are set by specifying 'freq0',
            'fwhmx', and 'fwhmy'. The scaling of the size with frequency is set using the 
            'freq_exp' factor (default -1). The size at a given frequency will then be 
            fwhmx_f = fwhmx * (freq0/freq)**freq_exp.
        spectral_response : function, or one of 'gauss', 'boxcar'
            function that takes vectors of frequency (in spectral_unit) and returns the 
            spectral response of the detector. Additional function arguments can be passed 
            as a dictionary of keyword arguments using spectral_kwargs.
            If 'gauss' a gaussian response will be used. The center frequency, width and
            amplitude can be controlled by specifying the 'freq0', 'fwhm', and 'A' in 
            spectral_kwargs.
            If 'boxcar' a boxcar response will be used. The center frequency, width and
            amplitude can be controlled by specifying the 'freq0', 'fwhm', and 'A' in 
            spectral_kwargs.
        noise_function : function, 'white', or None
            function that takes an integer number of samples and a timestep, and returns 
            a time series of noise - this can be used to add noise to a timestream. Additional
            function arguments can be passed as adictionary of keyword arguments using 
            noise_kwargs.
            If 'white' a default white noise function will be used. The rms for a 1 second
            sample and a DC offset can be specified using the 'rms' and 'bias' parameters in
            noise_kwargs.
        name : string, default=None
            A name for the detector
        spatial_unit : {'rad','deg','arcmin','arcsec'}, default='rad'
            The units in which the spatial response function is defined
        spatial_kwargs : dict, optional
            A dictionary of kwargs that will be passed to the spatial_response function
        spectral_unit : {'Hz','GHz','m','mm','um'}, default='Hz'
            The units in which the spectral response function is defined
        nominal_frequency : float, default=None
            Optionally specify a nominal frequency of the detector.
        spectral_kwargs : dict, optional
            A dictionary of kwargs that will be passed to the spectral_response function
        flux_unit : {'Jy','mJy'}, default='Jy'
            The units in which the the noise_function provides noise for the
            detector
        noise_kwargs : dict, optional
            A dictionary of kwargs that will be passed to the noise_function function        
        """

        # Name for the detector
        self.name = name
        self.nominal_frequency = nominal_frequency

        # Collect response functions, assign the correct ones for some defaults
        # that can be entered as strings instead of functions. Verify that each
        # one is actually a function.
        if spatial_response in ['gauss','Gauss','gaussian','Gaussian']:
            self.spatial_response = gauss_psf
        elif spatial_response in ['specgauss','spec-gauss','SpecGauss','Spec-Gauss','spectral-gaussian','Spectral-Gaussian']:
            self.spatial_response = gauss_psf_freq_dependent
        elif not hasattr(spatial_response, '__call__'):
            raise ValueError("spatial_response must have a __call__ method")
        else:
            self.spatial_response = spatial_response

        if spectral_response in ['gauss','Gauss','gaussian','Gaussian']:
            self.spectral_response = gauss_response
        elif spectral_response in ['box','boxcar','Box','Boxcar']:
            self.spectral_response = boxcar_response
        elif not hasattr(spectral_response, '__call__'):
            raise ValueError("spectral_response must have a __call__ method")
        else:
            self.spectral_response = spectral_response
        
        if noise_function is None:
            self.noise_function = zero_noise
        elif noise_function in ['none','None']:
            self.noise_function = zero_noise
        elif noise_function in ['white','White']:
            self.noise_function = white_noise
        elif not hasattr(noise_function, '__call__'):
            raise ValueError("noise_function must have a __call__ method")
        else:
            self.noise_function = noise_function
        
        # Make sure function keyword arguments are correctly setup 
        if not isinstance(spatial_kwargs, dict):
            raise ValueError("spatiral_kwargs must be a dict")
        else:
            self.spatial_kwargs = spatial_kwargs

        if not isinstance(spectral_kwargs, dict):
            raise ValueError("spectral_kwargs must be a dict")
        else:
            self.spectral_kwargs = spectral_kwargs
            
        if not isinstance(noise_kwargs, dict):
            raise ValueError("noise_kwargs must be a dict")
        else:
            self.noise_kwargs = noise_kwargs

        # Deal with units
        _check_unit(spatial_unit,'spatial')
        self.spatial_unit = spatial_unit
        _check_unit(spectral_unit,'spectral')
        self.spectral_unit = spectral_unit
        _check_unit(flux_unit,'flux')
        self.flux_unit = flux_unit

        # Set up some places to hold additional data later
        self.field_initialized = False

        self.map_initialized = False
        self.map_noise_initialized = False
        self.map = None
        self.map_noise = None
        self.map_axes = None

        self.timestream_initialized = False
        self.timestream_noise_initialized = False
        self.timestream = None
        self.timestream_noise = None
        self.timestream_axes = None


    def plot_response(self,fmin,fmax,fspatial,xmin,xmax,ymin,ymax,figsize=(10,5)):

        fig,axes = plt.subplots(1,2,figsize=figsize)
        fig.subplots_adjust(left=.1,right=.95,top=.95,bottom=.15,wspace=.35)
        axes[0].set(xlabel='Frequency [{}]'.format(self.spectral_unit),ylabel='Response')
        axes[1].set(aspect='equal',xlabel='dx [{}]'.format(self.spatial_unit),ylabel='dy [{}]'.format(self.spatial_unit))

        f = np.linspace(fmin,fmax,1000)
        axes[0].plot(f,self.spectral_response(f,**self.spectral_kwargs))
        axes[0].axvline(fspatial,color='k',ls='--')

        x = np.linspace(xmin,xmax,100)
        y = np.linspace(ymin,ymax,100)
        axes[1].pcolor(x,y,self.spatial_response(x,y,fspatial,**self.spatial_kwargs)[:,:,0],vmin=0)

        plt.show()


    def set_field(self,grid,field_property_idx=None,_check=True):
        if grid.n_properties > 1 and field_property_idx is None:
            field_property_idx = 0
            warnings.warn("Grid has multiple properties, property index 0 will be used. You can specify the desired property with field_property_idx.")
        if _check:
            _check_grid_detector_compatibility(self,grid,field_property_idx)

        self.field = grid
        self.field_property_idx = field_property_idx
        self.field_initialized = True


    def _setup_beam(self,kernel_size=None,spatial_response_norm='peak'):

        if kernel_size is None:
            kernel_size = np.copy(self.field.side_length[:2])
        kernel_size = np.array(kernel_size)
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must have length 2 (xsize, ysize)")
        
        # Check that the value of norm is recognized
        if spatial_response_norm not in ['peak','area','none']:
            raise ValueError("spatial_response_norm must be one of 'peak', 'area', 'none'")
        
        # Set up axes of beam convolution kernel
        center = [0,0,self.field.center_point[2]]
        side_length = [kernel_size[0],kernel_size[1],self.field.side_length[2]]

        # Evaluate beam response over kernel axes
        beam = _grid(n_properties=1,center_point=center,side_length=side_length,pixel_size=self.field.pixel_size,axunits=self.field.axunits,gridunits=self.field.gridunits)
        beam.init_grid()

        shape = beam.grid.shape
        beam.grid = np.expand_dims(self.spatial_response(beam.axes_centers[0],beam.axes_centers[1],beam.axes_centers[2],**self.spatial_kwargs),-1)
        if beam.grid.shape != shape:
            raise ValueError("spatial_response does not produce an array of the correct shape")

        if spatial_response_norm == 'peak':
            beam.grid = beam.grid / np.max(beam.grid,axis=(0,1,3)).reshape(1,1,-1,1)
        if spatial_response_norm == 'area':
            beam.grid = beam.grid / np.sum(beam.grid,axis=(0,1,3)).reshape(1,1,-1,1)
        if spatial_response_norm == 'none':
            pass

        return beam


    def map_field(self,kernel_size=None,spatial_response_norm='peak',beam=None,pad=None,_check=True):
        # Either speciffy kernel_size and spatial_response_norm or specify beam (with a grid containing the beam)
        # Note that beams provided using beam will not be re-normalized.

        if beam is not None and kernel_size is not None:
            raise ValueError("If beam is specified, kernel_size should be left as None")

        # Set off so it only has to be done once when an Instrument object iterates over many detectors
        if _check:
            if not self.field_initialized:
                raise ValueError("No field specified - cannot map field")

        if beam is None:
            beam = self._setup_beam(kernel_size=None,spatial_response_norm=spatial_response_norm)
                    
        if _check and beam.grid.ndim != 4:
            raise ValueError("Beam does not have three dimensions")

        # Convolve with beam
        self.map = self.field.convolve(beam,ax=(0,1),in_place=False,pad=pad)

        # Evaluate spectral response over frequency axis
        spec = self.spectral_response(beam.axes_centers[2],**self.spectral_kwargs)

        # Collapse along the third dimension using spectral response
        self.map = self.map.collapse_dimension(ax=2,weights=spec,mode='average',in_place=True)
        self.map_initialized = True


    def sample(self,positions,dt,properties=None,signal=True,noise=True):
                
        if signal and noise:
            self.sample_signal(positions,properties=properties)
            self.sample_noise(len(positions),dt)
            self.timestream = self.timestream_signal + self.timestream_noise.reshape(-1,1)
        elif signal:
            self.sample_signal(positions,properties=properties)
            self.timestream = self.timestream_signal
        elif noise:
            self.sample_noise(len(positions),dt)
            self.timestream = self.timestream_noise.reshape(-1,1)
        self.timestream_positions = positions
        self.timestream_dt = dt

        return self.timestream


    def sample_signal(self,positions,properties=None):
        if not self.map_initialized:
            raise ValueError("No map initialized, cannot sample")
        
        self.timestream_signal = self.map.sample(positions=positions,properties=properties)
        return self.timestream_signal


    def sample_noise(self,nsamples,dt):
        self.timestream_noise = self.noise_function(nsamples,dt,**self.noise_kwargs)
        return self.timestream_noise


class Instrument():

    def __init__(self,name='instrument',spatial_unit=None,spectral_unit=None,flux_unit=None,offset_type='ra-dec'):

        self.name = name

        # Deal with unit inputs
        if spatial_unit is None or spectral_unit is None or flux_unit is None:
            if spatial_unit is not None or spectral_unit is not None or flux_unit is not None:
                raise ValueError("If units are specified then spatial, spectral, and flux units must all be provided")
            self.flag_units_init = False
        else:
            self.init_units(spatial_unit,spectral_unit,flux_unit)

        if offset_type not in ['ra-dec']:
            raise ValueError("Supported offset types are 'ra-dec'")

        # Place for detector info
        self.detector_names = []
        self.detectors = {}
        self.detector_frequencies = {}
        self.detector_offsets = {}
        self.detector_offsets = {}
        self.detector_maps_initialized = {}
        self.detector_maps = {}

        # Flags for things
        self.field_initialized = False

    def init_units(self,spatial_unit,spectral_unit,flux_unit):
        
        _check_unit(spatial_unit,'spatial')
        self.spatial_unit = spatial_unit
        _check_unit(spectral_unit,'spectral')
        self.spectral_unit = spectral_unit
        _check_unit(flux_unit,'flux')
        self.flux_unit = flux_unit

        self.flag_units_init = True

    def add_detector(self,detector,name=None,nominal_frequency=None,offset=(0,0)):
        
        if not isinstance(detector,Detector):
            raise ValueError("detector must be an instance of the instrument.Detector class")

        # Give a name if none specified
        if name is None:
            if detector.name is None:
                name = len(self.detector_names)
                # If the number is already used, then add 1
                while str(name) in self.detector_names:
                    name += 1
                name = str(name)
            else:
                name = str(detector.name)

        # Check if detector has a nominal frequency
        if nominal_frequency is None:
            if detector.nominal_frequency is not None:
                nominal_frequency = detector.nominal_frequency

        # Check offset has the right shape
        if len(np.array(offset)) != 2:
            raise ValueError("Detector offset must be convertable to a numpy array and have length 2")

        # Check units
        if self.flag_units_init:
            if self.spectral_unit != detector.spectral_unit:
                raise ValueError("detector spectral unit ({}) not consistent with instrument spectral unit ({})".format(detector.spectral_unit,self.spectral_unit))
            if self.spatial_unit != detector.spatial_unit:
                raise ValueError("detector spatial unit ({}) not consistent with instrument spatial unit ({})".format(detector.spatial_unit,self.spatial_unit))
            if self.flux_unit != detector.flux_unit:
                raise ValueError("detector flux unit ({}) not consistent with instrument flux unit ({})".format(detector.flux_unit,self.flux_unit))
        # If units not set up, inherit them from the detector
        else:
            print("Inheriting unit system from detector")
            self.init_units(detector.spatial_unit,detector.spectral_unit,detector.flux_unit)

        # Add detector
        self.detector_names.append(name)
        self.detector_frequencies[name] = nominal_frequency
        self.detector_offsets[name] = np.array(offset)
        self.detectors[name] = detector
        self.detector_maps[name] = False

    def plot_detector_response(self,detector,fmin,fmax,fspatial,xmin,xmax,ymin,ymax,figsize=(10,5)):
        if detector not in self.detector_names:
            raise ValueError("Specified detector must be in self.detector_names")
        
        self.detectors[detector].plot_response(fmin,fmax,fspatial,xmin,xmax,ymin,ymax,figsize)

    def plot_response(self,fmin,fmax,figsize=(10,5)):

        fig,axes = plt.subplots(1,1,figsize=figsize)
        fig.subplots_adjust(left=.1,right=.95,top=.95,bottom=.15,wspace=.35)
        axes.set(xlabel='Frequency [{}]'.format(self.spectral_unit),ylabel='Response')

        f = np.linspace(fmin,fmax,5000)
        cmap = cm.get_cmap('Spectral')

        for i,name in enumerate(self.detector_names):
            det = self.detectors[name]
            c = cmap(i/len(self.detector_names))
            axes.plot(f,det.spectral_response(f,**det.spectral_kwargs),color=c)

        plt.show()

    def set_field(self,grid,field_property_idx=None):
        if grid.n_properties > 1 and field_property_idx is None:
            field_property_idx = 0
            warnings.warn("Grid has multiple properties, property index 0 will be used. You can specify the desired property with field_property_idx.")
        _check_grid_detector_compatibility(self,grid,field_property_idx)

        for name in self.detector_names:
            self.detectors[name].set_field(grid,field_property_idx,_check=False)

        self.field = grid
        self.field_property_idx = field_property_idx
        self.field_initialized = True

    def map_field(self,kernel_size=None,pad=None,spatial_response_norm='peak',remake_all=True):
        # If you've added new detectors and want to add maps for only those, set remake_all=False

        if not self.field_initialized:
            raise ValueError("No field specified - cannot map field")
        if len(self.detector_names) == 0:
            raise ValueError("No detectors added, cannot map field")

        ref_detector = self.detector_names[0]
        beam = self.detectors[ref_detector]._setup_beam(kernel_size=kernel_size,spatial_response_norm=spatial_response_norm)

        for name in self.detector_names:
            if remake_all or not self.detector_maps_initialized[name]:
                print("mapping detector {}".format(name))
                self.detectors[name].map_field(beam=beam,pad=pad,_check=False)
                self.detector_maps_initialized[name] = True
                self.detector_maps_initialized[name] = self.detectors[name].map.grid[:,:,0]

    # TO DO
    def sample(self,):
        pass
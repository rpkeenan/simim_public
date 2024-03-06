import numpy as np
import warnings
from typing import Callable

from simim._pltsetup import *
from simim.map import Grid
from simim.instrument.helpers import _check_unit, _check_grid_detector_compatibility, _specunit_type, _dict_none_copy


from simim.instrument.noise_functions import zero_noise, white_noise
from simim.instrument.spatial_response import gauss_psf, gauss_psf_freq_dependent
from simim.instrument.spectral_response import gauss_response, boxcar_response
from simim.map import GridFromAxesAndFunction

# Refactor Outline:
##     method to map spectral cube over all detectors (optional: just one detector)
##     method to sample spectral cube over all detectors (should be possible with current grid code: just one detector)
##     methods to sample just noise/just signal
##     * method to attach detectors, combine detectors from other instruments
##     * method to plot detector response (out of all detectors) next to the map of a field from that detector (optional: animate)
## Detector class - wraps around instrument to setup a single-detector instrument, redefines methods to do the one det versions


class _DetectorInfo:

    reffreq: float
    spatial_response: Callable
    spatial_kwargs: dict
    spectral_response: Callable
    spectral_kwargs: dict
    noise_function: Callable
    noise_kwargs: dict

    def __init__(self, name: str):
        self.name = name


class Instrument:
    """Stores and evaluates instrument response to a simulated sky
    
    An Instrument object represents an telescope+instrument capable of observing
    a simulated sky. Instruments are built by adding one or more Detectors,
    which describe a single detector of the instrument (or a collection of
    identical detectors).

    Each detector is defined in terms of its spatial and spectral response and
    an optional noise function. Spatial and spectral response are assumed to be
    independent, although spatial response is evaluated as functions of all
    three map dimensions (so a spectrally varrying PSF is possible). When
    extracting detector response to a map, the map will be convolved with the
    spatial response and then averaged along the spectral response dimension,
    weighted by the spectral response.

    A few default responses are predefined and can be specified by giving a
    string rather than a function when attaching new detectors. 
    """
    def __init__(self, 
                 spatial_unit: str = None, 
                 spectral_unit: str = None, 
                 flux_unit: str = None,
                 best_spatial_res: float = None, 
                 best_spectral_res: float = None,
                 default_spatial_response: Callable = None,
                 default_spatial_kwargs: dict = None,
                 default_spectral_response: Callable = None,
                 default_spectral_kwargs: dict = None,
                 default_noise_function: Callable = None,
                 default_noise_kwargs: dict = None
                 ) -> None:
        """
        Parameters
        ----------
        spatial_unit : {'rad','deg','arcmin','arcsec'}, default='rad'
            The units in which the spatial response function is defined
        spectral_unit : {'Hz','GHz','m','mm','um'}, default='Hz'
            The units in which the spectral response function is defined
        flux_unit : {'Jy','mJy'}, default='Jy'
            The units in which the the noise_function provides noise for the
            detector and in which the signal in any attached fields is defined
        best_spatial_res : float (optional)
            The best angular resolution of any detector to be attached to the
            instrument. This is used for checking that resolution of attached
            spectral cubes is adequate. If no value is given, no checks are
            performed. Should be specified in units matching spatial_unit
        best_spectral_res : float (optional)
            The best spectral resolution of any detector to be attached to the
            instrument. This is used for checking that resolution of attached
            spectral cubes is adequate. If no value is given, no checks are
            performed. Should be specified in units matching spectral_unit
        default_spatial_response : function, or one of 'gauss', 'spec-gauss'
        (optional)
            The default spatial response function to be used when adding
            detectors. Used when no spatial response is specified in the
            add_detector method; if none is given then a spatial response must
            be specified each time a detector is added. If 'gauss' a two
            dimensional gaussian will be used. The center (default zero) and
            width can be controlled by specifying the 'cx', 'cy', 'fwhmx',
            'fwhmy' in spatial_kwargs. If 'spec-gauss' a two dimensional
            gaussian that varries as a function of frequency will be used. The
            center (default zero) can be controlled by specifying the 'cx' and
            'cy' in spatial_kwargs. The width at a specific frequency are set by
            specifying 'freq0', 'fwhmx', and 'fwhmy'. The scaling of the size
            with frequency is set using the 'freq_exp' factor (default -1). The
            size at a given frequency will then be fwhmx_f = fwhmx *
            (freq0/freq)**freq_exp.
        default_spatial_kwargs : dict (optional)
            A dictionary of kwargs that will be passed to the
            default_spatial_response function. If not given, then any
            spatial_kwargs must be provided each time a detector is added.
        default_spectral_response : function, or one of 'gauss', 'boxcar'
        (optional)
            The default spectral response function to be used when adding
            detectors. Used when no spectral response is specified in the
            add_detector method; if none is given then a spectral response must
            be specified each time a detector is added. If 'gauss' a gaussian
            response will be used. The center frequency, width and amplitude can
            be controlled by specifying the 'freq0', 'fwhm', and 'A' in
            spectral_kwargs. If 'boxcar' a boxcar response will be used. The
            center frequency, width and amplitude can be controlled by
            specifying the 'freq0', 'fwhm', and 'A' in spectral_kwargs.
        default_spectral_kwargs : dict (optional)
            A dictionary of kwargs that will be passed to the
            default_spectral_response function. If not given, then any
            spectral_kwargs must be provided each time a detector is added.
        default_noise_function : function or 'white' or 'none' (optional)
            The default noise function to be used when adding detectors. Used
            when no noise is specified in the add_detector method; if none is
            given then a noise function must be specified each time a detector
            is added. See set_noise_functions method for details.
        default_noise_kwargs : dict (optional)
            A dictionary of kwargs that will be passed to the
            default_noise_function. If not given, then any spectral_kwargs must
            be provided each time a detector is added. If 'white' a default
            white noise function will be used. The rms for a 1 second sample and
            a DC offset can be specified using the 'rms' and 'bias' parameters
            in noise_kwargs.
        """

        # Tracking detectors
        self.detectors = {}
        self.detector_names = []
        self.detector_counter = 0

        # Tracking fields
        self.fields = {}
        self.field_props = {}
        self.field_names = []
        self.field_counter = 0

        # Tracking maps
        self.maps_consistent = False
        self.maps = {}
        self.noisemaps = {}


        # Collect response functions, assign the correct ones for some defaults
        # that can be entered as strings instead of functions. Verify that each
        # one is actually a function.
        self.set_spatial_respones('instrument',spatial_response=default_spatial_response,spatial_kwargs=default_spatial_kwargs)
        self.set_spectral_respones('instrument',spectral_response=default_spectral_response,spectral_kwargs=default_spectral_kwargs)
        self.set_noise_functions('instrument',noise_function=default_noise_function,noise_kwargs=default_noise_kwargs)

        self.best_spatial_res = best_spatial_res
        self.best_spectral_res = best_spectral_res

        # Deal with units
        _check_unit(spatial_unit,'spatial')
        self.spatial_unit = spatial_unit
        _check_unit(spectral_unit,'spectral')
        self.spectral_unit = spectral_unit
        _check_unit(flux_unit,'flux')
        self.flux_unit = flux_unit


    def _check_detectors(self,*detector_names: str):
        """Check a detectors or detectors exists"""
        for d in detector_names:
            if d not in self.detector_names:
                raise ValueError(f"detector {d} not found")
            
        # Check for repeats
        v, n = np.unique(detector_names, return_counts=True)
        if np.any(n>1):
            raise ValueError("Repeat references to a detector found")

    def _check_fields(self,*field_names: str):
        """Check a field or fields exists"""
        for f in field_names:
            if f not in self.fields:
                raise ValueError(f"field {f} not found")

        # Check for repeats
        v, n = np.unique(field_names, return_counts=True)
        if np.any(n>1):
            raise ValueError("Repeat references to a field found")


    def _set_response(self,
                      *detector_names: str,
                      resp_function: Callable,
                      resp_kwargs: dict,
                      mode: str
                      ) -> None:
        
        if mode not in ['spatial', 'spectral', 'noise']:
            raise ValueError('mode not recognized')
        
        # Check detector_names
        if len(detector_names) == 0:
            do_inst = False
            detector_names = self.detector_names
        else:
            do_inst = 'instrument' in detector_names
            detector_names = [d for d in detector_names if d != 'instrument']
            self._check_detectors(*detector_names)

        # Interpret defaults
        if mode == 'spatial':
            if resp_function in ['gauss','Gauss','gaussian','Gaussian']:
                resp_function = gauss_psf
            elif resp_function in ['specgauss','spec-gauss','SpecGauss','Spec-Gauss','spectral-gaussian','Spectral-Gaussian']:
                resp_function = gauss_psf_freq_dependent
            elif resp_function is None:
                if len(detector_names)>0:
                    raise ValueError("cannot set detector responses to None")
            elif not hasattr(resp_function, '__call__'):
                raise ValueError("spatial_response must have a __call__ method")
        if mode == 'spectral':
            if resp_function in ['gauss','Gauss','gaussian','Gaussian']:
                resp_function = gauss_response
            elif resp_function in ['box','boxcar','Box','Boxcar']:
                resp_function = boxcar_response
            elif resp_function is None:
                if len(detector_names)>0:
                    raise ValueError("cannot set detector responses to None")
            elif not hasattr(resp_function, '__call__'):
                raise ValueError("spectral_response must have a __call__ method")
        if mode == 'noise':
            if resp_function is None:
                if len(detector_names)>0:
                    raise ValueError("cannot set detector responses to None; did you mean 'none'?")
            elif resp_function in ['none','None']:
                resp_function = zero_noise
            elif resp_function in ['white','White']:
                resp_function = white_noise
            elif not hasattr(resp_function, '__call__'):
                raise ValueError("noise_function must have a __call__ method")


        # Check kwargs
        default_kwargs = resp_kwargs
        if resp_kwargs is None:
            resp_kwargs = {}
        elif not isinstance(resp_kwargs, dict):
            raise ValueError("kwargs must be a dict")

        # Update responses
        for d in detector_names:
            if mode == 'spatial':
                self.detectors[d].spatial_response = resp_function
                self.detectors[d].spatial_kwargs = resp_kwargs.copy()
            if mode == 'spectral':
                self.detectors[d].spectral_response = resp_function
                self.detectors[d].spectral_kwargs = resp_kwargs.copy()
            if mode == 'noise':
                self.detectors[d].noise_function = resp_function
                self.detectors[d].noise_kwargs = resp_kwargs.copy()

        # Update default
        if do_inst:
            if mode == 'spatial':
                self.default_spatial_response = resp_function
                self.default_spatial_kwargs = _dict_none_copy(default_kwargs)
            if mode == 'spectral':
                self.default_spectral_response = resp_function
                self.default_spectral_kwargs = _dict_none_copy(default_kwargs)
            if mode == 'noise':
                self.default_noise_function = resp_function
                self.default_noise_kwargs = _dict_none_copy(default_kwargs)

        self.maps_consistent = False

    def set_spatial_respones(self,
                             *detector_names: str,
                             spatial_response: Callable,
                             spatial_kwargs: dict = None) -> None:
        """Set a new spatial response function for a detector, set of detectors,
        or the instrument default

        When no detector_names are specified, spatial response of all detectors
        will be updated. To instead change the instrument's default spatial
        response, pass the string 'instrument' with the list of detector_names.
        Note that updating the instrument default will not automatically change
        the spatial response of existing detectors that were created using this
        default.

        Parameters
        ----------
        *detector_names : str
            One or more strings naming the detectors to modify. If 'instrument'
            is passed as one of these strings it will also change the
            instrument's default_spatial_response attribute.
        spatial_response : function, or one of 'gauss', 'spec-gauss'
            function that takes vectors of x offset, y offset (in spatial_unit),
            and frequency (in spectral_unit) and returns the spatial response of
            the detector. Additional function arguments can be as a dictionary
            of keyword arguments using spatial_kwargs. If 'gauss' a two
            dimensional gaussian will be used. The center (default zero) and
            width can be controlled by specifying the 'cx', 'cy', 'fwhmx',
            'fwhmy' in spatial_kwargs. If 'spec-gauss' a two dimensional
            gaussian that varries as a function of frequency will be used. The
            center (default zero) can be controlled by specifying the 'cx' and
            'cy' in spatial_kwargs. The width at a specific frequency are set by
            specifying 'freq0', 'fwhmx', and 'fwhmy'. The scaling of the size
            with frequency is set using the 'freq_exp' factor (default -1). The
            size at a given frequency will then be fwhmx_f = fwhmx *
            (freq0/freq)**freq_exp.
        spatial_kwargs : dict, optional
            A dictionary of kwargs that will be passed to the spatial_response
            function
        """
        
        self._set_response(*detector_names, resp_function=spatial_response, resp_kwargs=spatial_kwargs, mode='spatial')

    def set_spectral_respones(self,
                              *detector_names: str,
                              spectral_response: Callable,
                              spectral_kwargs: dict = None) -> None:
        """Set a new spectral response function for a detector, set of 
        detectors, or the instrument default

        When no detector_names are specified, spectral response of all detectors
        will be updated. To instead change the instrument's default spectral
        response, pass the string 'instrument' with the list of detector_names.
        Note that updating the instrument default will not automatically change
        the spectral response of existing detectors that were created using this
        default.

        Parameters
        ----------
        *detector_names : str
            One or more strings naming the detectors to modify. If 'instrument'
            is passed as one of these strings it will also change the
            instrument's default_spectral_response attribute.
        spectral_response : function, or one of 'gauss', 'boxcar'
            function that takes vectors of frequency (in spectral_unit) and
            returns the spectral response of the detector. Additional function
            arguments can be passed as a dictionary of keyword arguments using
            spectral_kwargs. If 'gauss' a gaussian response will be used. The
            center frequency, width and amplitude can be controlled by
            specifying the 'freq0', 'fwhm', and 'A' in spectral_kwargs. If
            'boxcar' a boxcar response will be used. The center frequency, width
            and amplitude can be controlled by specifying the 'freq0', 'fwhm',
            and 'A' in spectral_kwargs.
        spectral_kwargs : dict, optional
            A dictionary of kwargs that will be passed to the spectral_response
            function
        """

        self._set_response(*detector_names, resp_function=spectral_response, resp_kwargs=spectral_kwargs, mode='spectral')

    def set_noise_functions(self,
                            *detector_names: str,
                            noise_function: Callable,
                            noise_kwargs: dict = None) -> None:
        """Set a new noise function for a detector, set of detectors,
        or the instrument default

        When no detector_names are specified, noise functions of all detectors
        will be updated. To change the instrument's default noise function, pass
        the string 'instrument' with the list of detector_names. Note that
        updating the instrument default will not automatically change the noise
        function of existing detectors that were created using this default.

        Parameters
        ----------
        *detector_names : str
            One or more strings naming the detectors to modify. If 'instrument'
            is passed as one of these strings it will also change the
            instrument's default_noise_function attribute.
        noise_function : function, 'white', or 'none'
            function that takes an integer number of samples and a timestep, and
            returns a time series of noise - this can be used to add noise to a
            timestream. Additional function arguments can be passed as
            adictionary of keyword arguments using noise_kwargs. If 'white' a
            default white noise function will be used. The rms for a 1 second
            sample and a DC offset can be specified using the 'rms' and 'bias'
            parameters in noise_kwargs.
        noise_kwargs : dict, optional
            A dictionary of kwargs that will be passed to the noise_response
            function
        """
        
        self._set_response(*detector_names, resp_function=noise_function, resp_kwargs=noise_kwargs, mode='noise')


    def add_detector(self, 
                     name: str = None,
                     nominal_frequency: float = None,
                     spatial_response: Callable = None, 
                     spatial_kwargs: dict = None,
                     spectral_response: Callable = None, 
                     spectral_kwargs: dict = None,
                     noise_function: Callable = None, 
                     noise_kwargs: dict = None):
        """Add a new detector to the instrument

        No arguments are required, if none are given, the detector will have the
        default spatial response, spectral response, and noise functions set
        when initializing the Instrument. If a default for one of these
        parameters wasn't set, then it must be specified here.

        If a default spatial response is set for the Instrument, then providing
        a value for spatial_response in the method call will overwrite this.
        Providing no value for spatial_response, but providing a spatial_kwargs
        dictionary will result in the default Instrument spectral response
        function being evaluated, but with different parameters from the
        defaults. 

        The same is also true for spectral_response/spectral_kwargs, and
        noise_function/noise_kwargs.

        Parameters
        ----------
        name : str (optional)
            A name for the detector. If none is given this will be set to a
            number (starting with 0 for the first detector added and counting
            up).
        nominal_frequency : float (optional)
            Optionally specify a nominal frequency of the detector. If no value
            is given it will be guessed by evaulauting the spatial response
            function. This is not used for any importatn computations, just
            providing back information about the detector and deciding how to
            generate quicklook plots.
        spatial_response : function, or one of 'gauss', 'spec-gauss' (optional)
            function that takes vectors of x offset, y offset (in spatial_unit),
            and frequency (in spectral_unit) and returns the spatial response of
            the detector. Additional function arguments can be as a dictionary
            of keyword arguments using spatial_kwargs. If 'gauss' a two
            dimensional gaussian will be used. The center (default zero) and
            width can be controlled by specifying the 'cx', 'cy', 'fwhmx',
            'fwhmy' in spatial_kwargs. If 'spec-gauss' a two dimensional
            gaussian that varries as a function of frequency will be used. The
            center (default zero) can be controlled by specifying the 'cx' and
            'cy' in spatial_kwargs. The width at a specific frequency are set by
            specifying 'freq0', 'fwhmx', and 'fwhmy'. The scaling of the size
            with frequency is set using the 'freq_exp' factor (default -1). The
            size at a given frequency will then be fwhmx_f = fwhmx *
            (freq0/freq)**freq_exp.
        spatial_kwargs : dict, optional (optional)
            A dictionary of kwargs that will be passed to the spatial_response
            function
        spectral_response : function, or one of 'gauss', 'boxcar' (optional)
            function that takes vectors of frequency (in spectral_unit) and
            returns the spectral response of the detector. Additional function
            arguments can be passed as a dictionary of keyword arguments using
            spectral_kwargs. If 'gauss' a gaussian response will be used. The
            center frequency, width and amplitude can be controlled by
            specifying the 'freq0', 'fwhm', and 'A' in spectral_kwargs. If
            'boxcar' a boxcar response will be used. The center frequency, width
            and amplitude can be controlled by specifying the 'freq0', 'fwhm',
            and 'A' in spectral_kwargs.
        spectral_kwargs : dict, optional (optional)
            A dictionary of kwargs that will be passed to the spectral_response
            function
        noise_function : function, 'white', or 'none' (optional)
            function that takes an integer number of samples and a timestep, and
            returns a time series of noise - this can be used to add noise to a
            timestream. Additional function arguments can be passed as
            adictionary of keyword arguments using noise_kwargs. If 'white' a
            default white noise function will be used. The rms for a 1 second
            sample and a DC offset can be specified using the 'rms' and 'bias'
            parameters in noise_kwargs.
        noise_kwargs : dict, optional (optional)
            A dictionary of kwargs that will be passed to the noise_function
            function        
        """

        # Set name
        if name is None:
            name = str(self.detector_counter)
        if name in self.detector_names:
            raise ValueError("Name already in use - this may be because you specified a pre-existing name or because another detector has been given a numerical name that would have been used as a default")
        
        # Invalid case - no function and no default
        if spatial_response is None and self.default_spatial_response is None:
            raise ValueError("No default spatial response is set; one must be provided when initializing detector")
        if spectral_response is None and self.default_spectral_response is None:
            raise ValueError("No default spectral response is set; one must be provided when initializing detector")
        if noise_function is None and self.default_noise_function is None:
            raise ValueError("No default spectral response is set; one must be provided when initializing detector")

        # Invalid case - no function or keywords, default specified without keywords
        if spatial_response is None and spatial_kwargs is None and self.default_spatial_kwargs is None:
            raise ValueError("Default spatial response has no default keyword arguments, these muse be provided (specify an empty dictionary if no arguments are required)")
        if spectral_response is None and spectral_kwargs is None and self.default_spectral_kwargs is None:
            raise ValueError("Default spectral response has no default keyword arguments, these muse be provided (specify an empty dictionary if no arguments are required)")
        if noise_function is None and noise_kwargs is None and self.default_noise_kwargs is None:
            raise ValueError("Default noise function has no default keyword arguments, these muse be provided (specify an empty dictionary if no arguments are required)")

        # Case 1a: function specified here, kwargs not
        # Set kwargs to empty dictionary
        if spatial_response is not None:
            spatial_response = spatial_response
            if spatial_kwargs is None:
                spatial_kwargs = {}
        if spectral_response is not None:
            spectral_response = spectral_response
            if spectral_kwargs is None:
                spectral_kwargs = {}
        if noise_function is not None:
            noise_function = noise_function
            if noise_kwargs is None:
                noise_kwargs = {}
        
        # Case 2: function not specified
        # Use defaults
        if spatial_response is None:
            spatial_response = self.default_spatial_response
            if  spatial_kwargs is None:
                spatial_kwargs = self.default_spatial_kwargs
            else:
                spatial_kwargs = spatial_kwargs
        if spectral_response is None:
            spectral_response = self.default_spectral_response
            if  spectral_kwargs is None:
                spectral_kwargs = self.default_spectral_kwargs
            else:
                spectral_kwargs = spectral_kwargs
        if noise_function is None:
            noise_function = self.default_noise_function
            if  noise_kwargs is None:
                noise_kwargs = self.default_noise_kwargs
            else:
                noise_kwargs = noise_kwargs
            
        # Make a detector
        det = _DetectorInfo(name=name)
        self.detectors[name] = det
        self.detector_names.append(name)
        cp_maps_consistent = self.maps_consistent

        try:
            self.set_spatial_respones(name, spatial_response=spatial_response, spatial_kwargs=spatial_kwargs)
            self.set_spectral_respones(name, spectral_response=spectral_response, spectral_kwargs=spectral_kwargs)
            self.set_noise_functions(name, noise_function=noise_function, noise_kwargs=noise_kwargs)
        except:
            self.del_detectors(name)
            self.maps_consistent = cp_maps_consistent
            raise

        # Figure out reffreq
        if nominal_frequency is not None:
            self.detectors[name].reffreq = nominal_frequency
        else:
            try:
                xlim1 = -20
                xlim2 = 20
                xmaxlast = -np.inf
                delta = 1

                while delta > 0.01:
                    x = np.logspace(xlim1,xlim2,50001)
                    y = self.detectors[name].spectral_response(x,**self.detectors[name].spectral_kwargs)
                    xmax = x[np.nanargmax(y)]
                    
                    span = xlim2-xlim1
                    # If max is at the edge, give up
                    if xmax == x.min() or xmax == x.max():
                        warnings.warn("auto-determination of response peak failed")
                        xmax = np.nan
                        break
                    # Else zoom in window and re-do - zooms by factor of 2
                    else:
                        xlim1 = np.log10(xmax) - span/4
                        xlim2 = np.log10(xmax) + span/4
                    
                        delta = np.abs(xmaxlast-xmax)
                        xmaxlast = xmax

                self.detectors[name].reffreq = xmax
            except:
                warnings.warn("auto-determination of response peak failed")
                self.detectors[name].reffreq = np.nan


        self.detector_counter += 1
        self.maps_consistent = False

    def del_detectors(self, *detector_names: str) -> None:
        """Remove one or more detectors from the instrument's detector listing
        
        Parameters
        ----------
        *detector_names : str
            One or more strings naming the detectors to remove
        """

        self._check_detectors(*detector_names)

        for d in detector_names:
            self.detectors.pop(d)
            self.detector_names.remove(d)

        if len(detector_names) > 0:
            self.maps_consistent = False

    def add_field(self, grid: Grid, 
                  name: str = None, 
                  field_property_idx: int = None, 
                  in_place: bool = True,
                  _check: bool = True) -> None:
        """Add a field for the instrument to observe
        
        Provide a simim.map.Grid instance representing the field to be observed
        by the instrument. The Grid instance should generally be free from
        pre-applied observational effects (these are handled by the Instrument
        object itself), and should be pixelized at a higher resolution than the
        best resolution of the instrument elements

        Multiple calls to this method will add multiple fields that can be
        accessed based on their assigned name. For methods which apply to a
        specific field the default behavior is to use the most recently added
        field.

        Parameters
        ----------
        grid : simim.map.Grid
            A 3d simim.map.Grid grid that contains spatial dimensions in its
            first two axes and the spectral dimension in its third.
        name : str (optional)
            Name for the field, if none is given this will match the numerical
            index of the field.
        field_property_idx : int (optional)
            The property index of the grid property to map. Defaults to 0, and
            never needs to be set for grids containing only one property
        in_place : bool (optional)
            When True (default) a reference to the grid will be stored with the
            Instrument instance. When False a copy of the grid is stored. Note
            that Instrument methods don't transform the grid, so storing a copy
            is only necessary if the grid will be changed by external code in a
            way that shouldn't be reflected in the instrument modeling.
        _check : bool
            Flag for checking unit compatibility between grid and self.
        """

        if field_property_idx is None:
            field_property_idx = 0
        if field_property_idx > grid.n_dimensions-1 or field_property_idx < 0:
            raise ValueError(f"grid does not have property {field_property_idx}")
        if _check:
            _check_grid_detector_compatibility(self,grid,field_property_idx)

        if name is None:
            name = str(self.field_counter)
        if name in self.fields:
            raise ValueError("Name already in use - this may be because you specified a pre-existing name or because a field has been given a numerical name that would have been used as a default")

        if self.best_spatial_res is not None:
            if grid.pixel_size[0] > self.best_spatial_res/5 or grid.pixel_size[1] > self.best_spatial_res/5:
                warnings.warn("grid pixel size is large relative to spatial resolution")
            if grid.pixel_size[0] > self.best_spatial_res or grid.pixel_size[1] > self.best_spatial_res:
                raise ValueError("grid pixel size is larger than spatial resolution")
        if self.best_spectral_res is not None:
            if grid.best_spectral_res[2] > self.best_spatial_res/5:
                warnings.warn("grid pixel size is large relative to spectral resolution")
            if grid.best_spectral_res[2] > self.best_spatial_res:
                raise ValueError("grid pixel size is larger than spectral resolution")

        if in_place:
            self.fields[name] = grid
            self.field_props[name] = field_property_idx
        else:
            self.fields[name] = grid.copy(properties=field_property_idx)
            self.field_props[name] = 0
        
        self.field_names.append(name)
        self.field_counter += 1

    def del_fields(self, *field_names: str) -> None:
        """Remove one or more fields from the instrument's field listing
        
        Parameters
        ----------
        *field_names : str
            One or more strings naming the fields to remove
        """
        
        self._check_fields(*field_names)

        for f in field_names:
            self.fields.pop(f)
            self.field_props.pop(f)
            self.field_names.remove(f)

            if f in self.maps:
                self.maps.pop(f)
                self.maps_consistent.pop(f)

### !
#     def map_field(self,kernel_size=None,spatial_response_norm='peak',beam=None,pad=None,_check=True):
#         # Either speciffy kernel_size and spatial_response_norm or specify beam (with a grid containing the beam)
#         # Note that beams provided using beam will not be re-normalized.

#         if beam is not None and kernel_size is not None:
#             raise ValueError("If beam is specified, kernel_size should be left as None")

#         # Set off so it only has to be done once when an Instrument object iterates over many detectors
#         if _check:
#             if not self.field_initialized:
#                 raise ValueError("No field specified - cannot map field")

#         if beam is None:
#             beam = self._setup_beam(kernel_size=None,spatial_response_norm=spatial_response_norm)
                    
#         if _check and beam.grid.ndim != 4:
#             raise ValueError("Beam does not have three dimensions")

#         # Convolve with beam
#         self.map = self.field.convolve(beam,ax=(0,1),in_place=False,pad=pad)

#         # Evaluate spectral response over frequency axis
#         spec = self.spectral_response(beam.axes_centers[2],**self.spectral_kwargs)

#         # Collapse along the third dimension using spectral response
#         self.map = self.map.collapse_dimension(ax=2,weights=spec,mode='average',in_place=True)
#         self.map_initialized = True

    def map_fields(self, 
                   *field_names: Grid,
                   kernel_size: int = None,
                   spatial_response_norm: str = 'peak',
                   beam: Grid = None,
                   pad: int = None):
        """Generate maps of the specified fields based on instruemnt
        response settings

        Parameters
        ----------
        *field_names : str
            One or more strings naming the fields to remove, if no names
            are specified all fields will be mapped
        """

        # Check detector_names
        if len(detector_names) == 0:
            detector_names = self.detector_names
    


        # Either speciffy kernel_size and spatial_response_norm or specify beam (with a grid containing the beam)
        # Note that beams provided using beam will not be re-normalized.

        if beam is not None and kernel_size is not None:
            raise ValueError("If beam is specified, kernel_size should be left as None")



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



        self.maps_consistent[fieldname] = True




    @pltdeco
    def plot_detector_response(self, 
                               *detector_names: str, 
                               xmin: float = None, xmax: float = None, ymin: float = None, ymax: float = None,
                               fmin: float = None, fmax: float = None, fspatial: float = None, 
                               figsize: tuple = (10,5)):
        """Plot the spatial and spectral response functions for one or more
        detectors
        
        Parameters
        ----------
        *detector_names : str
            One or more detector names to plot. If no names are given, all
            detectors will be plotted.
        fmin, fmax : float (optional)
            The minimum and maximum frequency to plot the spectrum over, if None
            given it +/-10% of the detector representative frequency will be used
        fspatial : float (optional)
            The frequency at which to evaluate the detector spatial response.
            If None given the detector representative frequency will be used
        xmin, xmax, ymin, ymax : float
            The minimum and maximum spatial coordinates defining the region over
            which to plot the detector spatial response. If None given the 5 
            times the instrument best_spatial_resolution will be used.
        figsize : tuple
            Size of the resulting plots
        """
    
        if len(detector_names) == 0:
            detector_names = self.detector_names
        else:
            self._check_detectors(*detector_names)

        if fmin is None:
            fmin = 0.9 * np.min([self.detectors[k].reffreq for k in detector_names])
        if fmax is None:
            fmax = 1.1 * np.max([self.detectors[k].reffreq for k in detector_names])
        if xmin is None:
            xmin = -5 * self.best_spatial_res
        if xmax is None:
            xmax = 5 * self.best_spatial_res
        if ymin is None:
            ymin = -5 * self.best_spatial_res
        if ymax is None:
            ymax = 5 * self.best_spatial_res

        for dn in detector_names:
            d = self.detectors[dn]
            if fspatial is None:
                fs_d = d.reffreq
            else:
                fs_d = fspatial

            fig,axes = plt.subplots(1,2,figsize=figsize)
            fig.subplots_adjust(left=.1,right=.95,top=.9,bottom=.15,wspace=.35)
            fig.suptitle(f"Detector: {dn}")
            axes[0].set(xlabel=f'Frequency [{self.spectral_unit}]',ylabel='Response')
            axes[1].set(aspect='equal',xlabel=f'dx [{self.spatial_unit}]',ylabel=f'dy [{self.spatial_unit}]')

            f = np.linspace(fmin,fmax,1000)
            axes[0].plot(f,d.spectral_response(f,**d.spectral_kwargs))
            axes[0].axvline(fs_d,color='k',ls='--')

            x = np.linspace(xmin,xmax,100)
            y = np.linspace(ymin,ymax,100)
            axes[1].pcolor(x,y,d.spatial_response(x,y,fs_d,**d.spatial_kwargs)[:,:,0],vmin=0)

        plt.show()

    @pltdeco
    def plot_spectral_response(self,
                               fmin: float = None, fmax: float = None,
                               figsize: tuple = (5,5)) -> None:
        """Plot the spectral response of every detector
        
        Parameters
        ----------
        fmin, fmax : float (optional)
            The minimum and maximum frequency to plot the spectrum over.
        figsize : tuple (optional)
            Size of the resulting plots
        """
    
        if fmin is None:
            fmin = 0.9 * np.min([d.reffreq for d in self.detectors.values])
        if fmax is None:
            fmax = 1.1 * np.max([d.reffreq for d in self.detectors.values])

        fig,axes = plt.subplots(1,1,figsize=figsize)
        fig.subplots_adjust(left=.1,right=.95,top=.95,bottom=.15,wspace=.35)
        axes.set(xlabel=f'Frequency [{self.spectral_unit}]',ylabel='Response')

        f = np.linspace(fmin,fmax,5000)

        spectype = _specunit_type(self.spectral_unit)
        if spectype == 'freq':
            cmap = colormaps['Spectral']
        elif spectype == 'wl':
            cmap = colormaps['Spectral_r']
        else: 
            cmap = colormaps['Spectral']

        order = np.argsort(self.detector_freqs)
        for i in np.argsort(self.detector_freqs):
            d = self.detectors[self.detector_names[i]]
            axes.plot(f, d.spectral_response(f, **d.spectral_kwargs), color=cmap(i/(len(self.detectors)-.99)))

        plt.show()


# #### OLD VERSIONS:

# class Detector():
#     def _setup_beam(self,kernel_size=None,spatial_response_norm='peak'):

#         if kernel_size is None:
#             kernel_size = np.copy(self.field.side_length[:2])
#         kernel_size = np.array(kernel_size)
#         if len(kernel_size) != 2:
#             raise ValueError("kernel_size must have length 2 (xsize, ysize)")
        
#         # Check that the value of norm is recognized
#         if spatial_response_norm not in ['peak','area','none']:
#             raise ValueError("spatial_response_norm must be one of 'peak', 'area', 'none'")
        
#         # Set up axes of beam convolution kernel
#         center = [0,0,self.field.center_point[2]]
#         side_length = [kernel_size[0],kernel_size[1],self.field.side_length[2]]

#         # Evaluate beam response over kernel axes
#         beam = Grid(n_properties=1,center_point=center,side_length=side_length,pixel_size=self.field.pixel_size,axunits=self.field.axunits,gridunits=self.field.gridunits)
#         beam.init_grid()

#         shape = beam.grid.shape
#         beam.grid = np.expand_dims(self.spatial_response(beam.axes_centers[0],beam.axes_centers[1],beam.axes_centers[2],**self.spatial_kwargs),-1)
#         if beam.grid.shape != shape:
#             raise ValueError("spatial_response does not produce an array of the correct shape")

#         if spatial_response_norm == 'peak':
#             beam.grid = beam.grid / np.max(beam.grid,axis=(0,1,3)).reshape(1,1,-1,1)
#         if spatial_response_norm == 'area':
#             beam.grid = beam.grid / np.sum(beam.grid,axis=(0,1,3)).reshape(1,1,-1,1)
#         if spatial_response_norm == 'none':
#             pass

#         return beam

#     def map_field(self,kernel_size=None,spatial_response_norm='peak',beam=None,pad=None,_check=True):
#         # Either speciffy kernel_size and spatial_response_norm or specify beam (with a grid containing the beam)
#         # Note that beams provided using beam will not be re-normalized.

#         if beam is not None and kernel_size is not None:
#             raise ValueError("If beam is specified, kernel_size should be left as None")

#         # Set off so it only has to be done once when an Instrument object iterates over many detectors
#         if _check:
#             if not self.field_initialized:
#                 raise ValueError("No field specified - cannot map field")

#         if beam is None:
#             beam = self._setup_beam(kernel_size=None,spatial_response_norm=spatial_response_norm)
                    
#         if _check and beam.grid.ndim != 4:
#             raise ValueError("Beam does not have three dimensions")

#         # Convolve with beam
#         self.map = self.field.convolve(beam,ax=(0,1),in_place=False,pad=pad)

#         # Evaluate spectral response over frequency axis
#         spec = self.spectral_response(beam.axes_centers[2],**self.spectral_kwargs)

#         # Collapse along the third dimension using spectral response
#         self.map = self.map.collapse_dimension(ax=2,weights=spec,mode='average',in_place=True)
#         self.map_initialized = True

#     def sample(self,positions,dt,properties=None,signal=True,noise=True):
                
#         if signal and noise:
#             self.sample_signal(positions,properties=properties)
#             self.sample_noise(len(positions),dt)
#             self.timestream = self.timestream_signal + self.timestream_noise.reshape(-1,1)
#         elif signal:
#             self.sample_signal(positions,properties=properties)
#             self.timestream = self.timestream_signal
#         elif noise:
#             self.sample_noise(len(positions),dt)
#             self.timestream = self.timestream_noise.reshape(-1,1)
#         self.timestream_positions = positions
#         self.timestream_dt = dt

#         return self.timestream

#     def sample_signal(self,positions,properties=None):
#         if not self.map_initialized:
#             raise ValueError("No map initialized, cannot sample")
        
#         self.timestream_signal = self.map.sample(positions=positions,properties=properties)
#         return self.timestream_signal

#     def sample_noise(self,nsamples,dt):
#         self.timestream_noise = self.noise_function(nsamples,dt,**self.noise_kwargs)
#         return self.timestream_noise

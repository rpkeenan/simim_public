import numpy as np
import warnings
from simim.map.gridder import _grid

def _check_shape(x):
    if not isinstance(x,np.ndarray):
        try:
            x = np.array(x,ndmin=1)
        except:
            raise ValueError("input axis could not cast to numpy ndarray")
        
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        raise ValueError("input axis must be a one dimensional numpy ndarray")
    
    return x

known_spectral_units = ['GHz','Hz','m','mm','um']
known_spatial_units = ['rad','deg','arcmin','arcsec']
known_flux_units = ['Jy','mJy']
def _check_unit(unit,axtype):
    if axtype == 'spectral':
        known_units = known_spectral_units
    elif axtype == 'spatial':
        known_units = known_spatial_units
    elif axtype == 'flux':
        known_units = known_flux_units
    else:
        raise ValueError("axtype {} not recognized".format(axtype))

    if unit not in known_units:
        raise ValueError("{} axis unit '{}' not recognized".format(axtype,unit))

def _check_grid_detector_compatibility(detector,grid,field_property_idx):
    if not isinstance(grid,_grid):
        raise ValueError("Must specify an simim grid object for mapping")
    if not grid.grid_active:
        raise ValueError("Specified grid does not contain data")
    if np.any(grid.fourier_space):
        raise ValueError("Grid has some axes in Fourier space, mapping requires all axes be in map space")
    if grid.n_dimensions != 3:
        raise ValueError("Grid is not three dimensional")

    # Verify that all axis units are matched (if none specified in grid, provide a warning)
    # Convert any units that need converting
    if grid.axunits[0] is None:
        warnings.warn("Grid unit for x axis not specified. Assuming it matches detector spatial unit")
    else:
        try: 
            _check_unit(grid.axunits[0],'spatial')
        except:
            raise ValueError("grid unit for x axis not recognized")
        if grid.axunits[0] != detector.spatial_unit:
            raise ValueError("Grid and detector x axis units don't match, unit conversion has not been implemented")

    if grid.axunits[1] is None:
        warnings.warn("Grid unit for y axis not specified. Assuming it matches detector spatial unit")
    else:
        try: 
            _check_unit(grid.axunits[1],'spatial')
        except:
            raise ValueError("grid unit for y axis not recognized")
        if grid.axunits[1] != detector.spatial_unit:
            raise ValueError("Grid and detector y axis units don't match, unit conversion has not been implemented")

    if grid.axunits[2] is None:
        warnings.warn("Grid unit for spectral axis not specified. Assuming it matches detector spatial unit")
    else:
        try: 
            _check_unit(grid.axunits[2],'spectral')
        except:
            raise ValueError("grid unit for spectral axis not recognized")
        if grid.axunits[2] != detector.spectral_unit:
            raise ValueError("Grid and detector spectral axis units don't match, unit conversion has not been implemented")

    # Check flux units
    if grid.gridunits is None:
        warnings.warn("Grid flux unit not specified. Assuming it matches detector flux unit")
    else:
        try: 
            _check_unit(grid.gridunits,'flux')
        except:
            raise ValueError("grid flux unit not recognized")
        if grid.gridunits[field_property_idx] != detector.flux_unit:
            raise ValueError("Grid and detector flux units don't match, unit conversion has not been implemented")

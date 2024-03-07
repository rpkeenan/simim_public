import numpy as np
from simim.instrument._helpers import _check_shape

# Inputs must be the axes of a grid (ie dx coords, dy coords, frequency coords)
def gauss_psf(dx,dy,freq,fwhmx=1,fwhmy=1,cx=0,cy=0):
    """Produce a Gaussian PSF for Instrument class
    
    Parameters
    ----------
    dx : 1d array
        Array of x positions (ra offsets)
    dy : 1d array
        Array of y positions (dec offsets)
    freq : 1d array
        Array of z positions (frequencies)
    fwhmx, fwhmy : float
        FWHM of beam in ra and dec directions
    cx, cy : floats
        Centers of the beam in ra and dec offset dimensions (defaults to 0, 0)

    Returns
    -------
    beam
        An array of shape (len(dx),len(dy),len(freq)) containing the beam evaluated
        at every point in the resulting spatial grid
    """
    dx = _check_shape(dx)
    dy = _check_shape(dy)
    freq = _check_shape(freq)

    x = (dx-cx).reshape(-1,1,1)
    y = (dy-cy).reshape(1,-1,1)
    z = np.ones((1,1,len(freq)))
    response = np.exp(-4*np.log(2)*(x**2/fwhmx**2 + y**2/fwhmy**2)) * z
    return response

def gauss_psf_freq_dependent(dx,dy,freq,freq0=100e9,fwhmx=1,fwhmy=1,cx=0,cy=0,freq_exp=-1):
    """Produce a Gaussian PSF with a variable width as a function of frequency
    
    Parameters
    ----------
    dx : 1d array
        Array of x positions (ra offsets)
    dy : 1d array
        Array of y positions (dec offsets)
    freq : 1d array
        Array of z positions (frequencies)
    freq0 : float
        Frequency value at which fwhmx and fwhmy are specified
    fwhmx, fwhmy : float
        FWHM of beam in ra and dec directions at frequency fwhm0
    cx, cy : floats
        Centers of the beam in ra and dec offset dimensions (defaults to 0, 0)
    freq_exp : float
        FWHM will scale by (freq0/freq)^freq_exp. Defaults to -1, correct for a
        typical beam and z axis in frequency (rather than wavelength) units

    Returns
    -------
    beam
        An array of shape (len(dx),len(dy),len(freq)) containing the beam
        evaluated at every point in the resulting spatial grid
    """
    dx = _check_shape(dx)
    dy = _check_shape(dy)
    freq = _check_shape(freq)

    x = (dx-cx).reshape(-1,1,1)
    y = (dy-cy).reshape(1,-1,1)
    fwhmx_freq = (fwhmx * (freq/freq0)**-freq_exp).reshape(1,1,-1)
    fwhmy_freq = (fwhmy * (freq/freq0)**-freq_exp).reshape(1,1,-1)
    response = np.exp(-4*np.log(2)*(x**2/fwhmx_freq**2 + y**2/fwhmy_freq**2))
    return response

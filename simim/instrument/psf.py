import numpy as np
from simim.instrument.helpers import _check_shape

# Inputs must be the axes of a grid (ie dx coords, dy coords, frequency coords)
def gauss_psf(dx,dy,freq,fwhmx=1,fwhmy=1,cx=0,cy=0,det_xoffset=0,det_yoffset=0):
    dx = _check_shape(dx)
    dy = _check_shape(dy)
    freq = _check_shape(freq)

    x = (dx-cx).reshape(-1,1,1)
    y = (dy-cy).reshape(1,-1,1)
    z = np.ones((1,1,len(freq)))
    response = np.exp(-4*np.log(2)*(x**2/fwhmx**2 + y**2/fwhmy**2)) * z
    return response

def gauss_psf_freq_dependent(dx,dy,freq,freq0=100,fwhmx=1,fwhmy=1,cx=0,cy=0,freq_exp=-1,det_xoffset=0,det_yoffset=0):
    dx = _check_shape(dx)
    dy = _check_shape(dy)
    freq = _check_shape(freq)

    x = (dx-cx).reshape(-1,1,1)
    y = (dy-cy).reshape(1,-1,1)
    fwhmx_freq = (fwhmx * (freq/freq0)**-freq_exp).reshape(1,1,-1)
    fwhmy_freq = (fwhmy * (freq/freq0)**-freq_exp).reshape(1,1,-1)
    response = np.exp(-4*np.log(2)*(x**2/fwhmx_freq**2 + y**2/fwhmy_freq**2))
    return response

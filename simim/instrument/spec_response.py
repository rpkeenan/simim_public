import numpy as np
from simim.instrument.helpers import _check_shape

def gauss_response(freq,A=1,freq0=100,fwhm=1):
    freq = _check_shape(freq)
    response = A * np.exp(-4*np.log(2)*((freq-freq0)**2/fwhm**2))
    return response

def boxcar_response(freq,A=1,freq0=100,width=1):
    freq = _check_shape(freq)
    response = np.zeros(freq.shape)
    response[(freq>=freq0-width/2) & (freq<=freq0+width/2)] = A
    return response
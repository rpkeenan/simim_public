import numpy as np

def zero_noise(Nsamples: int, dt: float, rng=None):
    """Returns all zeros for noise array
    
    Looks like other noise functions for compatibility, ease of use
    as a default. Built into Instrument module as the "none" option
    for noise_functions.

    Parameters
    ----------
    Nsamples : int
        Number of noise samples to draw
    dt : float
        Timestep over which to apply noise parameters (ignored)
    rng : numpy rng instance or None
        A random number generator to use (ignored)

    Returns
    -------
    noise_samples
        Array of zeros with length Nsamples
    """
    return np.zeros(Nsamples)

def white_noise(Nsamples,dt,rms=1,bias=0,rng=None):
    """Function to generate white noise
    
    Generate white noise. Parameters should specify the timestep 
    in seconds, the RMS noise in a 1 second interval, and a bias
    term which sets the center of the distribution for noise draws.
    Noise draws will be distributed as N(rms/sqrt(dt)) + bias.

    This is a built in option for Instrument instances when specifying
    noise_functions, accessible with the keyword "white".

    Parameters
    ----------
    Nsamples : int
        Number of noise samples to draw
    dt : float
        Timestep over which to apply noise parameters (in seconds)
    rms : float
        RMS noise fluctuation in a 1 second period
    bias : float
        Constant offset to apply to all noise draws
    rng : numpy rng instance or None
        A random number generator to use

    Returns
    -------
    noise_samples
        Array of noise draws with length Nsamples
    """

    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(loc=bias,scale=rms/np.sqrt(dt),size=Nsamples)
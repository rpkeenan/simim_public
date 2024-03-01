import numpy as np

default_rng = np.random.default_rng()

def zero_noise(Nsamples,dt,rng=default_rng):
    return np.zeros(Nsamples)

def white_noise(Nsamples,dt,rms=1,bias=0,rng=default_rng):
    return rng.normal(loc=bias,scale=rms/np.sqrt(dt),size=Nsamples)
'''Code for generating noise parameters of an observation

1. noise_from_T_N - generate rms noise from observation noise temperature
2. T_N_from_NEP - convert from NEP to noise temperature in K units
3. noise_from_NEP - generate rms noise from NEP
4. visibility_noise_from_T_N - generate rms noise in a visibility from noise temp

Updated 7.2.19
Version 6.0

Version 6.0 --- version 6.0 includes observation functions
'''

from numpy import sqrt

k_B = 1.3806485e-23

################################################################################
######################### SECTION 1: Noise Conversions #########################
################################################################################

'''NOISE_FROM_T_N:
Function to compute receiver rms noise (in K) given observation noise temp for
a map space observation with a single dish telescope

Required arguments:
T_N - the noise temperature of the observation in K
int_time - the integration time of for a single pointing (or pixel)
dnu - the frequency channel widht of the observation

Optional arguments:
npol (= 1) - the number of polarizations of the receiver
npoint (= 1) - the number of repeated pointings coadded to produce a pixel

Returns:
sigma_N - the rms noise of the observation in K
'''

def noise_from_T_N(T_N, int_time, dnu, npol=1, npoint=1):
    return T_N / sqrt(int_time * dnu * npol * npoint)



'''T_N_FROM_NEP:
Function to compute noise temperature (in K) given NEP (in W / sqrt(Hz))

Required arguments:
NEP - the Noise Equivalent Power in W / sqrt(Hz)
efficiency - the optical efficiency of the system (or 1 if optical NEP is used)
dnu - the frequency channel widht of the system

Returns:
T_N - the noise temperature of the system in K
'''

def T_N_from_NEP(NEP, efficiency, dnu):
    return NEP / (k_B * efficiency * sqrt(2*dnu))



'''NOISE_FROM_NEP:
Function to compute rms noise (in K) given NEP (in W / sqrt(Hz)). Wrapper around
T_N_FROM_NEP and NOISE_FROM_T_N.

Required arguments:
NEP - the Noise Equivalent Power in W / sqrt(Hz)
efficiency - the optical efficiency of the system (or 1 if optical NEP is used)
int_time - the integration time of for a single pointing (or pixel)
dnu - the frequency channel widht of the system

Optional arguments:
npol (= 1) - the number of polarizations of the receiver
npoint (= 1) - the number of repeated pointings coadded to produce a pixel

Returns:
sigma_N - the rms noise of the observation in K
'''

def noise_from_NEP(NEP, efficiency, int_time, dnu, npol=1, npoint=1):
    T_N = T_N_from_NEP(NEP, efficency, dnu)
    return noise_from_T_N(T_N, int_time, dnu, npol, npoint)



'''VISIBILITY_NOISE_FROM_T_N:
Function to compute visibility rms noise (in K) given observation noise temp for
a uv space observation with an interferometer

Required arguments:
T_N - the noise temperature of the observation in K
int_time - the integration time of for a single pointing
dnu - the frequency channel widht of the observation
omega_beam - the primary beam size in str

Optional arguments:
npol (= 1) - the number of polarizations of the receiver

Returns:
V_N - the rms noise of the visibility in K
'''

def visibility_noise_from_T_N(T_N, int_time, dnu, omega_beam, npol=1):
    return omega_beam * T_N / sqrt(int_time * dnu * npol)

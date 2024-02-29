import numpy as np

from simim.galprops.log10normal import log10normal
import simim.constants as const

def yang22(mass,redshift,line='CII',
           fduty=True,
           scatter=True, fix_scatter=None,
           rng=np.random.default_rng()):
    """Compute CII, CI, and CO luminosities based on the fits of Yang et al.
    2022. 

    Parameters
    ----------
    mass : float or array
        The mass of the halo(s), in Msun
    redshift : float or array
        The redshift of the halo(s)
    line: str
        One of 'CII', 'CI1-0', 'CI2-1', 'CO1-0', 'CO2-1', 'CO3-2', 'CO4-3', 'CO5-4'
    fduty : bool, optional
        Toggles duty fraction on or off, default is on (True)
    scatter : bool, optional
        Toggles scattering on or off, default is on (True)
    fix_scatter : None or float, optional
        If scatter is True, the default behavior is to use the scatter from Yang et al.
        model to determine the width of the scatter applied. Alternatively
        the fix_scatter parameter can be set to a different scatter value. The value
        should be specified in dex.
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_line : float or array
        The assigned line luminosities in Lsun
    """

    if line not in ['CII', 'CI1-0', 'CI2-1', 'CO1-0', 'CO2-1', 'CO3-2', 'CO4-3', 'CO5-4']:
        raise ValueError("line not recognized")
    elif line == 'CII':
        m1, n, a, b, sig = _yang22_cii_fit(redshift)
    elif line in ['CI1-0','CI2-1']:
        m1, n, a, b, sig = _yang22_ci_fit(redshift,line)
    elif line in ['CO1-0', 'CO2-1', 'CO3-2', 'CO4-3', 'CO5-4']:
        m1, n, a, b, sig = _yang22_co_fit(redshift,line)

    if fix_scatter is not None:
        sig = fix_scatter

    m1 = 10**m1
    n = 10**n
    L = 2 * n * mass * ((mass/m1)**-a + (mass/m1)**b)**-1

    if scatter:
        L = log10normal(L, sig, preserve_linear_mean=False, rng=rng)

    if fduty:
        f = _yang22_fduty(mass, redshift)
        L[np.random.rand(len(L))>f] = 0

    return L

### All of this is just implementing the gust of the models
def _yang22_cii_fit(redshift):
    """Fit parameters for CII - return as a function of redshift"""
    return_scalar = np.isscalar(redshift)
    redshift = np.array(redshift,ndmin=1)

    m1 = np.zeros(len(redshift))
    m1[redshift<=0.0] = 12.11
    m1[(redshift<4.0) & (redshift>0.0)] = (12.11 * redshift**-0.04105)[(redshift<4.0) & (redshift>0.0)]
    m1[(redshift<5.0) & (redshift>=4.0)] = (8.69 + 1.26*redshift - 0.143*redshift**2)[(redshift<5.0) & (redshift>=4.0)]
    m1[redshift>=5] = (11.92 - 0.1068*redshift)[redshift>=5]

    n = np.zeros(len(redshift))
    n[redshift<4.0] = (-0.907*np.exp(-redshift/0.867) - 3.04)[redshift<4.0]
    n[(redshift<5.0) & (redshift>=4.0)] = (-5.467 + 1.056*redshift - 0.1133*redshift**2)[(redshift<5.0) & (redshift>=4.0)]
    n[redshift>=5] = (-2.37 - 0.130*redshift)[redshift>=5]

    a = np.zeros(len(redshift))
    a[redshift<4.0] = (1.35 + 0.450*redshift - 0.0805*redshift**2)[redshift<4.0]
    a[(redshift<5.0) & (redshift>=4.0)] = (6.135 - 1.786*redshift + 0.1837*redshift**2)[(redshift<5.0) & (redshift>=4.0)]
    a[redshift>=5] = (2.82 - 0.298*redshift + 0.0196*redshift**2)[redshift>=5]

    b = np.zeros(len(redshift))
    b[redshift<4.0] = (2.57 * np.exp(-redshift/1.55) + 0.0575)[redshift<4.0]
    b[(redshift<5.0) & (redshift>=4.0)] = (0.443 - 0.0521*redshift)[(redshift<5.0) & (redshift>=4.0)]
    b[redshift>=5] = (1760 * np.exp(-redshift/0.520) + 0.0782)[redshift>=5]

    sig = 0.32 * np.exp(-redshift/1.5) + 0.18

    if return_scalar:
        return m1[0], n[0], a[0], b[0], sig
    return m1, n, a, b, sig

_yang22_co_params = np.zeros(5,dtype=[('name','U5'),
                                     ('m1a','f'),('m1b','f'),('m4a','f'),('m4b','f'),('m5a','f'),('m5b','f'),
                                     ('n1a','f'),('n1b','f'),('n1c','f'),('n4a','f'),('n4b','f'),('n4c','f'),('n5a','f'),('n5b','f'),
                                     ('a1a','f'),('a1b','f'),('a1c','f'),('a4a','f'),('a4b','f'),('a4c','f'),('a5a','f'),('a5b','f'),('a5c','f'),
                                     ('b1a','f'),('b1b','f'),('b1c','f'),('b4a','f'),('b4b','f'),('b5a','f'),('b5b','f'),('b5c','f'),
                                     ('siga','f'),('sigb','f'),('sigc','f')
                                     ])
_yang22_co_params['name'] = np.array(['CO1-0','CO2-1','CO3-2','CO4-3','CO5-4'])
_yang22_co_params['m1a'] = np.array([12.13,12.12,12.1,12.09,12.12])
_yang22_co_params['m1b'] = np.array([-0.1678,-0.1704,-0.171,-0.1662,-0.1796])
_yang22_co_params['m4a'] = np.array([11.75,11.74,11.74,11.73,11.73])
_yang22_co_params['m4b'] = np.array([-0.06833,-0.07050,-0.07228,-0.07457,-0.07789])
_yang22_co_params['m5a'] = np.array([11.63,11.63,11.62,11.6,11.58])
_yang22_co_params['m5b'] = np.array([-0.04943,-0.05266,-0.0550,-0.0529,-0.05359])

_yang22_co_params['n1a'] = np.array([-6.855,-5.95,-5.53,-5.35,-5.37])
_yang22_co_params['n1b'] = np.array([0.2366,0.278,0.329,0.404,0.515])
_yang22_co_params['n1c'] = np.array([-0.05013,-0.0521,-0.0570,-0.0657,-0.0802])
_yang22_co_params['n4a'] = np.array([-6.554,-5.57,-5.06,-4.784,-3.81])
_yang22_co_params['n4b'] = np.array([-0.03725,-0.0250,-0.0150,0,-0.359])
_yang22_co_params['n4c'] = np.array([0,0,0,0,0.0419])
_yang22_co_params['n5a'] = np.array([-6.274,-5.26,-4.72,-4.40,-4.21])
_yang22_co_params['n5b'] = np.array([-0.09087,-0.0849,-0.0808,-0.0744,-0.0674])

_yang22_co_params['a1a'] = np.array([1.642,1.69,1.843,2.24,2.14])
_yang22_co_params['a1b'] = np.array([0.1663,0.126,0.08405,-0.0891,0.110])
_yang22_co_params['a1c'] = np.array([-0.03238,-0.0280,-0.02485,0,-0.0371])
_yang22_co_params['a4a'] = np.array([3.73,4.557,5.253,5.74,6.12])
_yang22_co_params['a4b'] = np.array([-0.833,-1.215,-1.502,-1.67,-1.77])
_yang22_co_params['a4c'] = np.array([0.0884,0.1300,0.1609,0.178,0.188])
_yang22_co_params['a5a'] = np.array([2.56,2.47,2.53,2.59,2.87])
_yang22_co_params['a5b'] = np.array([-0.223,-0.210,-0.220,-0.206,-0.257])
_yang22_co_params['a5c'] = np.array([0.0142,0.0132,0.0139,0.0120,0.0157])

_yang22_co_params['b1a'] = np.array([1.77,1.80,1.88,2.017,2.39])
_yang22_co_params['b1b'] = np.array([2.72,2.76,2.74,2.870,2.55])
_yang22_co_params['b1c'] = np.array([-0.0827,-0.0678,-0.0623,-0.1127,-0.0890])
_yang22_co_params['b4a'] = np.array([0.598,0.657,0.707,0.762,0.846])
_yang22_co_params['b4b'] = np.array([-0.0710,-0.0794,-0.0879,-0.0984,-0.115])
_yang22_co_params['b5a'] = np.array([33.4,38.3,31.5,41.6,21.8])
_yang22_co_params['b5b'] = np.array([0.846,0.841,0.879,0.843,0.957])
_yang22_co_params['b5c'] = np.array([0.160,0.169,0.170,0.172,0.168])

_yang22_co_params['siga'] = np.array([0.357,0.36,0.40,0.42,0.44])
_yang22_co_params['sigb'] = np.array([-0.0701,-0.072,-0.083,-0.091,-0.085])
_yang22_co_params['sigc'] = np.array([0.00621,0.0064,0.0070,0.0079,0.0063])

def _yang22_co_fit(redshift,line):
    """Fit parameters for CO - as a function of redshift for a given line"""
    if line not in ['CO1-0','CO2-1','CO3-2','CO4-3','CO5-4']:
        raise ValueError('CO line not recognized')
    pars = _yang22_co_params[_yang22_co_params['name']==line][0]

    return_scalar = np.isscalar(redshift)
    redshift = np.array(redshift)

    m1 = np.zeros(len(redshift))
    m1[redshift<4.0] = (pars['m1a'] + pars['m1b']*redshift)[redshift<4.0]
    m1[(redshift<5.0) & (redshift>=4.0)] = (pars['m4a'] + pars['m4b']*redshift)[(redshift<5.0) & (redshift>=4.0)]
    m1[redshift>=5] = (pars['m5a'] + pars['m5b']*redshift)[redshift>=5]

    n = np.zeros(len(redshift))
    n[redshift<4.0] = (pars['n1a'] + pars['n1b']*redshift + pars['n1c']*redshift**2)[redshift<4.0]
    n[(redshift<5.0) & (redshift>=4.0)] = (pars['n4a'] + pars['n4b']*redshift + pars['n4c']*redshift**2)[(redshift<5.0) & (redshift>=4.0)]
    n[redshift>=5] = (pars['n5a'] + pars['n5b']*redshift)[redshift>=5]

    a = np.zeros(len(redshift))
    a[redshift<4.0] = (pars['a1a'] + pars['a1b']*redshift + pars['a1c']*redshift**2)[redshift<4.0]
    a[(redshift<5.0) & (redshift>=4.0)] = (pars['a4a'] + pars['a4b']*redshift + pars['a4c']*redshift**2)[(redshift<5.0) & (redshift>=4.0)]
    a[redshift>=5] = (pars['a5a'] + pars['a5b']*redshift + pars['a5c']*redshift**2)[redshift>=5]

    b = np.zeros(len(redshift))
    b[redshift<4.0] = (pars['b1a']*np.exp(-redshift/pars['b1b']) + pars['b1c'])[redshift<4.0]
    b[(redshift<5.0) & (redshift>=4.0)] = (pars['b4a'] + pars['b4b']*redshift)[(redshift<5.0) & (redshift>=4.0)]
    b[redshift>=5] = (pars['b5a']*np.exp(-redshift/pars['b5b']) + pars['b5c'])[redshift>=5]

    sig = pars['siga'] + pars['sigb']*redshift + pars['sigc']*redshift**2

    if return_scalar:
        return m1[0], n[0], a[0], b[0], sig
    return m1, n, a, b, sig



_yang22_ci_params = np.zeros(2,dtype=[('name','U5'),
                                     ('m1a','f'),('m1b','f'),('m1c','f'),('m4a','f'),('m4b','f'),('m4c','f'),('m5a','f'),('m5b','f'),('m5c','f'),
                                     ('n1a','f'),('n1b','f'),('n1c','f'),('n4a','f'),('n4b','f'),('n4c','f'),('n5a','f'),('n5b','f'),('n5c','f'),
                                     ('a1a','f'),('a1b','f'),('a1c','f'),('a4a','f'),('a4b','f'),('a4c','f'),('a5a','f'),('a5b','f'),
                                     ('b1a','f'),('b1b','f'),('b4a','f'),('b4b','f'),('b5a','f'),('b5b','f'),('b5c','f'),
                                     ('siga','f'),('sigb','f'),('sigc','f')
                                     ])
_yang22_ci_params['name'] = np.array(['CI1-0','CI2-1'])
_yang22_ci_params['m1a'] = np.array([1.026,1.135])
_yang22_ci_params['m1b'] = np.array([2.346,2.616])
_yang22_ci_params['m1c'] = np.array([11.20,11.13])
_yang22_ci_params['m4a'] = np.array([8.54,8.823])
_yang22_ci_params['m4b'] = np.array([1.33,1.206])
_yang22_ci_params['m4c'] = np.array([-0.157,-0.1445])
_yang22_ci_params['m5a'] = np.array([4.294,4.37])
_yang22_ci_params['m5b'] = np.array([2.479,2.38])
_yang22_ci_params['m5c'] = np.array([10.68,10.7])

_yang22_ci_params['n1a'] = np.array([-1.451,-2.03])
_yang22_ci_params['n1b'] = np.array([1.893,1.80])
_yang22_ci_params['n1c'] = np.array([-5.046,-4.44])
_yang22_ci_params['n4a'] = np.array([-6.96,-4.906])
_yang22_ci_params['n4b'] = np.array([0.750,0.05632])
_yang22_ci_params['n4c'] = np.array([-0.0807,0])
_yang22_ci_params['n5a'] = np.array([1.89,1.48])
_yang22_ci_params['n5b'] = np.array([5.73,6.14])
_yang22_ci_params['n5c'] = np.array([-6.03,5.31])

_yang22_ci_params['a1a'] = np.array([-0.741,-1.16])
_yang22_ci_params['a1b'] = np.array([0.739,0.706])
_yang22_ci_params['a1c'] = np.array([1.86,2.00])
_yang22_ci_params['a4a'] = np.array([8.42,9.275])
_yang22_ci_params['a4b'] = np.array([-2.91,-3.225])
_yang22_ci_params['a4c'] = np.array([0.320,0.3543])
_yang22_ci_params['a5a'] = np.array([1.04,1.17])
_yang22_ci_params['a5b'] = np.array([0.165,0.169])

_yang22_ci_params['b1a'] = np.array([1.26,1.54])
_yang22_ci_params['b1b'] = np.array([-0.198,-0.259])
_yang22_ci_params['b4a'] = np.array([0.837,0.94])
_yang22_ci_params['b4b'] = np.array([-0.103,-0.12])
_yang22_ci_params['b5a'] = np.array([2090,3960])
_yang22_ci_params['b5b'] = np.array([0.520,0.487])
_yang22_ci_params['b5c'] = np.array([0.204,0.221])

_yang22_ci_params['siga'] = np.array([0.39,0.46])
_yang22_ci_params['sigb'] = np.array([-0.076,-0.096])
_yang22_ci_params['sigc'] = np.array([0.0063,0.0079])

def _yang22_ci_fit(redshift,line):
    """Fit parameters for CI - as a function of redshift for a given line"""

    if line not in ['CI1-0','CI2-1']:
        raise ValueError('CI line not recognized')
    pars = _yang22_ci_params[_yang22_co_params['name']==line][0]

    return_scalar = np.isscalar(redshift)
    redshift = np.array(redshift)

    m1 = np.zeros(len(redshift))
    m1[redshift<4.0] = (pars['m1a']*np.exp(-redshift/pars['m1b']) + pars['m1c'])[redshift<4.0]
    m1[(redshift<5.0) & (redshift>=4.0)] = (pars['m4a'] + pars['m4b']*redshift + pars['m4c']*redshift**2)[(redshift<5.0) & (redshift>=4.0)]
    m1[redshift>=5] = (pars['m5a']*np.exp(-redshift/pars['m5b']) + pars['m5c'])[redshift>=5]

    n = np.zeros(len(redshift))
    n[redshift<4.0] = (pars['n1a']*np.exp(-redshift/pars['n1b']) + pars['n1c'])[redshift<4.0]
    n[(redshift<5.0) & (redshift>=4.0)] = (pars['n4a'] + pars['n4b']*redshift + pars['n4c']*redshift**2)[(redshift<5.0) & (redshift>=4.0)]
    n[redshift>=5] = (pars['n5a']*np.exp(-redshift/pars['n5b']) + pars['n5c'])[redshift>=5]

    a = np.zeros(len(redshift))
    a[redshift<4.0] = (pars['a1a']*np.exp(-redshift/pars['a1b']) + pars['a1c'])[redshift<4.0]
    a[(redshift<5.0) & (redshift>=4.0)] = (pars['a4a'] + pars['a4b']*redshift + pars['a4c']*redshift**2)[(redshift<5.0) & (redshift>=4.0)]
    a[redshift>=5] = (pars['a5a'] + pars['a5b']*redshift)[redshift>=5]

    b = np.zeros(len(redshift))
    b[redshift<4.0] = (pars['b1a'] + pars['b1b']*redshift)[redshift<4.0]
    b[(redshift<5.0) & (redshift>=4.0)] = (pars['b4a'] + pars['b4b']*redshift)[(redshift<5.0) & (redshift>=4.0)]
    b[redshift>=5] = (pars['b5a']*np.exp(-redshift/pars['b5b']) + pars['b5c'])[redshift>=5]

    sig = pars['siga'] + pars['sigb']*redshift + pars['sigc']*redshift**2

    if return_scalar:
        return m1[0], n[0], a[0], b[0], sig
    return m1, n, a, b, sig

def _yang22_fduty(mass,redshift):
    """Duty cycle for Yang model"""
    f = np.ones(len(np.array(mass,ndmin=1)))
    m2lowz = 10**(11.73+0.6634*redshift)
    glowz = 1.37 - 0.190*redshift + 0.0215*redshift**2


    f[redshift<4] = ((1+(np.array(mass,ndmin=1)/m2lowz)**glowz)**-1)[redshift<4]
    # Higher redshift f duty is always 1

    if np.isscalar(mass) and np.isscalar(redshift):
        return f[0]
    return f
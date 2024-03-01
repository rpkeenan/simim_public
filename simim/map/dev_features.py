# To do list:
#   Add function to map for multiplying by primary beam
#   Add masking to map
#   Transformations (X, Y, lum-->flux-->intensity-->temp, etc.)

# Possible features
#   Analytic calculation of power spectra

###############################################################################
### Unit Conversions ##########################################################
###############################################################################

# These are all from a (very) old version

def X(z, cosmo=cosmo):
    '''X:
    Function to compute X - the conversion factor from angle
    to comoving distance at a given redshift

    Required arguments:
    z - a redshift of array of redshifts

    Returns:
    X - the conversion factor at z in units of Mpc/rad
    '''
    X = cosmo.comoving_transverse_distance(z).value
    return X


def Y(z, nu_rest = NU_CO_1TO0, cosmo=cosmo):
    '''Y:
    Function to compute Y - the conversion factor from frequency
    interval to comoving distance at a given redshift

    Required arguments:
    z - a redshift of array of redshifts

    Optional arguments:
    nu_rest (= 115.27e9) - the rest frequency of the line of interest

    Returns:
    Y - the conversion factor at z in units of Mpc/Hz
    '''

    Y = (1+z)**2 * c/1e3 / (cosmo.H0.value*cosmo.efunc(z)*nu_rest)
    return Y


def L_to_F(L, z, cosmo=cosmo):
    '''L_TO_F:
    Function to calculate flux from a source.

    Required arguments:
    L - the luminosity of the source (in L_sun)
    z - the redshift of the source

    Returns:
    F - the flux of the line (in W m^-2)
    '''

    L = array(L)*L_sun
    z = array(z)

    if z.ndim > 0 and z.size > 100:
        z_spline = linspace(amin(z), amax(z), 100)
        DL_spline = interp1d(z_spline, cosmo.luminosity_distance(z_spline).value * Mpc)
        DL = DL_spline(z)
    else:
        DL = cosmo.luminosity_distance(z).value * Mpc

    F = L/(4*pi*DL**2)

    return F


def F_to_I(F, dOm = 1, dnu = 1):
    '''F_to_I:
    Function to calculate the intensity of an observation, given flux and the
    dimensions of the observation (dOmega, dnu). By default an observational dimension
    of 1 str x 1 Hz is used.

    Required arguments:
    F - the flux of the line in W m^-2

    Optional arguments:
    dOm (= 1) - angular area of the observation, in steradians
    dnu (= 1) - frequency channel width of observation, Hz

    Returns:
    I - the intensity in W m^-2 str^-1 Hz^-1
    '''

    I = F / (dOm * dnu)
    return I


def I_to_T(I, nu, z=None, mode='rest frequency'):
    '''I_TO_T:
    Function to calculate brightness temperature of a source, given intensity.
    See Chung et al 2019, eq 7.

    Required arguments:
    I - the intensity of the line (in W m^-2 str^-1 Hz^-1)
    nu - the frequency of the line in Hz
    z - the redshift of the source, needed only for 'rest frequency'
    mode (= 'rest frequency') - specify whether frequencies will be given as
    rest frequency of observed frequency. Options are 'rest frequency' or
    'observed frequency'

    Optional arguments:
    delta_function_line (= True) - if true the line will be treated as having
    no width, if false the returned intensity will be increased by a factor of (1+z)

    Returns:
    T - the brightness temperature of the line (in K)
    '''

    if mode == 'observed frequency':
        nu_obs = array(nu)
    elif mode == 'rest frequency':
        if z.ndim == 0:
            if z == None:
                raise InputError('z', 'redshifts must be specified for rest frequency mode')

        nu_rest = array(nu)
        z = array(z)
        nu_obs = nu_rest/(1+z)
    else:
        raise InputError('mode', 'mode not recognized')

    I = array(I)
    T = I * c**2 / (2*k*nu_obs**2)

    return T


def L_to_I(L, z, dOm, dnu, cosmo=cosmo):
    '''L_TO_I:
    Wrapper function for L_TO_F, F_to_I.

    Required arguments:
    L - the luminosity of the source (in L_sun)
    z - the redshift of the source
    dOm - angular area of the observation, in steradians
    dnu - frequency channel size of the observation, in Hz

    Returns:
    I - the intensity in W m^-2 str^-1 Hz^-1
    '''

    F = L_to_F(L, z, cosmo)
    I = F_to_I(F, dOm, dnu)
    return I


def L_to_T(L, nu, z, dOm, dnu, mode='rest frequency',cosmo=cosmo):
    '''L_TO_T:
    Wrapper function for L_TO_F, F_to_I, and I_TO_T.

    Required arguments:
    L - the luminosity of the source (in L_sun)
    nu - the frequency of the line in Hz
    z - the redshift of the source
    dOm - angular area of the observation, in steradians
    dnu - frequency channel size of the observation, in Hz
    mode (= 'rest frequency') - specify whether frequencies will be given as
    rest frequency of observed frequency. Options are 'rest frequency' or
    'observed frequency'

    Returns:
    T - the brightness temperature of the line (in K)
    '''

    F = L_to_F(L, z, cosmo=cosmo)
    I = F_to_I(F, dOm, dnu)
    T = I_to_T(I, nu, z, mode=mode)
    return T




###############################################################################
### Masking ###################################################################
###############################################################################

# These are all from a (very) old version

def mask_planes(axes, dim_inds=None, vals=None):
    '''MASK_PLANES:
    Function to generate a mask for planes in a grid. Takes
    the axes of the grid as its argument and a list of planes
    to mask in the grid. If no planes are given, a mask with
    all cells set to true is returned.

    Planes are specified in the following manner - dim_ind
    identifies the constant dimension of the the plane and
    val indicates the value of the plane in that dimension.
    The function takes these as a list - dim_inds is a list
    of dimensions containing the planes, vals is a list of the
    values. (Both arguments must be specified as lists)

    Required arguments:
    axes - the axes of the grid

    Optional arguments:
    dim_inds - a list containing the dimensions in which the
    planes to be masked lie.
    vals - a list containing the values for the planes to be
    masked

    Returns:
    mask - an array of boolean values defining a mask
    '''

    shape = ()
    for i in axes:
        shape = shape + tuple([len(i.flatten())])
    mask = ones(shape, dtype = bool)

    if dim_inds == None:
        return mask

    for i in range(len(dim_inds)):
        if dim_inds[i] >= len(axes):
            raise InputError('dim_inds','dimension of specified plane higher than dimensionality of the space')

        mask = moveaxis(mask, dim_inds[i], 0)

        ind = where(isclose(axes[dim_inds[i]].flatten(), vals[i]))
        mask[ind] = False
        mask = moveaxis(mask, 0, dim_inds[i])

    return mask


def mask_rows(axes, dim_inds=None, vals=None):
    '''MASK_ROWS:
    Function to generate a mask for rows/columns in a grd. Takes
    the axes of the grid as its argument and a list of rows
    to mask. If no rows are given, a mask with all cells set to
    true is returned.

    Rows are specified in the following manner - dim_ind
    identifies the dimension along which the row is defined.
    Val is a tuple indicating the location of the row in the other
    dimensions of the grid. The function takes these as lists -
    dim_inds is a list of dimensions, vals is a list of tuples
    specifying val. (Both arguments must be specified as lists)

    Required arguments:
    axes - the axes of the grid

    Optional arguments:
    dim_inds - a list containing the dimensions along which the rows
    are defined.
    vals - a list containing tuples of locations for the rows in the plane
    perpendicular to the rows.

    Returns:
    mask - an array of boolean values defining a mask
    '''

    shape = ()
    for i in axes:
        shape = shape + tuple([len(i.flatten())])
    mask = ones(shape, dtype = bool)

    if dim_inds == None:
        return mask

    for i in range(len(dim_inds)):
        if dim_inds[i] >= len(axes):
            raise InputError('dim_inds','dimension of specified row higher than dimensionality of the space')

        mask = moveaxis(mask, dim_inds[i], -1)
        other_dims = setdiff1d(arange(len(axes)), [dim_inds[i]])
        other_inds = []
        for j in range(len(other_dims)):
            if vals[i][j] >= len(axes[other_dims[j]].flatten()):
                print(vals[i][j], len(axes[other_dims[j]].flatten()))
                print('add an error flag here')

            other_inds.append(where(isclose(axes[other_dims[j]].flatten(), vals[i][j])))
        mask[other_inds] = False
        mask = moveaxis(mask, -1, dim_inds[i])

    return mask


def mask_points(axes, points=None):
    '''MASK_POINTS:
    Function to generate a mask for points in a grd. Takes
    the axes of the grid as its argument and a list of points
    to mask in the grid. If no points are given, a mask with
    all cells set to true is returned.

    Points are specified as a list of points to mask in the
    points variable

    Required arguments:
    axes - the axes of the grid

    Optional arguments:
    points - a list of all points to mask, should be given
    as a tuple for each point.

    Returns:
    mask - an array of boolean values defining a mask
    '''

    shape = ()
    for i in axes:
        shape = shape + tuple([len(i.flatten())])
    mask = ones(shape, dtype = bool)

    if points == None:
        return mask

    for i in points:
        if len(i) != len(axes):
            raise InputError('points','dimension of point greater than dimensionality of the space')

        mask[i] = False

    return mask


def mk_mask(axes, points=None, row_dims=None, row_inds=None, plane_dims=None, plane_inds=None):
    '''MK_MASK:
    Function to generate a mask for a grd. A wrapper function
    for MASK_POINTS, MASK_ROWS, and MASK_PLANES. See documentation
    for those functions.

    Required arguments:
    axes - the axes of the grid

    Optional arguments:
    points - a list of all points to mask, should be given
    as a tuple for each point.
    row_dims - a list containing the dimensions along which the rows
    are defined.
    row_inds - a list containing tuples of locations for the rows in the plane
    perpendicular to the rows.
    plane_dims - a list containing the dimensions in which the
    planes to be masked lie.
    plane_inds - a list containing the values for the planes to be
    masked

    Returns:
    mask - an array of boolean values defining a mask
    '''

    mask1 = mask_points(axes, points)
    mask2 = mask_rows(axes, row_dims, row_inds)
    mask3 = mask_planes(axes, plane_dims, plane_inds)
    mask = mask1*mask2*mask3
    return mask

###############################################################################
### Analytic PS Calculation ###################################################
###############################################################################

def ps_analytic(pos, val, volume):
    '''PS_ANALYTIC:
    Given the locations and magnitudes of a collection of points
    calculates a power function by treating each point as a delta
    function.

    Required arguments:
    pos - a list of positions (in any dimension)
    val - the values corresponding to the listed points
    volume - the volume of the space for which the power spectrum
    is being calculated

    Returns:
    PS - a function that takes a position or list of positions in
    k space and returns the power at those points
    '''

    def PS(r):
        return abs(dot(val,exp(-2j*pi*dot(pos,r))))**2/volume
    return PS



def av_ps_analytic(kmin,kmax,number,points,pos,val,volume):
    '''AV_PS_ANALYTIC:
    Given the locations and magnitudes of a collection of points
    calculates the one dimensional power spectrum treating each
    point as a delta function.

    Required arguments:
    kmin - the minimum k of the power spectrum to calculate
    kmax - the maximum k of the power spectrum to calculate
    number - the number of points (logarithmically spaced) that will
    points - the number of points in each k shell at which the power
    will be sampled
    pos - a list of positions (in any dimension)
    val - the values corresponding to the listed points
    volume - the volume of the space for which the power spectrum
    is being calculated

    Returns:
    p - the average power at k
    k - wavenumbers corresponding to p
    '''

    F = ps_analytic(pos,val,volume)
    k = logspace(log10(kmin),log10(kmax),number)
    p = empty(number)

    for i in range(number):
        z = rand(points)*k[i]
        theta = rand(points)*pi/2
        x = sqrt(k[i]**2-z**2)*cos(theta)
        y = sqrt(k[i]**2-z**2)*sin(theta)
        r = array([x,y,z])
        p[i] = mean(F(r/2/pi))
        print( i)

    return p, k




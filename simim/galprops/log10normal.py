import numpy as np

def log10normal(x, dex, preserve_linear_mean=False, rng=np.random.default_rng()):
    """Helper function for introducing a log normal scatter. Can be set to preserve
    linear mean if desired

    Parameters
    ----------
    x : float or array
        The linear mean of the distribution (the scatter will be added
        around log10(x)).
    dex : float
        The scale parameter for the distribution, in log10 units
    preserve_linear_mean : bool, default=True
        Determines whether the scatter should preserve the (linear) mean
        value or not.
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    x_scattered : array
        The scattered values
    """

    mu = np.array(x)

    # Adjust value so the linear mean is preserved
    if preserve_linear_mean:
        mu = mu / 10**(dex**2 * np.log(10)/2)

    x_scattered = mu * np.power(10,dex*rng.normal(loc=0,scale=1,size=mu.shape))

    return x_scattered

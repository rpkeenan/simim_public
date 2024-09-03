import numpy as np
from scipy.stats import rv_continuous
from scipy.special import gamma, gammaincc, expi

def am_dfcat(prop1_cat: np.ndarray, prop2_rv: rv_continuous, 
             prop2_min: float = 0, prop2_max: float = np.inf, 
             missing_pmass_low_p1: float = 0, missing_pmass_high_p1: float = 0, 
             missing_pmass_low_p2: float = 0, missing_pmass_high_p2: float = 0,
             extrapolate_low_p1: float = 'nan',extrapolate_high_p1: float = 'nan',
             extrapolate_low_p2: float = 'nan',extrapolate_high_p2: float = 'nan',
             reverseorder: bool = False):
    """Abundance match prop1 and prop2 and return functions for determining
    the value of one property when the other is specified

    This code takes a catalog for property 1 and a distribution function for
    property 2 and abundance matches them to determine the relation between the
    two properties.

    Note: not all possible parameter combinations for this function have been
    tested, use with care and implement a few sanity checks.

    Arguments
    ---------
    prop1_cat : 1d array-like
        A set of values of property 1
    prop2_rv : scipy.stats.rv_continuous instance
        A scipy.stats.rv_continuous object describing the probability
        distribution of property 2
    prop2_min, prop2_max : float
        The minimum and maximum values for which to evaluate the property 2 CDF
        function, outside this range the extrapolation parameters are used.
        Defaults to 0 and infinity. The probability mass falling outside these
        limits can be specified by missing_pmass_low_p2 and
        missing_pmass_high_p2.
    missing_pmass_low_p1 : float between 0 and 1
        The fraction of the total probability mass that is missing bellow the
        lower limit of prop1_cat; should be between 0 and 1. Defaults to 0
    missing_pmass_high_p1 : float between 0 and 1
        The fraction of the total probability mass that is missing above the
        upper limit of prop1_cat; should be between 0 and 1. Defaults to 0
    missing_pmass_low_p2 : float between 0 and 1
        The fraction of the total probability mass that is missing bellow the
        lower limit where prop2_rv.cdf=0; should be between 0 and 1. Defaults to
        0
    missing_pmass_high_p2 : float between 0 and 1
        The fraction of the total probability mass that is missing above the
        upper limit where prop2_rv.cdf=1, should be between 0 and 1. Defaults to
        0
    extrapolate_low_p1, extrapolate_low_p2 : float, 'flat', 'nan'
        Option for how to extrapolate to lower values. If flat, minimum values
        will be used. If 'nan' all unmatched values will be assigned np.nan as a
        value of the other property. If any other value, unmatched values will
        be assigned this value
    extrapolate_high_p1, extrapolate_high_p2 : float, 'flat', 'nan'
        Option for how to extrapolate to higher values. If flat, maximum values
        will be used. If 'nan' all unmatched values will be assigned np.nan as a
        value of the other property. If any other value, unmatched values will
        be assigned this value
    reverseorder : bool, optional
        The order in which to perform the matching. By default high values will
        be matched to high values and low values to low values. Setting
        reverseorder to True will result in high values of one property being
        associated to low values of the other and vice versa 

    Returns
    -------
    p2ofp1 : function
        Function that returns the value of p2 for a given p1
    p1ofp2 : function
        Function that returns the value of p1 for a given p2
    """

    if np.any(np.array([missing_pmass_low_p1,missing_pmass_low_p2,missing_pmass_high_p1,missing_pmass_high_p2])>1):
        raise ValueError("missing probability mass cannot be greater than 1")
    if np.any(np.array([missing_pmass_low_p1,missing_pmass_low_p2,missing_pmass_high_p1,missing_pmass_high_p2])<0):
        raise ValueError("missing probability mass cannot be less than 0")
    if missing_pmass_low_p1+missing_pmass_high_p1>1 or missing_pmass_low_p2+missing_pmass_high_p2>1:
        raise ValueError("total missing probability mass cannot be greater than 1")
    
    # Fill with limiting values:
    fillers = [np.min(prop1_cat), np.max(prop1_cat), prop2_min, prop2_max]
    
    # Where told to, use other values instead:
    for i,extrapolate in enumerate([extrapolate_low_p1,extrapolate_high_p1,extrapolate_low_p2,extrapolate_high_p2]):
        if extrapolate not in ['flat','none']:
            try:
                fillers[i] = float(extrapolate)
            except:
                raise ValueError("extrapolate mode not recognized")
        elif extrapolate=='none':
            fillers[i] = np.nan

    o1 = np.sort(prop1_cat)
    cdf1 = np.linspace(missing_pmass_low_p1,1-missing_pmass_high_p1,len(o1))
    def cdf_func_p1(p1):
        return np.interp(p1, o1, cdf1,left=0,right=1)
    def ppf_func_p1(cum1):
        return np.interp(cum1, cdf1, o1,left=fillers[0],right=fillers[1])

    def cdf_func_p2(p2):
        cdf = missing_pmass_low_p2 + (prop2_rv.cdf(p2)-prop2_rv.cdf(prop2_min))/(prop2_rv.cdf(prop2_max)-prop2_rv.cdf(prop2_min)) * (1-missing_pmass_high_p2-missing_pmass_low_p2)
        if np.asarray(p2).ndim > 0:
            cdf[p2<prop2_min] = 0
            cdf[p2>prop2_max] = 1
        else:
            if p2<prop2_min:
                cdf=0
            elif p2>prop2_max:
                cdf=1        
        return cdf
    def ppf_func_p2(cum2):
        cstar = (cum2-missing_pmass_low_p2)/(1-missing_pmass_high_p2-missing_pmass_low_p2) * (prop2_rv.cdf(prop2_max)-prop2_rv.cdf(prop2_min))/1 + prop2_rv.cdf(prop2_min)

        if np.asarray(cum2).ndim > 0:
            ppf = np.zeros(len(cum2))
            ppf[(cum2>missing_pmass_low_p2) & (cum2<(1-missing_pmass_high_p2))] = prop2_rv.ppf(cstar[(cum2>missing_pmass_low_p2) & (cum2<(1-missing_pmass_high_p2))])
            ppf[cum2<missing_pmass_low_p2] = fillers[2]
            ppf[cum2>(1-missing_pmass_high_p2)] = fillers[3]
        else:
            if cum2<missing_pmass_low_p2:
                ppf = fillers[2]
            elif cum2>1-missing_pmass_high_p2:
                ppf = fillers[3]
            else:
                ppf = prop2_rv.ppf(cstar)
    
        return ppf

    if reverseorder:
        def p1ofp2(p2):
            return ppf_func_p1(1-cdf_func_p2(p2))
        def p2ofp1(p1):
            return ppf_func_p2(1-cdf_func_p1(p1))
    else:
        def p1ofp2(p2):
            return ppf_func_p1(cdf_func_p2(p2))
        def p2ofp1(p1):
            return ppf_func_p2(cdf_func_p1(p1))

    return p2ofp1, p1ofp2


def am_dfdf(prop1_rv:rv_continuous, prop2_rv:rv_continuous, 
            prop1_min: float = 0, prop1_max: float = np.inf, 
            prop2_min: float = 0, prop2_max: float = np.inf, 
            missing_pmass_low_p1: float = 0, missing_pmass_high_p1: float = 0, 
            missing_pmass_low_p2: float = 0, missing_pmass_high_p2: float = 0,
            extrapolate_low_p1: float = 'nan', extrapolate_high_p1: float = 'nan',
            extrapolate_low_p2: float = 'nan', extrapolate_high_p2: float = 'nan',
            reverseorder: bool = False):
    """Abundance match prop1 and prop2 and return functions that for determining
    the value of one property when the other is specified

    This code takes distribution functions for two properties, and directly
    abundance matches them to determine the relation between the two properties.

    Note: not all possible parameter combinations for this function have been
    tested, use with care and implement a few sanity checks.

    Arguments
    ---------
    prop1_rv : scipy.stats.rv_continuous instance
        A scipy.stats.rv_continuous object describing the probability distribution
        of property 1
    prop2_rv : scipy.stats.rv_continuous instance
        A scipy.stats.rv_continuous object describing the probability distribution
        of property 2
    prop1_min, prop1_max, prop2_min, prop2_max : float
        The minimum and maximum values for which to evaluate the CDF 
        functions, outside this range the extrapolation parameters are used. Defaults 
        to 0 and infinity. The probability mass falling outside these limits can 
        be specified by missing_pmass_low_p2 and missing_pmass_high_p2.
    missing_pmass_low_p1 : float between 0 and 1
        The fraction of the total probability mass that is missing bellow the
        lower limit of prop1_cat; should be between 0 and 1. Defaults to 0
    missing_pmass_high_p1 : float between 0 and 1
        The fraction of the total probability mass that is missing above the
        upper limit of prop1_cat; should be between 0 and 1. Defaults to 0
    missing_pmass_low_p2 : float between 0 and 1
        The fraction of the total probability mass that is missing bellow the
        lower limit where prop2_rv.cdf=0; should be between 0 and 1. Defaults to
        0
    missing_pmass_high_p2 : float between 0 and 1
        The fraction of the total probability mass that is missing above the
        upper limit where prop2_rv.cdf=1, should be between 0 and 1. Defaults to
        0
    extrapolate_low_p1, extrapolate_low_p2 : float, 'flat', 'nan'
        Option for how to extrapolate to lower values. If flat, minimum values
        will be used. If 'nan' all unmatched values will be assigned np.nan as a
        value of the other property. If any other value, unmatched values will
        be assigned this value
    extrapolate_high_p1, extrapolate_high_p2 : float, 'flat', 'nan'
        Option for how to extrapolate to higher values. If flat, maximum values
        will be used. If 'nan' all unmatched values will be assigned np.nan as a
        value of the other property. If any other value, unmatched values will
        be assigned this value
    reverseorder : bool, optional
        The order in which to perform the matching. By default high values will
        be matched to high values and low values to low values. Setting
        reverseorder to True will result in high values of one property being
        associated to low values of the other and vice versa 

    Returns
    -------
    p2ofp1 : function
        Function that returns the value of p2 for a given p1
    p1ofp2 : function
        Function that returns the value of p1 for a given p2
    """

    if np.any(np.array([missing_pmass_low_p1,missing_pmass_low_p2,missing_pmass_high_p1,missing_pmass_high_p2])>1):
        raise ValueError("missing probability mass cannot be greater than 1")
    if np.any(np.array([missing_pmass_low_p1,missing_pmass_low_p2,missing_pmass_high_p1,missing_pmass_high_p2])<0):
        raise ValueError("missing probability mass cannot be less than 0")
    if missing_pmass_low_p1+missing_pmass_high_p1>1 or missing_pmass_low_p2+missing_pmass_high_p2>1:
        raise ValueError("total missing probability mass cannot be greater than 1")
    
    # Fill with limiting values:
    fillers = [prop1_min, prop1_max, prop2_min, prop2_max]
    
    # Where told to, use other values instead:
    for i,extrapolate in enumerate([extrapolate_low_p1,extrapolate_high_p1,extrapolate_low_p2,extrapolate_high_p2]):
        if extrapolate not in ['flat','none']:
            try:
                fillers[i] = float(extrapolate)
            except:
                raise ValueError("extrapolate mode not recognized")
        elif extrapolate=='none':
            fillers[i] = np.nan


    def cdf_func_p1(p1):
        cdf = missing_pmass_low_p1 + (prop1_rv.cdf(p1)-prop1_rv.cdf(prop1_min))/(prop1_rv.cdf(prop1_max)-prop1_rv.cdf(prop1_min)) * (1-missing_pmass_high_p1-missing_pmass_low_p1)
        if np.asarray(p1).ndim > 0:
            cdf[p1<prop1_min] = 0
            cdf[p1>prop1_max] = 1
        else:
            if p1<prop1_min:
                cdf=0
            elif p1>prop1_max:
                cdf=1        
        return cdf
    def ppf_func_p1(cum1):
        cstar = (cum1-missing_pmass_low_p1)/(1-missing_pmass_high_p1-missing_pmass_low_p1) * (prop1_rv.cdf(prop1_max)-prop1_rv.cdf(prop1_min))/1 + prop1_rv.cdf(prop1_min)

        if np.asarray(cum1).ndim > 0:
            ppf = np.zeros(len(cum1))
            ppf[(cum1>missing_pmass_low_p1) & (cum1<(1-missing_pmass_high_p1))] = prop1_rv.ppf(cstar[(cum1>missing_pmass_low_p1) & (cum1<(1-missing_pmass_high_p1))])
            ppf[cum1<missing_pmass_low_p1] = fillers[2]
            ppf[cum1>(1-missing_pmass_high_p1)] = fillers[3]
        else:
            if cum1<missing_pmass_low_p1:
                ppf = fillers[2]
            elif cum1>1-missing_pmass_high_p1:
                ppf = fillers[3]
            else:
                ppf = prop1_rv.ppf(cstar)
    
        return ppf
    
    def cdf_func_p2(p2):
        cdf = missing_pmass_low_p2 + (prop2_rv.cdf(p2)-prop2_rv.cdf(prop2_min))/(prop2_rv.cdf(prop2_max)-prop2_rv.cdf(prop2_min)) * (1-missing_pmass_high_p2-missing_pmass_low_p2)
        if np.asarray(p2).ndim > 0:
            cdf[p2<prop2_min] = 0
            cdf[p2>prop2_max] = 1
        else:
            if p2<prop2_min:
                cdf=0
            elif p2>prop2_max:
                cdf=1        
        return cdf
    def ppf_func_p2(cum2):
        cstar = (cum2-missing_pmass_low_p2)/(1-missing_pmass_high_p2-missing_pmass_low_p2) * (prop2_rv.cdf(prop2_max)-prop2_rv.cdf(prop2_min))/1 + prop2_rv.cdf(prop2_min)

        if np.asarray(cum2).ndim > 0:
            ppf = np.zeros(len(cum2))
            ppf[(cum2>missing_pmass_low_p2) & (cum2<(1-missing_pmass_high_p2))] = prop2_rv.ppf(cstar[(cum2>missing_pmass_low_p2) & (cum2<(1-missing_pmass_high_p2))])
            ppf[cum2<missing_pmass_low_p2] = fillers[2]
            ppf[cum2>(1-missing_pmass_high_p2)] = fillers[3]
        else:
            if cum2<missing_pmass_low_p2:
                ppf = fillers[2]
            elif cum2>1-missing_pmass_high_p2:
                ppf = fillers[3]
            else:
                ppf = prop2_rv.ppf(cstar)
    
        return ppf

    if reverseorder:
        def p1ofp2(p2):
            return ppf_func_p1(1-cdf_func_p2(p2))
        def p2ofp1(p1):
            return ppf_func_p2(1-cdf_func_p1(p1))
    else:
        def p1ofp2(p2):
            return ppf_func_p1(cdf_func_p2(p2))
        def p2ofp1(p1):
            return ppf_func_p2(cdf_func_p1(p1))

    # pcs = np.array([.25,.5,.75])
    # print('ppf_func_p1:', ppf_func_p1(pcs))
    # print('cdf_func_p2:', cdf_func_p2(np.array([22411.17569982, 240140.0741543, 8718959.09962908])))
    # print('ppf_func_p1(cdf_func_p2):', ppf_func_p1(cdf_func_p2(np.array([22411.17569982, 240140.0741543, 8718959.09962908]))))
    # print('p1ofp2:',p1ofp2(np.array([22411.17569982, 240140.0741543, 8718959.09962908])))

    # print()

    # print('ppf_func_p2:', ppf_func_p2(pcs))
    # print('cdf_func_p1:', cdf_func_p1(np.array([1.38999992e+10, 2.23399997e+10, 4.90000015e+10])))
    # print('ppf_func_p2(cdf_func_p1):', ppf_func_p2(cdf_func_p1(np.array([1.38999992e+10, 2.23399997e+10, 4.90000015e+10]))))
    # print('p2ofp1:',p2ofp1(np.array([1.38999992e+10, 2.23399997e+10, 4.90000015e+10])))

    return p2ofp1, p1ofp2



class schechter_gen(rv_continuous):
    """Scipy distribution generator for a schechter function"""

    def _uppergammamod(self, a, x):
        """Schechter function parameterized as
            Phi(L) = Phi* x (L/L*)^alpha x exp(-L/L*)
        The logarithm of this is
            logPhi* + alpha x L - alpha x L* - L + L*
            logPhi* + (alpha-1) x L - (alpha+1) x L*
        The integral of the LF, which serves as the denominator of the likelihood function
        is
            int(L_min,inf)(Phi(L)dL) = Phi* x int(L_min,inf)[(L/L*)^alpha x exp(-L/L*) dL]
        substitute u = L/L*, umin = L_min/L*, du = dL/L* to get
                            = Phi* x int(u_min,inf)[u^alpha x exp(-u) du*L*]
                            = Phi* x L* x int(u_min,inf)[u^alpha x exp(-u) du]
                            = Phi* x L* x upper_gamma(alpha+1,u_min)
        the latter step is only true for the case where alpha > -1, due to the definition
        of the gamma function. However, we can use integration by parts to extend the
        result for smaller alpha.
        Let f = t^(a-1), f' = (a-1)t^(a-2), g = -e^-t, g' = e^-t. Then using integration
        by parts we have:
            int(x,inf)[f g' dt] = f(inf)g(inf)-f(x)g(x) - int(x,inf)[f' g dt]
        for a > 0 the LHS is upper_gamma(a,x), so we have
            upper_gamma(a,x) = 0 + x^(a-1)e^-x + (a-1)int(x,inf)[t^(a-2) e^-t dt]
            int(x,inf)[t^(a-2) e^-t dt] = [upper_gamma(a,x) - x^(a-1)e^-x] / (a-1)
        Here the LHS has the form of upper_gamma(a-1,x), but where a-1 may be less than 0.
        We can call this upper_gamma_mod(a-1,x). Using the above result recursively, we
        have
            upper_gamma_mod(a-n,x) = [upper_gamma_mod(a-(n-1),x) - x^(a-n)e^-x] / (a-n)
        which allows us to extend the upper gamma function to a < 0. For a = 0 the integral
        is
            int(x,inf)[t^-1 e^-t dt]
        which is related to the exponential integral (https://mathworld.wolfram.com/ExponentialIntegral.html),
        which can be evaluated using a scipy function."""
        
        amod = a%1
        n = np.asarray(amod-a).astype('int')
        n[n<0] = 0

        if np.asarray(a).size == 1:
            if a>0:
                gp = gamma(a) * gammaincc(a, x)
            elif a<=0 and amod==0:
                gp = -expi(-x)
            elif a<0 and amod!=0:
                gp = gamma(amod) * gammaincc(amod, x)

        elif np.asarray(x).shape == np.asarray(a.shape):
            gp = np.zeros(x.shape)

            c1 = np.asarray(a) > 0
            c2 = (np.asarray(a) <= 0) & (np.asarray(amod)==0)
            c3 = (np.asarray(a) < 0) & (np.asarray(amod)!=0)

            gp[c1] = gamma(a[c1]) * gammaincc(a[c1], x[c1])
            gp[c2] = -expi(-x[c2])
            gp[c3] = gamma(amod[c3]) * gammaincc(amod[c3], x[c3])

        elif np.asarray(x).size == 1:
            gp = np.zeros(a.shape)
            c1 = np.asarray(a) > 0
            c2 = (np.asarray(a) <= 0) & (np.asarray(amod)==0)
            c3 = (np.asarray(a) < 0) & (np.asarray(amod)!=0)

            gp[c1] = gamma(a[c1]) * gammaincc(a[c1], x[c1])
            gp[c2] = -expi(-x[c2])
            gp[c3] = gamma(amod[c3]) * gammaincc(amod[c3], x[c3])

        else:
            raise ValueError('a and x shapes incompatible')



        for i in range(np.max(n)):
            if np.asarray(a).size == 1:
                gp = (gp - x**(amod-(i+1))*np.exp(-x)) / (amod - (i+1))

            else:
                gp[n>1] = (gp[n>1] - x**(amod[n>1]-(i+1))*np.exp(-x[n>1])) / (amod[n>1] - (i+1))

        return gp
    
    def _pdf(self, x, x0, alpha, xmin):
        norm = self.self.uppergammamod(alpha+1, xmin/x0)
        p = (np.asarray(x)/x0)**alpha * 1/x0 * np.exp(-np.asarray(x)/x0) / norm
        p[x<xmin] = 0
        return p
    
    def _logpdf(self, x, x0, alpha, xmin):
        norm = np.log(self.uppergammamod(alpha+1, xmin/x0))
        p = alpha * np.log(np.asarray(x)) - (alpha+1) * np.log(x0) + (-x/x0) - norm
        p[x<xmin] = -np.inf
        return p
    
    def _cdf(self, x, x0, alpha, xmin):
        norm = self.uppergammamod(alpha+1, xmin/x0)
        int = self.uppergammamod(alpha+1, x/x0)
        c = (norm-int)/norm
        c[x<xmin] = 0
        return c

    def _logcdf(self, x, x0, alpha, xmin):
        norm = self.uppergammamod(alpha+1, xmin/x0)
        int = self.uppergammamod(alpha+1, x/x0)
        c = np.log(norm-int) - np.log(norm)
        c[x<xmin] = -np.inf
        return c

    def _get_support(self, x0, alpha, xmin, **kwargs):
        return xmin, np.inf
        
    def _argcheck(self, x0, alpha, xmin):
        """
        Get rid of the argcheck method since it's unclear what it does
        but breaks the pdf and cdf
        """
        if x0<=0:
            return 0
        if xmin<=0:
            return 0
        return 1
    

class double_schechter_gen(rv_continuous):
    """Scipy distribution generator for a double schechter function"""

    def lum_function(self, x, x0, alpha1, phi1, alpha2, phi2, xmin):
        """Returns phi(M*) to be multiplied by dM*"""
        phi = np.exp(-x/x0) * (phi1 * (x/x0)**alpha1 + phi2 * (x/x0)**alpha2) * 1/x0
        if np.asarray(x).ndim > 0:
            phi[x<xmin] = 0
        elif x<xmin:
            phi = 0
        return phi

    def lum_function_int(self, x, x0, alpha1, phi1, alpha2, phi2, xmin):
        int = phi1 * self._uppergammamod(alpha1+1, x/x0) + phi2 * self._uppergammamod(alpha2+1,x/x0)
        if np.asarray(x).ndim > 0:
            int[x<xmin] = phi1 * self._uppergammamod(alpha1+1, xmin/x0) + phi2 * self._uppergammamod(alpha2+1,xmin/x0)
        else:
            if x<xmin:
                int = 1

        return int

    def _uppergammamod(self, a, x):
        """Double schechter function parameterized as
            Phi(x) = [Phi1 (x/x*)^alpha1 + Phi2 (x/x*)^alpha2)] exp(-x/x*) / x*

        The integral of the LF, which serves as the denominator of the likelihood function
        is int(x_min,inf)(Phi(x)dx) = phi1 int(x_min,inf)[(x/x*)^alpha1 exp(-x/x*) dx/x*]
            + phi2 int(x_min,inf)[(x/x*)^alpha2 exp(-x/x*) dx/x*]
        substitute u = x/x*, umin = x_min/x*, du = dx/x* to get
                            = Phi1 int(u_min,inf)[u^alpha1 exp(-u) du] + Phi2 int(u_min,inf)[u^alpha2 exp(-u) du]
                            = Phi1 upper_gamma(alpha1+1,u_min) + Phi2 upper_gamma(alpha2+1,u_min)
        the latter step is only true for the case where alpha > -1, due to the definition
        of the gamma function. However, we can use integration by parts to extend the
        result for smaller alpha.
        Let f = t^(a-1), f' = (a-1)t^(a-2), g = -e^-t, g' = e^-t. Then using integration
        by parts we have:
            int(x,inf)[f g' dt] = f(inf)g(inf)-f(x)g(x) - int(x,inf)[f' g dt]
        for a > 0 the LHS is upper_gamma(a,x), so we have
            upper_gamma(a,x) = 0 + x^(a-1)e^-x + (a-1)int(x,inf)[t^(a-2) e^-t dt]
            int(x,inf)[t^(a-2) e^-t dt] = [upper_gamma(a,x) - x^(a-1)e^-x] / (a-1)
        Here the LHS has the form of upper_gamma(a-1,x), but where a-1 may be less than 0.
        We can call this upper_gamma_mod(a-1,x). Using the above result recursively, we
        have
            upper_gamma_mod(a-n,x) = [upper_gamma_mod(a-(n-1),x) - x^(a-n)e^-x] / (a-n)
        which allows us to extend the upper gamma function to a < 0. For a = 0 the integral
        is
            int(x,inf)[t^-1 e^-t dt]
        which is related to the exponential integral (https://mathworld.wolfram.com/ExponentialIntegral.html),
        which can be evaluated using a scipy function."""
        
        amod = a%1
        n = np.asarray(amod-a).astype('int')
        n[n<0] = 0

        if np.asarray(a).size == 1:
            if a>0:
                gp = gamma(a) * gammaincc(a, x)
            elif a<=0 and amod==0:
                gp = -expi(-x)
            elif a<0 and amod!=0:
                gp = gamma(amod) * gammaincc(amod, x)

        elif np.asarray(x).shape == np.asarray(a.shape):
            gp = np.zeros(x.shape)

            c1 = np.asarray(a) > 0
            c2 = (np.asarray(a) <= 0) & (np.asarray(amod)==0)
            c3 = (np.asarray(a) < 0) & (np.asarray(amod)!=0)

            gp[c1] = gamma(a[c1]) * gammaincc(a[c1], x[c1])
            gp[c2] = -expi(-x[c2])
            gp[c3] = gamma(amod[c3]) * gammaincc(amod[c3], x[c3])

        elif np.asarray(x).size == 1:
            gp = np.zeros(a.shape)
            c1 = np.asarray(a) > 0
            c2 = (np.asarray(a) <= 0) & (np.asarray(amod)==0)
            c3 = (np.asarray(a) < 0) & (np.asarray(amod)!=0)

            gp[c1] = gamma(a[c1]) * gammaincc(a[c1], x[c1])
            gp[c2] = -expi(-x[c2])
            gp[c3] = gamma(amod[c3]) * gammaincc(amod[c3], x[c3])

        else:
            raise ValueError('a and x shapes incompatible')



        for i in range(np.max(n)):
            if np.asarray(a).size == 1:
                gp = (gp - x**(amod-(i+1))*np.exp(-x)) / (amod - (i+1))

            else:
                gp[n>1] = (gp[n>1] - x**(amod[n>1]-(i+1))*np.exp(-x[n>1])) / (amod[n>1] - (i+1))

        return gp
    
    def _pdf(self, x, x0, alpha1, phi1, alpha2, phi2, xmin):
        norm = self.lum_function_int(xmin, x0, alpha1, phi1, alpha2, phi2, xmin)
        p = self.lum_function(x, x0, alpha1, phi1, alpha2, phi2, xmin)
        p[x<xmin] = 0

        return p
    
    def _cdf(self, x, x0, alpha1, phi1, alpha2, phi2, xmin):
        norm = self.lum_function_int(xmin, x0, alpha1, phi1, alpha2, phi2, xmin)
        int = self.lum_function_int(x, x0, alpha1, phi1, alpha2, phi2, xmin)
        c = (norm-int)/norm
        c[x<xmin] = 0

        return c

    def _ppf(self, cdf, x0, alpha1, phi1, alpha2, phi2, xmin):
        if np.asarray(phi1).ndim > 0:
            x0 = np.asarray(x0)[0]
            phi1 = np.asarray(phi1)[0]
            phi2 = np.asarray(phi2)[0]
            alpha1 = np.asarray(alpha1)[0]
            alpha2 = np.asarray(alpha2)[0]
            xmin = np.asarray(xmin)[0]
        
        xint = np.logspace(np.log10(xmin),np.log10(xmin)+20,10001)
        yint = self.lum_function_int(xint, x0, alpha1, phi1, alpha2, phi2, xmin)
        yint = 1 - yint/self.lum_function_int(xmin, x0, alpha1, phi1, alpha2, phi2, xmin)
        
        return np.interp(cdf, yint, xint)

    def _get_support(self, x0, alpha1, phi1, alpha2, phi2, xmin, **kwargs):
        return xmin, np.inf
        
    def _argcheck(self, x0, alpha1, phi1, alpha2, phi2, xmin):
        """
        Get rid of the argcheck method since it's unclear what it does
        but breaks the pdf and cdf
        """
        if x0<=0:
            return 0
        if xmin<=0:
            return 0
        if phi1<0:
            return 0
        if phi2<0:
            return 0
        return 1

class modified_schechter_gen(rv_continuous):
    """Scipy distribution generator for a modified schechter function"""

    def lum_function(self, x, phi0, x0, alpha, sigma, xmin):
        '''Returns modified schechter function.
        suitable for integration in d log L space'''
        phi = phi0 * (x/x0)**(1-alpha) * np.exp(-1/(2*sigma**2) * np.log10(1+x/x0)**2)
        phi[x < xmin] = 0
        return phi

    def lum_function_int(self, x, phi0, x0, alpha, sigma, xmin):
        xint = np.logspace(np.log10(xmin),np.log10(xmin)+20,10001)
        xdif = np.log10(xint[1]) - np.log10(xint[0])
        yint = np.cumsum(self.lum_function(xint[::-1], phi0, x0, alpha, sigma, xmin))[::-1] * xdif
        
        int = np.interp(x, xint, yint)

        return int

    def _pdf(self, x, phi0, x0, alpha, sigma, xmin):
        if np.asarray(phi0).ndim > 0:
            phi0 = np.asarray(phi0)[0]
            x0 = np.asarray(x0)[0]
            alpha = np.asarray(alpha)[0]
            sigma = np.asarray(sigma)[0]
            xmin = np.asarray(xmin)[0]

        norm = self.lum_function_int(xmin, phi0, x0, alpha, sigma, xmin)

        p = self.lum_function(np.asarray(x), phi0, x0, alpha, sigma, xmin) / norm
        return p
    
    def _cdf(self, x, phi0, x0, alpha, sigma, xmin):
        if np.asarray(phi0).ndim > 0:
            phi0 = np.asarray(phi0)[0]
            x0 = np.asarray(x0)[0]
            alpha = np.asarray(alpha)[0]
            sigma = np.asarray(sigma)[0]
            xmin = np.asarray(xmin)[0]

        norm = self.lum_function_int(xmin, phi0, x0, alpha, sigma, xmin)
        int = self.lum_function_int(x, phi0, x0, alpha, sigma, xmin)

        c = (norm-int)/norm
        c[np.asarray(x)<xmin] = 0
        return c
    
    def _ppf(self, cdf, phi0, x0, alpha, sigma, xmin):
        if np.asarray(phi0).ndim > 0:
            phi0 = np.asarray(phi0)[0]
            x0 = np.asarray(x0)[0]
            alpha = np.asarray(alpha)[0]
            sigma = np.asarray(sigma)[0]
            xmin = np.asarray(xmin)[0]
        
        xint = np.logspace(np.log10(xmin),np.log10(xmin)+20,10001)
        xdif = np.log10(xint[1]) - np.log10(xint[0])
        yint = np.cumsum(self.lum_function(xint, phi0, x0, alpha, sigma, xmin)) * xdif
        yint /= yint[-1]
        
        return np.interp(cdf, yint, xint)


    def _get_support(self, phi0, x0, alpha, sigma, xmin, **kwargs):
        return xmin, xmin*1e20
        
    def _argcheck(self, phi0, x0, alpha, sigma, xmin):
        """
        Get rid of the argcheck method since it's unclear what it does
        but breaks the pdf and cdf
        """
        if x0<=0:
            return 0
        if xmin<=0:
            return 0
        if sigma<=0:
            return 0
        return 1

class double_power_law_gen(rv_continuous):
    """Scipy distribution generator for a double power law lf function"""
    
    def lum_function(self, x, phi0, x0, alpha, beta, xmin):
        '''Returns double power law function.'''
        phi = phi0 * ((x/x0)**(alpha) + (x/x0)**beta)**-1
        phi[x < xmin] = 0
        return phi

    def lum_function_int(self, x, phi0, x0, alpha, beta, xmin):
        xint = np.logspace(np.log10(xmin),np.log10(xmin)+20,10001)
        xdif = np.log10(xint[1]) - np.log10(xint[0])
        yint = np.cumsum(self.lum_function(xint[::-1], phi0, x0, alpha, beta, xmin))[::-1] * xdif
        
        int = np.interp(x, xint, yint)

        return int

    def _pdf(self, x, phi0, x0, alpha, beta, xmin):
        if np.asarray(phi0).ndim > 0:
            phi0 = np.asarray(phi0)[0]
            x0 = np.asarray(x0)[0]
            alpha = np.asarray(alpha)[0]
            beta = np.asarray(beta)[0]
            xmin = np.asarray(xmin)[0]

        norm = self.lum_function_int(xmin, phi0, x0, alpha, beta, xmin)

        p = self.lum_function(np.asarray(x), phi0, x0, alpha, beta, xmin) / norm
        return p
    
    def _cdf(self, x, phi0, x0, alpha, beta, xmin):
        if np.asarray(phi0).ndim > 0:
            phi0 = np.asarray(phi0)[0]
            x0 = np.asarray(x0)[0]
            alpha = np.asarray(alpha)[0]
            beta = np.asarray(beta)[0]
            xmin = np.asarray(xmin)[0]

        norm = self.lum_function_int(xmin, phi0, x0, alpha, beta, xmin)
        int = self.lum_function_int(x, phi0, x0, alpha, beta, xmin)

        c = (norm-int)/norm
        c[np.asarray(x)<xmin] = 0
        return c
    
    def _ppf(self, cdf, phi0, x0, alpha, beta, xmin):
        if np.asarray(phi0).ndim > 0:
            phi0 = np.asarray(phi0)[0]
            x0 = np.asarray(x0)[0]
            alpha = np.asarray(alpha)[0]
            beta = np.asarray(beta)[0]
            xmin = np.asarray(xmin)[0]
        
        xint = np.logspace(np.log10(xmin),np.log10(xmin)+20,10001)
        xdif = np.log10(xint[1]) - np.log10(xint[0])
        yint = np.cumsum(self.lum_function(xint, phi0, x0, alpha, beta, xmin)) * xdif
        yint /= yint[-1]
        
        return np.interp(cdf, yint, xint)


    def _get_support(self, phi0, x0, alpha, beta, xmin, **kwargs):
        return xmin, xmin*1e20
        
    def _argcheck(self, phi0, x0, alpha, beta, xmin):
        """
        Get rid of the argcheck method since it's unclear what it does
        but breaks the pdf and cdf
        """
        if x0<=0:
            return 0
        if xmin<=0:
            return 0
        return 1
"""List of accepted simulations and code to check if an input matches

All simulations which are handled by the current version of SimIM are 
listed here, this will have to be modified if adding support for 
additional simulations"""

_illsims = ['Illustris-1','Illustris-2','Illustris-3','Illustris-1-Dark','Illustris-2-Dark','Illustris-3-Dark']
_tngsims = ['TNG300-1','TNG300-2','TNG300-3','TNG300-1-Dark','TNG300-2-Dark','TNG300-3-Dark','TNG100-1','TNG100-2','TNG100-3','TNG100-1-Dark','TNG100-2-Dark','TNG100-3-Dark']
_umsims = ['UniverseMachine-BolshoiPlanck','UniverseMachine-SMDPL','UniverseMachine-MDPL2','_testbox']
_umuchuusims = ['UniverseMachine-uchuu']
_acceptedsims = _illsims + _tngsims + _umsims + _umuchuusims

def _checksim(sim):
    """Check if a specified simulation name is acceptable"""

    if not sim in _acceptedsims:
        raise NameError("Simulation '{}' not recognized/supported".format(sim))

    return

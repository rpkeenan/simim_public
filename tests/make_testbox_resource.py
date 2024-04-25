"""This script exists to create the test catalogs packaged in the resources folder shipped with SimIM.
It is included for reproducibility, but should never need to be run. It assumes raw data for UM-BolshoiPlanck 
is present on the machine."""
import os
import numpy as np
try:
    from importlib.resources import files
except:
    from importlib_resources import files
from simim._paths import _SimIMPaths
from simim.siminterface.universemachine import _testsnaps

# First snap, z~2, second to last snap (because last is at z<0)

def _maketestsnaps():
    paths = _SimIMPaths()
    raw_path = os.path.join(paths.sims['UniverseMachine-BolshoiPlanck'],'raw')
    resource_path = files('simim').joinpath('resources','_testbox')

    dtype_raw = np.dtype(dtype=[('id', 'i'), ('descid','i'),('upid','i'),('flags','i'),('uparent_dist','f'),
                                ('x','f'),('y','f'),('z','f'),('vx','f'),('vy','f'),('vz','f'),
                                ('m','f'),('v','f'),('mp','f'),('vmp','f'),('r','f'),
                                ('rank1','f'),('rank2','f'),('ra','f'),('rarank','f'),
                                ('sm','f'),('icl','f'),('sfr','f'),('obs_sm','f'),('obs_sfr','f'),('obs_ssfr','f'),('smhm','f'),('obs_uv','f')])
    fmt_out = ['%i','%i','%i','%i','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f']

    for ts in _testsnaps:
        print(f'loading on {ts}')
        data = np.loadtxt(os.path.join(raw_path,ts), dtype=dtype_raw, comments='#')
        
        print(f'working on {ts}')
        data = data[data['x']<=10]
        data = data[data['y']<=10]
        data = data[data['z']<=10]

        print(f'saving {ts}')
        np.savetxt(resource_path.joinpath(ts), data, fmt=fmt_out)

if __name__ == '__main__':
    _maketestsnaps()




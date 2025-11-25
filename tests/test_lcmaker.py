# To do

import pytest
import os
from shutil import rmtree
import numpy as np

from simim import lightcone
from simim._paths import _SimIMPaths
from simim.siminterface import UniversemachineCatalogs

def test_lcmaker_path(monkeypatch):

    with pytest.warns():
        cat = UniversemachineCatalogs('_testbox')
    cat.download_meta()
    cat.download()
    cat.format()
    monkeypatch.setattr('builtins.input', lambda _ : 'y')
    cat.clean_raw()

    lc = lightcone.LCMaker('_testbox','_testlc',0.05,1,redshift_min=0,redshift_max=1,overwrite=True)

    paths = _SimIMPaths()
    assert lc.lc_path == os.path.join(paths.lcs['_testbox'], '_testlc')

    lc.build_lightcones(1)
    assert os.path.exists(os.path.join(paths.lcs['_testbox'],'_testlc','lc_0000.hdf5'))

    os.remove(os.path.join(paths.sims['_testbox'],'meta.npy'))
    os.remove(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    os.remove(os.path.join(paths.sims['_testbox'],'data.hdf5'))
    rmtree(os.path.join(paths.lcs['_testbox'],'_testlc'))

def test_spheremaker_path(monkeypatch):

    with pytest.warns():
        cat = UniversemachineCatalogs('_testbox')
    cat.download_meta()
    cat.download()
    cat.format()
    monkeypatch.setattr('builtins.input', lambda _ : 'y')
    cat.clean_raw()

    lc = lightcone.SphereMaker('_testbox','_testlc',redshift_min=0,redshift_max=0.002,overwrite=True)

    paths = _SimIMPaths()
    assert lc.lc_path == os.path.join(paths.lcs['_testbox'], '_testlc')

    lc.build_lightcones(1)
    assert os.path.exists(os.path.join(paths.lcs['_testbox'],'_testlc','lc_0000.hdf5'))

    os.remove(os.path.join(paths.sims['_testbox'],'meta.npy'))
    os.remove(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    os.remove(os.path.join(paths.sims['_testbox'],'data.hdf5'))
    rmtree(os.path.join(paths.lcs['_testbox'],'_testlc'))

def test_spheremaker(monkeypatch):

    with pytest.warns():
        cat = UniversemachineCatalogs('_testbox')
    cat.download_meta()
    cat.download()
    cat.format()
    monkeypatch.setattr('builtins.input', lambda _ : 'y')
    cat.clean_raw()

    lc = lightcone.SphereMaker('_testbox','_testlc',redshift_min=0,redshift_max=0.002,overwrite=True)
    lc.build_lightcones(1)

    # Check geometry of resulting light cone
    lc0 = lightcone.LCHandler('_testbox','_testlc',0)
    z = lc0.return_property('redshift')
    ra = lc0.return_property('ra')
    dec = lc0.return_property('dec')

    assert np.all(z>=0)
    assert np.all(z<=0.01)
    assert np.all(ra>=0)
    assert np.all(ra<=np.pi*2)
    assert np.all(dec<=np.pi/2)
    assert np.all(dec>=-np.pi/2)

    # Half sphere
    lc = lightcone.SphereMaker('_testbox','_testlc',redshift_min=0,redshift_max=0.002,overwrite=True,mode='hemisphere')
    lc.build_lightcones(1)
    lc0 = lightcone.LCHandler('_testbox','_testlc',0)
    z = lc0.return_property('redshift')
    ra = lc0.return_property('ra')
    dec = lc0.return_property('dec')

    assert np.all(z>=0)
    assert np.all(z<=0.01)
    assert np.all(ra>=0)
    assert np.all(ra<=np.pi*2)
    assert np.all(dec<=np.pi/2)
    assert np.all(dec>0)

    # Quarter sphere
    lc = lightcone.SphereMaker('_testbox','_testlc',redshift_min=0,redshift_max=0.002,overwrite=True,mode='quarter')
    lc.build_lightcones(1)
    lc0 = lightcone.LCHandler('_testbox','_testlc',0)
    z = lc0.return_property('redshift')
    ra = lc0.return_property('ra')
    dec = lc0.return_property('dec')

    assert np.all(z>=0)
    assert np.all(z<=0.01)
    assert np.all(ra>=0)
    assert np.all(ra<=np.pi)
    assert np.all(dec<=np.pi/2)
    assert np.all(dec>0)

    # Eighth sphere
    lc = lightcone.SphereMaker('_testbox','_testlc',redshift_min=0,redshift_max=0.002,overwrite=True,mode='eighth')
    lc.build_lightcones(1)
    lc0 = lightcone.LCHandler('_testbox','_testlc',0)
    z = lc0.return_property('redshift')
    ra = lc0.return_property('ra')
    dec = lc0.return_property('dec')

    assert np.all(z>=0)
    assert np.all(z<=0.01)
    assert np.all(ra>=0)
    assert np.all(ra<=np.pi/2)
    assert np.all(dec<=np.pi/2)
    assert np.all(dec>0)

    paths = _SimIMPaths()
    os.remove(os.path.join(paths.sims['_testbox'],'meta.npy'))
    os.remove(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    os.remove(os.path.join(paths.sims['_testbox'],'data.hdf5'))
    rmtree(os.path.join(paths.lcs['_testbox'],'_testlc'))


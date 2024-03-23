import os
import numpy as np
import pytest
import warnings

from simim.siminterface import UniversemachineCatalogs, SimHandler
from simim._paths import _SimIMPaths

_testsnaps = np.array(['sfr_catalog_0.055623.txt','sfr_catalog_0.506185.txt','sfr_catalog_0.994717.txt'])

def test_setup(monkeypatch):

    with pytest.warns():
        cat = UniversemachineCatalogs('_testbox')
    paths = _SimIMPaths()

    assert '_testbox' in paths.sims

    assert os.path.exists(paths.sims['_testbox'])
    assert ~os.path.exists(os.path.join(paths.sims['_testbox'],'raw'))
    assert ~os.path.exists(os.path.join(paths.sims['_testbox'],'meta.npy'))
    assert ~os.path.exists(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    assert ~os.path.exists(os.path.join(paths.sims['_testbox'],'data.hdf5'))

    cat.download_meta()
    assert ~os.path.exists(os.path.join(paths.sims['_testbox'],'raw'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'meta.npy'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    assert ~os.path.exists(os.path.join(paths.sims['_testbox'],'data.hdf5'))

    cat.download()
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'raw'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'raw','sfr_catalog_0.055623.txt'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'meta.npy'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    assert ~os.path.exists(os.path.join(paths.sims['_testbox'],'data.hdf5'))

    cat.format()
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'raw'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'raw','sfr_catalog_0.055623.txt'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'meta.npy'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'data.hdf5'))

    monkeypatch.setattr('builtins.input', lambda _ : 'y')
    cat.clean_raw()
    assert ~os.path.exists(os.path.join(paths.sims['_testbox'],'raw'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'meta.npy'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    assert os.path.exists(os.path.join(paths.sims['_testbox'],'data.hdf5'))

    os.remove(os.path.join(paths.sims['_testbox'],'meta.npy'))
    os.remove(os.path.join(paths.sims['_testbox'],'meta_snaps.npy'))
    os.remove(os.path.join(paths.sims['_testbox'],'data.hdf5'))

def test_setup_newpath(monkeypatch):

    monkeypatch.setattr('builtins.input', lambda _ : 'y')
    with pytest.warns():
        cat = UniversemachineCatalogs('_testbox',path='./')

    paths = _SimIMPaths()

    assert '_testbox' in paths.sims
    assert paths.sims['_testbox'] == os.path.abspath('./')

    cat.download_meta()
    cat.download()
    cat.format()

    h = SimHandler('_testbox')
    assert h.path == os.path.abspath('./')

    monkeypatch.setattr('builtins.input', lambda _ : 'y')
    cat.clean_raw()

    os.remove('./meta.npy')
    os.remove('./meta_snaps.npy')
    os.remove('./data.hdf5')
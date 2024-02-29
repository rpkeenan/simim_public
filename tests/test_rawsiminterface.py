import os
import pathlib
import shutil
import pytest

import simim
from simim._paths import _SimIMPaths

def test_import():
    """Make sure setupsimim is available"""
    print(dir(simim))
    assert 'setupsimim' in dir(simim)

@pytest.fixture
def setup_paths():
    path = str(pathlib.Path(__file__).parent.resolve())

    p = _SimIMPaths()

    if os.path.abspath(p.root) == os.path.abspath(path):
        raise ValueError("Somethings wrong - test shouldn't be running in the data directory")

    # clear root and set a new one, but save values
    # to make sure we get everything back in the end
    old_root = p.root
    old_root_file = p.root_file
    p.root = None
    p.root_file = os.path.join(path, 'test_root.txt')
    
    p._setuppath(path)

    yield p, path, old_root, old_root_file

    os.remove(os.path.join(path, 'test_root.txt'))
    shutil.rmtree(os.path.join(path, 'simim_resources'))

def test_folder_setup(setup_paths):
    """Check that paths are created correctly"""

    p, path, old_root, old_root_file = setup_paths

    assert os.path.exists(path + os.sep + 'simim_resources')
    assert os.path.exists(path + os.sep + 'simim_resources' + os.sep + '.paths')
    assert os.path.exists(path + os.sep + 'simim_resources' + os.sep + 'simulations')
    assert os.path.exists(path + os.sep + 'simim_resources' + os.sep + 'lightcones')
    assert os.path.exists(path + os.sep + 'simim_resources' + os.sep + 'galprops')

def test_tester(setup_paths):
    """Check that the test setup doesn't break anything
    this is also a good check on behavior of the root stuff"""
    
    p, path, old_root, old_root_file = setup_paths

    # Create second _SimIMPaths instance
    p2 = _SimIMPaths()
    assert p2.root_file == old_root_file
    if p2.root is None:
        assert old_root is None
    else:
        assert p2.root == old_root
    
def test_newsimpath(setup_paths):
    """Check newsimpath creates expected files"""
    
    p, path, old_root, old_root_file = setup_paths

    p._newsimpath('test', checkoverwrite=False)
    assert os.path.exists(path + os.sep + 'simim_resources' + os.sep + 'simulations' + os.sep + 'test')
    assert 'test' in p.sims
    assert p.sims['test'] == os.path.abspath(path + os.sep + 'simim_resources' + os.sep + 'simulations' + os.sep + 'test')

def test_newlcpath(setup_paths):
    """Check newlcpath creates expected files"""
    
    p, path, old_root, old_root_file = setup_paths

    p._newlcpath('test', checkoverwrite=False)
    assert os.path.exists(path + os.sep + 'simim_resources' + os.sep + 'lightcones' + os.sep + 'test')
    assert 'test' in p.lcs
    assert p.lcs['test'] == os.path.abspath(path + os.sep + 'simim_resources' + os.sep + 'lightcones' + os.sep + 'test')

def test_newproppath(setup_paths):
    """Check newproppath creates expected files"""
    
    p, path, old_root, old_root_file = setup_paths

    p._newproppath('test', checkoverwrite=False)
    assert os.path.exists(path + os.sep + 'simim_resources' + os.sep + 'galprops' + os.sep + 'test')
    assert 'test' in p.props
    assert p.props['test'] == os.path.abspath(path + os.sep + 'simim_resources' + os.sep + 'galprops' + os.sep + 'test')
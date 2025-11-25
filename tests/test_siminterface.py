import pytest

from simim import siminterface
from simim.siminterface import _sims

def test_simchecker():
    _sims._checksim('Illustris-1')

    with pytest.raises(Exception):
        _sims._checksim('Illustris-100')

    _sims._checksim('UniverseMachine-BolshoiPlanck')

    with pytest.raises(Exception):
        _sims._checksim('UniverseMachine-BigPickle')

    _sims._checksim('UniverseMachine-uchuu')

    with pytest.raises(Exception):
        _sims._checksim('Uchuu')

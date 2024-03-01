import pytest
import numpy as np

from simim.map.gridder import _Grid

def test_Grid():
    g = _Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()

    assert g.grid.shape == (10,10)
    assert g.axes[0].shape == (10,)
    assert g.axes_centers[0].shape == (9,)
    assert g.axes[0] == np.arange(11)
    assert g.axes_centers[0] == np.arange(10) + 0.5
    assert g.axes[1] == np.arange(11)
    assert g.axes_centers[1] == np.arange(10) + 0.5

def test_Grid_wrong_pix_size():
    g = _Grid(1, (5,5), (10,10), (1.1,1.1))
    g.init_grid()

    assert g.grid.shape == (9,9)
    assert g.axes[0].shape == (9,)
    assert g.axes_centers[0].shape == (8,)
    assert g.axes[0][0] == 0
    assert g.axes[0][-1] == 9*1.1
    assert g.axes[1][0] == 0
    assert g.axes[1][-1] == 9*1.1

def test_Grid_asym():
    g = _Grid(1, (5,10), (10,20), (1,1))
    g.init_grid()

    assert g.grid.shape == (10,20)
    assert g.axes[0].shape == (10,)
    assert g.axes[1].shape == (20,)
    assert g.axes_centers[0].shape == (9,)
    assert g.axes_centers[0].shape == (19,)
    assert g.axes[0][0] == 0
    assert g.axes[0][-1] == 10
    assert g.axes[1][0] == 0
    assert g.axes[1][-1] == 20

def test_Grid_1d():
    g = _Grid(1, 5, 10, 1)
    g.init_grid()

    assert g.grid.shape == (10,)
    assert g.axes[0].shape == (10,)
    assert g.axes_centers[0].shape == (9,)
    assert g.axes[0][0] == 0
    assert g.axes[0][-1] == 10

def test_Grid_init_options():
    g1 = _Grid(1, (5,5), (10,10), (1,1))
    g1.init()

    g2 = _Grid(1, (5,5), 10, (1,1))
    g2.init()

    g3 = _Grid(1, (5,5), (10,10), 1)
    g3.init()

    g3 = _Grid(1, (5,5), 10, 1)
    g3.init()

    assert g1.grid == g2.grid
    assert g1.grid == g3.grid


def test_crop():
    g = _Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()
    g.crop(0,1,9)

    assert g.grid.shape == (8,10)
    assert g.axes[0].shape == (8,)
    assert g.axes_centers[0].shape == (7,)
    assert g.axes[0] == np.arange(1,10)
    assert g.axes_centers[0] == np.arange(1,9) + 0.5

    g.crop(0,1.5,9.5)
    assert g.grid.shape == (7,10)
    assert g.axes[0].shape == (7,)
    assert g.axes_centers[0].shape == (6,)
    assert g.axes[0] == np.arange(2,10)
    assert g.axes_centers[0] == np.arange(2,9) + 0.5

    assert g.axes[1] == np.arange(10)
    assert g.axes_centers[1] == np.arange(9) + 0.5


def test_pad():
    g = _Grid(1, (5,5), (10,10), (1,0.5))
    g.init_grid()
    g.pad((0,1),2,15)

    assert g.grid[0,0] == 15

    g2 = _Grid(1, (5,5), (10,10), (1,1))
    g2.init_grid()

    g.unpad((0,1),2)
    assert g.grid == g2.grid
    assert g.axes[0] == g2.axes[0]
    assert g.axes_centers[0] == g2.axes_centers[0]
    assert g.axes[1] == g2.axes[1]
    assert g.axes_centers[1] == g2.axes_centers[1]

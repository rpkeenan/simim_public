import pytest
import numpy as np

from simim.map.gridder import Grid

def test_Grid():
    g = Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()

    assert g.grid.shape == (10,10,1)
    assert g.axes[0].shape == (11,)
    assert g.axes_centers[0].shape == (10,)
    assert np.all(g.axes[0] == np.arange(11))
    assert np.all(g.axes_centers[0] == np.arange(10) + 0.5)
    assert np.all(g.axes[1] == np.arange(11))
    assert np.all(g.axes_centers[1] == np.arange(10) + 0.5)

def test_Grid_wrong_pix_size():
    with pytest.warns():
        g = Grid(1, (5,5), (10,10), (1.1,1.1))
    g.init_grid()

    assert g.grid.shape == (10,10,1)
    assert g.axes[0].shape == (11,)
    assert g.axes_centers[0].shape == (10,)
    assert g.side_length[0] == 11
    assert g.side_length[1] == 11
    assert g.axes[0][0] == 5-1.1*5
    assert g.axes[0][-1] == 5+1.1*5
    assert g.axes[1][0] == 5-1.1*5
    assert g.axes[1][-1] == 5+1.1*5

def test_Grid_asym():
    g = Grid(1, (5,10), (10,20), (1,1))
    g.init_grid()

    assert g.grid.shape == (10,20,1)
    assert g.axes[0].shape == (11,)
    assert g.axes[1].shape == (21,)
    assert g.axes_centers[0].shape == (10,)
    assert g.axes_centers[1].shape == (20,)
    assert g.axes[0][0] == 0
    assert g.axes[0][-1] == 10
    assert g.axes[1][0] == 0
    assert g.axes[1][-1] == 20

def test_Grid_1d():
    g = Grid(1, 5, 10, 1)
    g.init_grid()

    assert g.grid.shape == (10,1)
    assert g.axes[0].shape == (11,)
    assert g.axes_centers[0].shape == (10,)
    assert g.axes[0][0] == 0
    assert g.axes[0][-1] == 10

def test_Grid_init_options():
    g1 = Grid(1, (5,5), (10,10), (1,1))
    g1.init_grid()

    g2 = Grid(1, (5,5), 10, (1,1))
    g2.init_grid()

    g3 = Grid(1, (5,5), (10,10), 1)
    g3.init_grid()

    g3 = Grid(1, (5,5), 10, 1)
    g3.init_grid()

    assert np.all(g1.grid == g2.grid)
    assert np.all(g1.grid == g3.grid)


def test_crop():
    g = Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()

    g.crop(0,1,9)
    assert g.grid.shape == (8,10,1)
    assert g.axes[0].shape == (9,)
    assert g.axes_centers[0].shape == (8,)
    assert np.all(g.axes[0] == np.arange(1,10))
    assert np.all(g.axes_centers[0] == np.arange(1,9) + 0.5)

    g.crop(0,1.5,8.5)
    assert g.grid.shape == (8,10,1)
    assert g.axes[0].shape == (9,)
    assert g.axes_centers[0].shape == (8,)
    assert np.all(g.axes[0] == np.arange(1,10))
    assert np.all(g.axes_centers[0] == np.arange(1,9) + 0.5)

    g.crop(0,1.6,8.4)
    assert g.grid.shape == (6,10,1)
    assert g.axes[0].shape == (7,)
    assert g.axes_centers[0].shape == (6,)
    assert np.all(g.axes[0] == np.arange(2,9))
    assert np.all(g.axes_centers[0] == np.arange(2,8) + 0.5)

    assert np.all(g.axes[1] == np.arange(11))
    assert np.all(g.axes_centers[1] == np.arange(10) + 0.5)


def test_pad():
    g = Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()
    g.pad((0,1),2,15)

    assert g.grid[0,0] == 15

    g2 = Grid(1, (5,5), (10,10), (1,1))
    g2.init_grid()

    g.pad((0,1),-2)

    assert np.all(g.grid == g2.grid)
    assert np.all(g.axes[0] == g2.axes[0])
    assert np.all(g.axes_centers[0] == g2.axes_centers[0])
    assert np.all(g.axes[1] == g2.axes[1])
    assert np.all(g.axes_centers[1] == g2.axes_centers[1])

def test_pad_opt():
    g1 = Grid(1, (5,5), (10,10), (1,0.5))
    g1.init_grid()
    g1.pad((0,1),2,15)

    g2 = Grid(1, (5,5), (10,10), (1,0.5))
    g2.init_grid()
    g2.pad((0,1),(2,2),15)

    assert np.all(g1.grid == g2.grid)

def test_collapse():
    g = Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()
    g2 = g.collapse_dimension(0,in_place=False)
    g = g.collapse_dimension(0,in_place=True)

    for h in [g,g2]:
        assert h.n_dimensions == 1
        assert len(h.axes) == 1
        assert len(h.axes_centers) == 1
        assert len(h.fourier_axes) == 1
        assert len(h.fourier_axes_centers) == 1

        assert np.all(h.center_point == np.array([5]))
        assert np.all(h.pixel_size == np.array([1]))
        assert np.all(h.side_length == np.array([10]))
        assert h.grid.shape == (10,1)

def test_add_from_cat():
    g = Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()
    positions = np.array([[0.5,1.4],[1.2,9.9],[6.2,8.8]])
    values = 0.1*np.ones(len(positions))
    
    g.add_from_cat(positions=positions)

    assert np.sum(g.grid) == 3
    assert g.grid[0,1,0] == 1
    assert g.grid[1,9,0] == 1
    assert g.grid[6,8,0] == 1

    g.add_from_cat(positions=positions, values=values)
    assert np.isclose(np.sum(g.grid), 3.3)
    assert np.isclose(g.grid[0,1,0], 1.1)
    assert np.isclose(g.grid[1,9,0], 1.1)
    assert np.isclose(g.grid[6,8,0], 1.1)

    g.add_from_cat(positions=positions, values=values, new_props=True)
    assert np.isclose(np.sum(g.grid), 3.6)
    assert np.isclose(g.grid[0,1,0], 1.1)
    assert np.isclose(g.grid[1,9,0], 1.1)
    assert np.isclose(g.grid[6,8,0], 1.1)
    assert np.isclose(g.grid[0,1,1], .1)
    assert np.isclose(g.grid[1,9,1], .1)
    assert np.isclose(g.grid[6,8,1], .1)

def test_add_from_cat_edges():
    g = Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()
    positions = np.array([[2,2],[8,8],[10,10],[0,0]],dtype=float)
    values = 0.1*np.ones(len(positions))
    
    g.add_from_cat(positions=positions,values=values)

    assert np.isclose(np.sum(g.grid), 0.3)
    assert np.isclose(g.grid[0,0,0], 0.1)
    assert np.isclose(g.grid[2,2,0], 0.1)
    assert np.isclose(g.grid[8,8,0], 0.1)
    assert np.isclose(g.grid[9,9,0], 0.0)

def test_sample():
    g = Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()
    positions1 = np.array([[0.5,1.4],[1.2,9.9],[6.2,8.8]])
    positions2 = np.array([[1.5,3.4],[3.2,3.9],[8.2,7.8]])
    values1 = 0.1*np.ones(len(positions1))
    values2 = 0.3*np.ones(len(positions2))
    
    g.add_from_cat(positions=positions1,values=values1)

    s1 = g.sample(positions=positions1)
    s2 = g.sample(positions=positions2)

    assert np.all(np.isclose(s1, 0.1))
    assert np.all(np.isclose(s2, 0.0))

    g.add_from_cat(positions=positions2,values=values2)

    s1 = g.sample(positions=positions1)
    s2 = g.sample(positions=positions2)

    assert np.all(np.isclose(s1, 0.1))
    assert np.all(np.isclose(s2, 0.3))
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

def test_add_new_prop():
    g = Grid(1, (5,5), (10,10), (1,1))

    g.add_new_prop()
    assert g.n_properties == 2

    with pytest.raises(Exception):
        g.add_new_prop(10)
    with pytest.raises(Exception):
        g.add_new_prop(np.zeros((10,10)))

    g.init_grid()
    assert g.n_properties == 2
    assert g.grid.shape == (10,10,2)

    g.add_new_prop(np.zeros((10,10)))
    assert g.n_properties == 3
    assert g.grid.shape == (10,10,3)

    g.add_new_prop(np.zeros((10,10,2)))
    assert g.n_properties == 5
    assert g.grid.shape == (10,10,5)


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

def test_add_from_cat_property_idx():
    g = Grid(1, (5,5), (10,10), (1,1))
    g.init_grid()
    positions = np.array([[0.5,1.4],[1.2,9.9],[6.2,8.8]])
    values = 0.1*np.ones(len(positions))
    
    # One property in grid:
    g.add_from_cat(positions=positions, properties=0)
    assert np.sum(g.grid) == 3
    assert g.grid[0,1,0] == 1
    assert g.grid[1,9,0] == 1
    assert g.grid[6,8,0] == 1

    # Add on top of existing
    g.add_from_cat(positions=positions, values=values, properties=0)
    assert np.isclose(np.sum(g.grid), 3.3)
    assert np.isclose(g.grid[0,1,0], 1.1)
    assert np.isclose(g.grid[1,9,0], 1.1)
    assert np.isclose(g.grid[6,8,0], 1.1)

    # Add into new property
    g.add_from_cat(positions=positions, values=values, new_props=True)
    assert np.isclose(np.sum(g.grid), 3.6)
    assert np.isclose(g.grid[0,1,0], 1.1)
    assert np.isclose(g.grid[1,9,0], 1.1)
    assert np.isclose(g.grid[6,8,0], 1.1)
    assert np.isclose(g.grid[0,1,1], .1)
    assert np.isclose(g.grid[1,9,1], .1)
    assert np.isclose(g.grid[6,8,1], .1)

    # Add on top of original property
    g.add_from_cat(positions=positions, values=values, properties=0)
    assert np.isclose(np.sum(g.grid), 3.9)
    assert np.isclose(g.grid[0,1,0], 1.2)
    assert np.isclose(g.grid[1,9,0], 1.2)
    assert np.isclose(g.grid[6,8,0], 1.2)
    assert np.isclose(g.grid[0,1,1], .1)
    assert np.isclose(g.grid[1,9,1], .1)
    assert np.isclose(g.grid[6,8,1], .1)

    # Add on top of new property
    g.add_from_cat(positions=positions, values=values, properties=1)
    assert np.isclose(np.sum(g.grid), 4.2)
    assert np.isclose(g.grid[0,1,0], 1.2)
    assert np.isclose(g.grid[1,9,0], 1.2)
    assert np.isclose(g.grid[6,8,0], 1.2)
    assert np.isclose(g.grid[0,1,1], .2)
    assert np.isclose(g.grid[1,9,1], .2)
    assert np.isclose(g.grid[6,8,1], .2)

    # Add on top of both properties - values only has 1 property dim
    g.add_from_cat(positions=positions, values=values, properties=None)
    assert np.isclose(np.sum(g.grid), 4.8)
    assert np.isclose(g.grid[0,1,0], 1.3)
    assert np.isclose(g.grid[1,9,0], 1.3)
    assert np.isclose(g.grid[6,8,0], 1.3)
    assert np.isclose(g.grid[0,1,1], .3)
    assert np.isclose(g.grid[1,9,1], .3)
    assert np.isclose(g.grid[6,8,1], .3)

    # Add on top of both properties - values only has 2 property dims
    values_new = 0.2*np.ones(len(positions))
    g.add_from_cat(positions=positions, values=np.array([values,values_new]).T, properties=None)
    assert np.isclose(np.sum(g.grid), 5.7)
    assert np.isclose(g.grid[0,1,0], 1.4)
    assert np.isclose(g.grid[1,9,0], 1.4)
    assert np.isclose(g.grid[6,8,0], 1.4)
    assert np.isclose(g.grid[0,1,1], .5)
    assert np.isclose(g.grid[1,9,1], .5)
    assert np.isclose(g.grid[6,8,1], .5)



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



def test_add_array():
    g = Grid(1, (5,4), (10,8), (1,1))
    g.init_grid()
    positions0 = np.array([[0.5],[1.2]])
    spec0 = np.ones((len(positions0),g.n_pixels[1]))
    positions1 = np.array([[5]])
    spec1 = np.ones((len(positions1),g.n_pixels[0]))

    # Add some spectra:
    g.add_from_pos_plus_array(positions=positions0,values=spec0)
    assert np.sum(g.grid) == 16
    assert np.all(g.grid[0] == 1)
    assert np.all(g.grid[1] == 1)

    # Other axis
    g.add_from_pos_plus_array(positions=positions1,values=spec1,ax=0)
    assert np.sum(g.grid) == 26
    assert np.all(g.grid[2:,5] == 1)
    assert np.all(g.grid[:2,5] == 2)

    # New axis
    g.add_from_pos_plus_array(positions=positions1,values=spec1,ax=0,new_prop=True)
    assert np.sum(g.grid) == 36
    assert np.all(g.grid[2:,5,0] == 1)
    assert np.all(g.grid[:2,5,0] == 2)
    assert np.all(g.grid[:,5,1] == 1)

    # Add to multi-prop axis
    g.add_from_pos_plus_array(positions=positions1,values=spec1,ax=0,properties=0)
    assert np.sum(g.grid) == 46
    assert np.all(g.grid[2:,5,0] == 2)
    assert np.all(g.grid[:2,5,0] == 3)
    assert np.all(g.grid[:,5,1] == 1)

    # Add to multi-prop axis
    g.add_from_pos_plus_array(positions=positions1,values=spec1,ax=0,properties=None)
    assert np.sum(g.grid) == 66
    assert np.all(g.grid[2:,5,0] == 3)
    assert np.all(g.grid[:2,5,0] == 4)
    assert np.all(g.grid[:,5,1] == 2)

def test_add_array_3d():
    g = Grid(1, (0,0,0), 5, 1)
    g.init_grid()

    position = np.array([[0,0]])
    value = np.ones(5)
    g.add_from_pos_plus_array(positions=position,values=np.array([value]),ax=0)
    g.add_from_pos_plus_array(positions=position,values=np.array([value]),ax=1)
    g.add_from_pos_plus_array(positions=position,values=np.array([value]),ax=2)

    assert np.sum(g.grid) == 15
    assert g.grid[2,2,2] == 3
    assert g.grid[2,2,0] == 1
    assert g.grid[2,0,2] == 1
    assert g.grid[0,2,2] == 1


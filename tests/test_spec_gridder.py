import pytest
import numpy as np

from simim.map.gridder import Grid

def delta_cum(f,pars):
    x = pars[:,0]
    a = pars[:,1]
    x=np.array(x,ndmin=1)
    a=np.array(a,ndmin=1)

    f=np.array(f,ndmin=1)
    if f.ndim != 1:
        raise ValueError("frequencies must be a 1d array")
    
    spec = np.where(f.reshape(1,-1)>=x.reshape(-1,1),a.reshape(-1,1),0)

    return spec

def flat_cum(f,pars):
    total = pars[:,0]
    min = np.min(f)
    max = np.max(f)
    spec = (f.reshape(1,-1)-min)/max * total.reshape(-1,1)
    return spec

def flat_diff(f,pars):
    total = pars[:,0]
    spec = np.ones((1,len(f))) * total.reshape(-1,1) / len(f)
    return spec

@pytest.fixture
def setup_3dgrid():
    g = Grid(1, (2.5,2.5,2.5), 5, 1)
    g.init_grid()

    positions1 = np.array([[0.5,0.5],[1.5,1.5],[2.5,2.5]])
    x1 = np.array([0.5,4.5,0.5])
    a1 = np.array([1,10,1])

    s1 = np.sum(a1)
    pvs1 = [[(0,0,0),1],[(1,1,4),10],[(2,2,0),1]]

    yield g, positions1, x1, a1, s1, pvs1

def test_basic(setup_3dgrid):
    g = setup_3dgrid[0]
    positions1 = setup_3dgrid[1]
    x1 = setup_3dgrid[2]
    a1 = setup_3dgrid[3]
    s1 = setup_3dgrid[4]
    pvs1 = setup_3dgrid[5]

    # Basic delta function test - cumulative version
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=-1)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]] == pv[1]

def test_flatspec(setup_3dgrid):
    g = setup_3dgrid[0]
    positions1 = setup_3dgrid[1]
    a1 = setup_3dgrid[3]
    s1 = setup_3dgrid[4]
    pvs1 = setup_3dgrid[5]

    # Basic flat spectrum test - cumulative version
    g.add_from_spec_func(positions=positions1,spec_function=flat_cum,spec_function_arguments=a1,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=-1)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]] == pv[1]/5

    # Basic flat spectrum test - differential version
    g.grid[:] = 0
    g.add_from_spec_func(positions=positions1,spec_function=flat_diff,spec_function_arguments=a1,
                         is_cumulative=False,eval_as_loop=False,new_prop=False,properties=0,spec_ax=-1)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]] == pv[1]/5

def test_eval_loop(setup_3dgrid):
    g = setup_3dgrid[0]
    positions1 = setup_3dgrid[1]
    x1 = setup_3dgrid[2]
    a1 = setup_3dgrid[3]
    s1 = setup_3dgrid[4]
    pvs1 = setup_3dgrid[5]

    g2 = g.copy()

    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=True,new_prop=False,properties=0,spec_ax=-1)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]] == pv[1]

    # Make sure the other way of doing it gives the same result
    g2.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=-1)
    assert np.all(g.grid==g2.grid)

def test_spec_axis(setup_3dgrid):
    g = setup_3dgrid[0]
    positions1 = setup_3dgrid[1]
    x1 = setup_3dgrid[2]
    a1 = setup_3dgrid[3]
    s1 = setup_3dgrid[4]
    pvs1 = setup_3dgrid[5]

    # Axis = 2 (same as axis = -1)
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=2)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]] == pv[1]

    # Axis = 1
    g.grid[:] = 0
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=1)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        p = (pv[0][0],pv[0][2],pv[0][1])
        assert g.grid[p] == pv[1]

    # Axis = 0
    g.grid[:] = 0
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=0)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        p = (pv[0][2],pv[0][0],pv[0][1])
        assert g.grid[p] == pv[1]

def test_spec_axis_eval_as_loop(setup_3dgrid):
    g = setup_3dgrid[0]
    positions1 = setup_3dgrid[1]
    x1 = setup_3dgrid[2]
    a1 = setup_3dgrid[3]
    s1 = setup_3dgrid[4]
    pvs1 = setup_3dgrid[5]

    # Axis = 2 (same as axis = -1)
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=True,new_prop=False,properties=0,spec_ax=2)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]] == pv[1]

    # Axis = 1
    g.grid[:] = 0
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=True,new_prop=False,properties=0,spec_ax=1)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        p = (pv[0][0],pv[0][2],pv[0][1])
        assert g.grid[p] == pv[1]

    # Axis = 0
    g.grid[:] = 0
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=True,new_prop=False,properties=0,spec_ax=0)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        p = (pv[0][2],pv[0][0],pv[0][1])
        assert g.grid[p] == pv[1]

def test_new_prop(setup_3dgrid):
    g = setup_3dgrid[0]
    positions1 = setup_3dgrid[1]
    x1 = setup_3dgrid[2]
    a1 = setup_3dgrid[3]
    s1 = setup_3dgrid[4]
    pvs1 = setup_3dgrid[5]

    # Basic delta function test - cumulative version
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=True,properties=0,spec_ax=-1)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]+(1,)] == pv[1]
        assert g.grid[pv[0]+(0,)] == 0

    # Check that this plays well with eval_as_loop
    g.grid[:] = 0
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=True,new_prop=True,properties=0,spec_ax=-1)
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]+(2,)] == pv[1]
        assert g.grid[pv[0]+(1,)] == 0
        assert g.grid[pv[0]+(0,)] == 0

def test_prop_spec(setup_3dgrid):
    g = setup_3dgrid[0]
    positions1 = setup_3dgrid[1]
    x1 = setup_3dgrid[2]
    a1 = setup_3dgrid[3]
    s1 = setup_3dgrid[4]
    pvs1 = setup_3dgrid[5]

    g.add_new_prop()
    g.add_new_prop()

    # Add in first property
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=-1)
    
    assert np.sum(g.grid) == s1
    for pv in pvs1:
        assert g.grid[pv[0]+(0,)] == pv[1]
        assert g.grid[pv[0]+(1,)] == 0
        assert g.grid[pv[0]+(2,)] == 0

    # Add in second property slot
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=1,spec_ax=-1)
    
    assert np.sum(g.grid) == s1*2
    for pv in pvs1:
        assert g.grid[pv[0]+(0,)] == pv[1]
        assert g.grid[pv[0]+(1,)] == pv[1]
        assert g.grid[pv[0]+(2,)] == 0

    # Add in second and third property slot
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=[1,2],spec_ax=-1)
    
    assert np.sum(g.grid) == s1*4
    for pv in pvs1:
        assert g.grid[pv[0]+(0,)] == pv[1]
        assert g.grid[pv[0]+(1,)] == pv[1]*2
        assert g.grid[pv[0]+(2,)] == pv[1]

    # Check that new_prop overrides properties
    assert g.grid.shape == (5,5,5,3)
    assert g.n_properties == 3
    g.add_from_spec_func(positions=positions1,spec_function=delta_cum,spec_function_arguments=np.array([x1,a1]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=True,properties=[1,2],spec_ax=-1)
    
    assert np.sum(g.grid) == s1*5
    assert g.grid.shape == (5,5,5,4)
    assert g.n_properties == 4
    for pv in pvs1:
        assert g.grid[pv[0]+(0,)] == pv[1]
        assert g.grid[pv[0]+(1,)] == pv[1]*2
        assert g.grid[pv[0]+(2,)] == pv[1]
        assert g.grid[pv[0]+(3,)] == pv[1]

def test_basic(setup_3dgrid):
    g = setup_3dgrid[0]
    spec_ax = 2

    # Get size of thing we're replacing:
    s = 2*np.prod(np.delete(g.n_pixels,spec_ax))

    # Random positions in grid
    positions = np.random.rand(s,2)*5
    x = np.random.rand(s)*5
    a = np.ones(s)
    ps = [np.floor(np.concatenate((positions[i],[x[i]]))).astype(int) for i in range(s)]
    pvs = [[tuple(ps[i]),a[i]] for i in range(s)]

    # Make sure the loop will be triggered:
    chunk_size = np.prod(np.delete(g.n_pixels,spec_ax)) # maximum number of spectral channels computed equals size of the grid
    loops = np.ceil(len(positions)/chunk_size)
    assert loops > 1

    # Basic delta function test with larger number of points
    g.add_from_spec_func(positions=positions,spec_function=delta_cum,spec_function_arguments=np.array([x,a]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=spec_ax,careful_with_memory=True)
    assert np.sum(g.grid) == s
    for pv in pvs:
        assert g.grid[pv[0]] >= 1

    # Check the result matches if careful_with_memory is false
    g.grid[:] = 0
    g.add_from_spec_func(positions=positions,spec_function=delta_cum,spec_function_arguments=np.array([x,a]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=False,properties=0,spec_ax=spec_ax,careful_with_memory=False)
    assert np.sum(g.grid) == s
    for pv in pvs:
        assert g.grid[pv[0]] >= 1

    # Check the result matches if eval_as_loop is used
    g.grid[:] = 0
    g.add_from_spec_func(positions=positions,spec_function=delta_cum,spec_function_arguments=np.array([x,a]).T,
                         is_cumulative=True,eval_as_loop=True,new_prop=False,properties=0,spec_ax=spec_ax,careful_with_memory=True)
    assert np.sum(g.grid) == s
    for pv in pvs:
        assert g.grid[pv[0]] >= 1

    # Check new_prop
    g.add_from_spec_func(positions=positions,spec_function=delta_cum,spec_function_arguments=np.array([x,a]).T,
                         is_cumulative=True,eval_as_loop=False,new_prop=True,properties=0,spec_ax=spec_ax,careful_with_memory=True)
    assert np.sum(g.grid) == 2*s
    assert g.grid.shape == (5,5,5,2)
    assert g.n_properties == 2
    for pv in pvs:
        assert g.grid[pv[0]+(0,)] >= 1
        assert g.grid[pv[0]+(1,)] >= 1
    assert np.all(g.grid[...,0]==g.grid[...,1])
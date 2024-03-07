import pytest
import numpy as np

from simim.instrument import Instrument, Detector
from simim.map import Grid
from simim.instrument.spectral_response import gauss_response
from simim.instrument.spatial_response import gauss_psf, gauss_psf_freq_dependent

# Tests for adding grids

def test_check_detectors():
    i1 = Instrument(default_spatial_response='gauss',
                    default_spectral_response='gauss',
                    default_noise_function='none',
                    default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                    default_spectral_kwargs={'fwhm':1e9,'freq0':100e9},
                    default_noise_kwargs={})

    i1.add_detector()
    i1.add_detector('lary')

    i1._check_detectors('0')
    i1._check_detectors('lary')
    i1._check_detectors('0','lary')
    with pytest.raises(Exception):
        i1._check_detectors('1')
    with pytest.raises(Exception):
        i1._check_detectors('0','1')
    
    i1.del_detectors('lary')
    i1._check_detectors('0')
    with pytest.raises(Exception):
        i1._check_detectors('lary')


def test_add_remove_detectors():
    i1 = Instrument(default_spatial_response='gauss',
                    default_spectral_response='gauss',
                    default_noise_function='none',
                    default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                    default_spectral_kwargs={'fwhm':1e9,'freq0':100e9},
                    default_noise_kwargs={})

    i1.add_detector()
    assert i1.detector_names == ['0']
    
    i1.del_detectors('0')
    assert i1.detector_names == []
    assert i1.detector_counter == 1

    i1.add_detector('alfie')
    i1.add_detector('µ')
    i1.add_detector()
    assert i1.detector_names == ['alfie','µ','3']
    assert i1.detector_counter == 4

    i1.del_detectors('3')
    assert i1.detector_names == ['alfie','µ']
    assert i1.detector_counter == 4

    i1.del_detectors('µ','alfie')
    assert i1.detector_names == []
    assert i1.detector_counter == 4

def test_change_response():
    i1 = Instrument(default_spatial_response='gauss',
                    default_spectral_response='gauss',
                    default_noise_function='none',
                    default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                    default_spectral_kwargs={'fwhm':1e9,'freq0':100e9},
                    default_noise_kwargs={})

    i1.add_detector()
    i1.add_detector(spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi})
    i1.add_detector(spatial_response='gauss',spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi})

    assert i1.detectors['0'].spatial_response == gauss_psf
    assert i1.detectors['0'].spatial_kwargs == i1.default_spatial_kwargs
    assert i1.detectors['0'].spatial_kwargs is not i1.default_spatial_kwargs

    assert i1.detectors['1'].spatial_response == gauss_psf
    assert i1.detectors['1'].spatial_kwargs == i1.default_spatial_kwargs
    assert i1.detectors['1'].spatial_kwargs is not i1.default_spatial_kwargs

    assert i1.detectors['2'].spatial_response == gauss_psf
    assert i1.detectors['2'].spatial_kwargs == i1.default_spatial_kwargs
    assert i1.detectors['2'].spatial_kwargs is not i1.default_spatial_kwargs

    i1.set_spatial_respones('instrument',spatial_response='specgauss')
    assert i1.default_spatial_response == gauss_psf_freq_dependent
    assert i1.default_spatial_kwargs is None
    assert i1.detectors['0'].spatial_response != i1.default_spatial_kwargs
    
    i1.set_spatial_respones('instrument',spatial_response='gauss')
    i1.set_spatial_respones(spatial_response='specgauss')
    assert i1.detectors['0'].spatial_response == gauss_psf_freq_dependent
    assert i1.detectors['0'].spatial_kwargs == {}
    assert i1.detectors['1'].spatial_response == gauss_psf_freq_dependent
    assert i1.detectors['1'].spatial_kwargs == {}
    assert i1.detectors['2'].spatial_response == gauss_psf_freq_dependent
    assert i1.detectors['2'].spatial_kwargs == {}

    assert i1.default_spatial_response == gauss_psf
    assert i1.default_spatial_kwargs is None
    assert i1.detectors['0'].spatial_response != i1.default_spatial_response

    i1.set_spatial_respones('instrument','0',spatial_response='gauss',spatial_kwargs={})
    assert i1.default_spatial_response == gauss_psf
    assert i1.default_spatial_kwargs == {}
    assert i1.detectors['0'].spatial_response == i1.default_spatial_response
    assert i1.detectors['0'].spatial_kwargs == i1.default_spatial_kwargs
    assert i1.detectors['0'].spatial_kwargs is not i1.default_spatial_kwargs


def test_detector_setups():
    i1 = Instrument(default_spatial_response='gauss',
                default_spectral_response='gauss',
                default_noise_function='none',
                default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                default_spectral_kwargs={'fwhm':1e9,'freq0':100e9},
                default_noise_kwargs={})
    
    assert i1.default_spectral_response == gauss_response

    # Use all defaults
    i1.add_detector(name='default',nominal_frequency=1e9,)

    # Overwrite kwargs
    i1.add_detector(name='kwargs',
                    nominal_frequency=2e9,
                    spatial_kwargs={'fwhmx':2/60/180*np.pi,'fwhmy':2/60/180*np.pi},
                    spectral_kwargs={'fwhm':2e9,'freq0':200e9},
                    noise_kwargs={'a':0})

    # Overwrite functions, no kwargs
    i1.add_detector(name='functions',
                    nominal_frequency=1e9,
                    spatial_response='specgauss',
                    spectral_response='boxcar',
                    noise_function='white')


    # Overwrite functions, no kwargs
    i1.add_detector(name='functions kwargs',
                    nominal_frequency=2e9,
                    spatial_response='specgauss',spatial_kwargs={'fwhmx':2/60/180*np.pi,'fwhmy':2/60/180*np.pi},
                    spectral_response='boxcar',spectral_kwargs={'fwhm':2e9,'freq0':200e9},
                    noise_function='white',noise_kwargs={'a':0})

    assert i1.detectors['default'].spectral_response == i1.default_spectral_response
    assert i1.detectors['kwargs'].spectral_response == i1.default_spectral_response
    assert i1.detectors['functions'].spectral_response != i1.default_spectral_response
    assert i1.detectors['functions kwargs'].spectral_response != i1.default_spectral_response
    assert i1.detectors['functions kwargs'].spectral_response == i1.detectors['functions'].spectral_response

    assert i1.detectors['default'].spectral_kwargs == i1.default_spectral_kwargs
    assert i1.detectors['default'].spectral_kwargs is not i1.default_spectral_kwargs
    assert i1.detectors['kwargs'].spectral_kwargs != i1.default_spectral_kwargs
    assert i1.detectors['functions'].spectral_kwargs != i1.default_spectral_kwargs
    assert i1.detectors['functions kwargs'].spectral_kwargs != i1.default_spectral_kwargs
    assert i1.detectors['functions kwargs'].spectral_kwargs != i1.detectors['functions'].spectral_kwargs
    assert i1.detectors['functions'].spectral_kwargs == {}
    assert i1.detectors['functions'].spectral_kwargs is not i1.default_spectral_kwargs

    assert i1.detectors['default'].spatial_response == i1.default_spatial_response
    assert i1.detectors['kwargs'].spatial_response == i1.default_spatial_response
    assert i1.detectors['functions'].spatial_response != i1.default_spatial_response
    assert i1.detectors['functions kwargs'].spatial_response != i1.default_spatial_response
    assert i1.detectors['functions kwargs'].spatial_response == i1.detectors['functions'].spatial_response

    assert i1.detectors['default'].spatial_kwargs == i1.default_spatial_kwargs
    assert i1.detectors['default'].spatial_kwargs is not i1.default_spatial_kwargs
    assert i1.detectors['kwargs'].spatial_kwargs != i1.default_spatial_kwargs
    assert i1.detectors['functions'].spatial_kwargs != i1.default_spatial_kwargs
    assert i1.detectors['functions kwargs'].spatial_kwargs != i1.default_spatial_kwargs
    assert i1.detectors['functions kwargs'].spatial_kwargs != i1.detectors['functions'].spatial_kwargs
    assert i1.detectors['functions'].spatial_kwargs == {}
    assert i1.detectors['functions'].spatial_kwargs is not i1.default_spatial_kwargs


    assert i1.detectors['default'].noise_function == i1.default_noise_function
    assert i1.detectors['kwargs'].noise_function == i1.default_noise_function
    assert i1.detectors['functions'].noise_function != i1.default_noise_function
    assert i1.detectors['functions kwargs'].noise_function != i1.default_noise_function
    assert i1.detectors['functions kwargs'].noise_function == i1.detectors['functions'].noise_function

    assert i1.detectors['default'].noise_kwargs == i1.default_noise_kwargs
    assert i1.detectors['default'].noise_kwargs is not i1.default_noise_kwargs
    assert i1.detectors['kwargs'].noise_kwargs != i1.default_noise_kwargs
    assert i1.default_noise_kwargs == {}
    assert i1.detectors['functions'].noise_kwargs == {}
    assert i1.detectors['functions'].noise_kwargs is not i1.default_noise_kwargs

def test_empty_detector_init():
    i1 = Instrument()
    with pytest.raises(Exception):
        i1.add_detector()

    i1.add_detector(spatial_response='gauss',
                    spectral_response='gauss',
                    noise_function='none')


def test_autofreq_fail():
    i1 = Instrument(default_spatial_response='gauss',default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                    default_spectral_response='gauss',
                    default_noise_function='none',default_noise_kwargs={})
    
    f = 0
    with pytest.warns():
        i1.add_detector(spectral_kwargs={'fwhm':1,'freq0':f})
    assert np.isnan(i1.detectors['0'].reffreq)

    i1.add_detector(spectral_kwargs={'fwhm':1,'freq0':f}, nominal_frequency=0)
    print(i1.detectors)
    assert i1.detectors['1'].reffreq == 0

def test_autofreq_gauss():
    i1 = Instrument(default_spatial_response='gauss',default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                    default_spectral_response='gauss',
                    default_noise_function='none',default_noise_kwargs={})
    
    f = 101e9
    i1.add_detector(spectral_kwargs={'fwhm':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['0'].reffreq)) < f/100

    f = 129e9
    i1.add_detector(spectral_kwargs={'fwhm':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['1'].reffreq)) < f/100

    f = 314.15e9
    i1.add_detector(spectral_kwargs={'fwhm':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['2'].reffreq)) < f/100

    f = 314.15
    i1.add_detector(spectral_kwargs={'fwhm':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['3'].reffreq)) < f/100

    f = 314.15e-9
    i1.add_detector(spectral_kwargs={'fwhm':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['4'].reffreq)) < f/100

    f = 314.15e15
    i1.add_detector(spectral_kwargs={'fwhm':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['5'].reffreq)) < f/100

def test_autofreq_box():
    i1 = Instrument(default_spatial_response='gauss',default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                    default_spectral_response='boxcar',
                    default_noise_function='none',default_noise_kwargs={})
    
    f = 101e9
    i1.add_detector(spectral_kwargs={'width':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['0'].reffreq)) < f/100

    f = 129e9
    i1.add_detector(spectral_kwargs={'width':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['1'].reffreq)) < f/100

    f = 314.15e9
    i1.add_detector(spectral_kwargs={'width':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['2'].reffreq)) < f/100

    f = 314.15
    i1.add_detector(spectral_kwargs={'width':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['3'].reffreq)) < f/100

    f = 314.15e-9
    i1.add_detector(spectral_kwargs={'width':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['4'].reffreq)) < f/100

    f = 314.15e15
    i1.add_detector(spectral_kwargs={'width':f/100,'freq0':f})
    assert np.abs((f-i1.detectors['5'].reffreq)) < f/100



# Errors should be raised for Grid without initialized values
def test_field_init():
    i1 = Instrument()
    g = Grid(1, (5,5,5), (10,10,10), (1,1,1))
    with pytest.raises(Exception):
        i1.add_field(g)

# Errors should be raised for Grid not in 3d
def test_field_3d():
    i1 = Instrument()
    g = Grid(1, (5,5), (10,10), (2,2))
    g.init_grid()
    with pytest.raises(Exception):
        i1.add_field(g)

    g = Grid(1, (5,5,5,5), (10,10,10,10), (2,2,2,2))
    g.init_grid()
    with pytest.raises(Exception):
        i1.add_field(g)

# Errors should be raised for Grid not having the requested property index
def test_field_index():
    i1 = Instrument(spectral_unit='Hz',spatial_unit='rad',flux_unit='Jy')
    g = Grid(1, (5,5,5), (10,10,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()
    i1.add_field(g,field_property_idx=0)
    with pytest.raises(Exception):
        i1.add_field(g,field_property_idx=1)
    with pytest.raises(Exception):
        i1.add_field(g,field_property_idx=-1)

# Errors should be raised for Grid not having high enough spatial resolution in 
# any dimension
def test_field_res():
    i1 = Instrument(spectral_unit='Hz',spatial_unit='rad',flux_unit='Jy',best_spatial_res=4,best_spectral_res=4)
    
    # This should work, but raise a warning
    g = Grid(1, (5,5,5), (10,10,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()
    with pytest.warns():
        i1.add_field(g)

    # Spectral too large
    g = Grid(1, (5,5,5), (10,10,10), (1,1,5), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()
    with pytest.raises(Exception):
        with pytest.warns(): # Also raises a warning since spatial pixel check happens first
            i1.add_field(g)

    # Spatial too large
    g = Grid(1, (5,5,5), (10,10,10), (1,5,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()
    with pytest.raises(Exception):
        i1.add_field(g)

    g = Grid(1, (5,5,5), (10,10,10), (5,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()
    with pytest.raises(Exception):
        i1.add_field(g)

    g = Grid(1, (5,5,5), (10,10,10), (5,5,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()
    with pytest.raises(Exception):
        i1.add_field(g)

# check_fields should know what fields have been added
def test_check_fields():
    i1 = Instrument(spatial_unit='rad',spectral_unit='Hz',flux_unit='Jy')
    g = Grid(1, (5,5,5), (10,10,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()

    i1.add_field(g)
    i1._check_fields('0')
    with pytest.raises(Exception):
        i1._check_fields('1')
    with pytest.raises(Exception):
        i1._check_fields('0','1')

    i1.del_fields('0')
    with pytest.raises(Exception):
        i1._check_fields('0')

# Various checks of adding and removing fields
def test_add_remove_fields():
    i1 = Instrument(spatial_unit='rad',spectral_unit='Hz',flux_unit='Jy')
    g = Grid(1, (5,5,5), (10,10,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()

    i1.add_field(g)
    assert i1.fields['0'] is g
    assert np.all(i1.fields['0'].grid == g.grid)

    i1.add_field(g,in_place=False)
    assert i1.fields['1'] is not g
    assert np.all(i1.fields['1'].grid == g.grid)

    assert i1.field_names == ['0','1']
    assert len(i1.fields) == 2
    assert i1.field_counter == 2

    i1.del_fields('1')
    assert i1.field_names == ['0']
    assert len(i1.fields) == 1
    assert i1.field_counter == 2

    i1.add_field(g,name='tony',field_property_idx=0)
    assert i1.field_names == ['0','tony']
    assert len(i1.fields) == 2
    assert i1.field_counter == 3


# Make sure objects with same psf are identified correctly
def test_detector_setups():
    i1 = Instrument(default_spatial_response='gauss',
                default_spectral_response='gauss',
                default_noise_function='none',
                default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                default_spectral_kwargs={'fwhm':1e9,'freq0':100e9},
                default_noise_kwargs={})
    
    # Use all defaults
    i1.add_detector(name='default',nominal_frequency=1e9,)

    # Overwrite kwargs w identical values
    i1.add_detector(name='kwargs',
                    nominal_frequency=1e9,
                    spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi})

    # Overwrite functions, with identical values
    i1.add_detector(name='functions',
                    nominal_frequency=1e9,
                    spatial_response='gauss', spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi})

    # Something different
    i1.add_detector(name='dif1',nominal_frequency=1e9,
                    spatial_response='specgauss', spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi,'freq0':100e9})
    
    # Same thing, with function passed differently
    i1.add_detector(name='dif2',nominal_frequency=1e9,
                    spatial_response=gauss_psf_freq_dependent, spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi,'freq0':100e9})

    # Different kwargs
    i1.add_detector(name='difkw1',
                    nominal_frequency=1e9,
                    spatial_kwargs={'fwhmx':2/60/180*np.pi,'fwhmy':2/60/180*np.pi})
    i1.add_detector(name='difkw2',
                    nominal_frequency=1e9,
                    spatial_kwargs={'fwhmx':2/60/180*np.pi,'fwhmy':2/60/180*np.pi})
    
    # Test
    clones = i1._find_spatial_clones()

    assert clones == [['default','kwargs','functions'],['dif1','dif2'],['difkw1','difkw2']]

def test_beam():
    i1 = Instrument(default_spatial_response='gauss',
            default_spectral_response='gauss',
            default_noise_function='none',
            default_spatial_kwargs={'fwhmx':1,'fwhmy':1},
            default_spectral_kwargs={'fwhm':1,'freq0':100},
            default_noise_kwargs={},
            best_spatial_res=5,
            spatial_unit='rad',spectral_unit='Hz',flux_unit='Jy')
    
    i1.add_detector(nominal_frequency=100)

    g = Grid(1, (50,50,100), (100,100,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()
    i1.add_field(g)

    beam = i1._setup_beam(g,i1.detectors['0'].spatial_response,i1.detectors['0'].spatial_kwargs,'peak')
    assert beam.n_dimensions == 3
    assert beam.n_properties == 1
    assert beam.grid.ndim == 4
    assert np.all(beam.pixel_size == g.pixel_size)
    assert beam.side_length[2] == g.side_length[2]
    assert beam.grid.shape == (51,51,10,1)
    assert beam.grid.max() == 1

    # Do a second time to make sure nothing weird happens
    beam = i1._setup_beam(g,i1.detectors['0'].spatial_response,i1.detectors['0'].spatial_kwargs,'peak')
    assert beam.n_dimensions == 3
    assert beam.n_properties == 1
    assert beam.grid.ndim == 4
    assert np.all(beam.pixel_size == g.pixel_size)
    assert beam.side_length[2] == g.side_length[2]
    assert beam.grid.shape == (51,51,10,1)
    assert beam.grid.max() == 1

def test_map():
    i1 = Instrument(default_spatial_response='gauss',
        default_spectral_response='gauss',
        default_noise_function='none',
        default_spatial_kwargs={'fwhmx':1,'fwhmy':1},
        default_spectral_kwargs={'fwhm':1,'freq0':100},
        default_noise_kwargs={},
        best_spatial_res=5,
        spatial_unit='rad',spectral_unit='Hz',flux_unit='Jy')
    
    i1.add_detector(name='det',nominal_frequency=100)

    g = Grid(1, (50,50,100), (100,100,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g.init_grid()
    i1.add_field(g,name='f1')

    i1.map_fields('f1')

    assert i1.maps_consistent['f1']
    assert np.all([type(i1.maps[f])==Grid for f in i1.maps])
    assert np.all([i1.maps[f].grid.ndim==3 for f in i1.maps])
    assert np.all([i1.maps[f].grid.shape[2]==1 for f in i1.maps])

    g2 = Grid(1, (50,50,100), (100,100,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g2.init_grid()
    i1.add_field(g2,name='f2')

    assert i1.maps_consistent['f1']
    assert i1.maps_consistent['f2'] == False

    i1.map_fields()

    assert i1.maps_consistent['f1']
    assert i1.maps_consistent['f2']
    assert np.all([type(i1.maps[f])==Grid for f in i1.maps])
    assert np.all([i1.maps[f].grid.ndim==3 for f in i1.maps])
    assert np.all([i1.maps[f].grid.shape[2]==1 for f in i1.maps])

    i1.add_detector(name='det2',nominal_frequency=100)
    
    assert i1.maps_consistent['f1'] == False
    assert i1.maps_consistent['f2'] == False

    i1.map_fields()
    assert i1.maps_consistent['f1']
    assert i1.maps_consistent['f2']
    assert np.all([type(i1.maps[f])==Grid for f in i1.maps])
    assert np.all([i1.maps[f].grid.ndim==3 for f in i1.maps])
    assert np.all([i1.maps[f].grid.shape[2]==2 for f in i1.maps])

def test_sample():

    i1 = Instrument(default_spatial_response='gauss',
        default_spectral_response='gauss',
        default_noise_function='white',
        default_spatial_kwargs={'fwhmx':1,'fwhmy':1},
        default_spectral_kwargs={'fwhm':1,'freq0':100},
        default_noise_kwargs={'rms':1},
        best_spatial_res=5,
        spatial_unit='rad',spectral_unit='Hz',flux_unit='Jy')
    
    i1.add_detector(name='det1',nominal_frequency=100)
    i1.add_detector(name='det2',nominal_frequency=100)

    g1 = Grid(1, (50,50,100), (100,100,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g1.init_grid()
    i1.add_field(g1,name='f1')
    g2 = Grid(1, (50,50,100), (100,100,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g2.init_grid()
    i1.add_field(g2,name='f2')

    i1.map_fields()
    
    positions = np.stack((np.linspace(0,100,100),np.linspace(20,80,100)),axis=1)
    samples = i1.sample_fields(positions,'f1',dt=1)
    assert type(samples) == np.ndarray
    assert samples.shape == (100,2)

    samples = i1.sample_fields(positions,dt=1)
    assert type(samples) == dict
    assert list(samples.keys()) == i1.field_names
    assert samples['f1'].shape == (100,2)
    assert samples['f2'].shape == (100,2)

def test_sample_noise():

    i1 = Instrument(default_spatial_response='gauss',
        default_spectral_response='gauss',
        default_noise_function='white',
        default_spatial_kwargs={'fwhmx':1,'fwhmy':1},
        default_spectral_kwargs={'fwhm':1,'freq0':100},
        default_noise_kwargs={'rms':1},
        best_spatial_res=5,
        spatial_unit='rad',spectral_unit='Hz',flux_unit='Jy')
    
    i1.add_detector(name='det1',nominal_frequency=100)

    g1 = Grid(1, (50,50,100), (100,100,10), (1,1,1), axunits=['rad','rad','Hz'], gridunits='Jy')
    g1.init_grid()
    i1.add_field(g1,name='f1')
    i1.map_fields()
    
    positions = np.stack((np.linspace(10,90,100),np.linspace(20,80,100)),axis=1)
    samples = i1.sample_fields(positions,'f1',dt=1,sample_noise=False)
    assert type(samples) == np.ndarray
    assert samples.shape == (100,1)
    assert np.all(samples==0)

    # Error when dt not specified and noise requested
    with pytest.raises(Exception):
        samples = i1.sample_fields(positions,'f1',sample_noise=True)

    samples = i1.sample_fields(positions,'f1',dt=1,sample_noise=True)
    assert type(samples) == np.ndarray
    assert samples.shape == (100,1)
    assert np.all(samples==0) == False


def test_detector():
    i1 = Instrument(default_spatial_response='gauss',
                    default_spectral_response='gauss',
                    default_noise_function='none',
                    default_spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                    default_spectral_kwargs={'fwhm':1e9,'freq0':100e9},
                    default_noise_kwargs={})
    i1.add_detector()

    i2 = Detector(spatial_response='gauss',spectral_response='gauss',noise_function='none',
                  spatial_kwargs={'fwhmx':1/60/180*np.pi,'fwhmy':1/60/180*np.pi},
                  spectral_kwargs={'fwhm':1e9,'freq0':100e9},
                  noise_kwargs={})
    
    assert i1.detectors['0'].spectral_response == i2.detectors['detector'].spectral_response
    assert i1.detectors['0'].spatial_response == i2.detectors['detector'].spatial_response
    assert i1.detectors['0'].noise_function == i2.detectors['detector'].noise_function

    assert i1.detectors['0'].spectral_kwargs == i2.detectors['detector'].spectral_kwargs
    assert i1.detectors['0'].spatial_kwargs == i2.detectors['detector'].spatial_kwargs
    assert i1.detectors['0'].noise_kwargs == i2.detectors['detector'].noise_kwargs

import pytest
import numpy as np

from simim.instrument import Instrument
from simim.map import Grid
from simim.instrument.spectral_response import gauss_response, boxcar_response
from simim.instrument.spatial_response import gauss_psf, gauss_psf_freq_dependent
from simim.instrument.noise_functions import white_noise, zero_noise

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

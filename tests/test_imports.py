def test_import_simim():
    """Make sure the expected files are imported - simim main"""

    import simim

    resources = dir(simim)
    for check_resource in ['setupsimim','siminterface','galprops','map']:
        assert check_resource in resources

def test_import_siminterface():
    """Make sure the expected files are imported - simim.siminterface"""

    import simim.siminterface

    resources = dir(simim.siminterface)
    for check_resource in ['IllustrisCatalogs','UniversemachineCatalogs','UchuuCatalogs','SnapHandler','SimHandler']:
        assert check_resource in resources

def test_import_galprops():
    """Make sure the expected files are imported - simim.galprops"""

    import simim.galprops

    resources = dir(simim.galprops)
    for check_resource in ['Prop','MultiProp','prop_behroozi_sfr','prop_li_co']:
        assert check_resource in resources

def test_import_map():
    """Make sure the expected files are imported - simim.map"""

    import simim.map

    resources = dir(simim.map)
    for check_resource in ['Grid','Gridder','LoadGrid','GridFromAxes','GridFromAxesAndFunction','PSF','SpectralPSF','gridder_function',]:
        assert check_resource in resources

def test_import_map():
    """Make sure the expected files are imported - simim.lightcone"""

    import simim.lightcone

    resources = dir(simim.lightcone)
    for check_resource in ['LCMaker','LCHandler']:
        assert check_resource in resources

def test_import_inst():
    """Make sure the expected files are imported - simim.instrument"""

    import simim.instrument

    resources = dir(simim.instrument)
    for check_resource in ['Instrument','Detector','gauss_psf','gauss_psf_freq_dependent','gauss_response','boxcar_response','white_noise','zero_noise']:
        assert check_resource in resources
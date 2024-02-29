def test_import_simim():
    """Make sure the expected files are imported - simim main"""

    import simim

    resources = dir(simim)
    for check_resource in ['setupsimim','siminterface']:
        assert check_resource in resources

def test_import_siminterface():
    """Make sure the expected files are imported - simim.siminterface"""

    import simim.siminterface

    resources = dir(simim.siminterface)
    for check_resource in ['IllustrisCatalogs','UniversemachineCatalogs','SnapHandler','SimHandler']:
        assert check_resource in resources


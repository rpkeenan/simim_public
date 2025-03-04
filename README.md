SimIM
=====

SimIM is a tool for simulating the extra-galactic sky in radio, sub-mm and
far-infrared. Full documentation is available via [Read the
Docs](https://simim.readthedocs.io). Source code is avialable via
[GitHub](https://github.com/rpkeenan/simim_public).

This is Version 0.3, which includes utilities for downloading and formatting
simulation data, adding new properties such as spectral line emission to
simulations, visualizing basic data products, constructing light cones, making
pixelized maps or data cubes, and mocking instruments/timestreams. A future
release will add more detailed documentation with examples of all functionality
and relevant scientific applications.

What's Here
-----------

1. simim directory - source code for the SimIM package
2. setupsimim.py - script for completing SimIM setup when building from GitHub source code
3. docs directory - code for building documentation
4. tests directory - unit tests; these cover more recent additions to the SimIM source code
but are relatively sparse for the galprops, lightcone, and siminterface modules.
5. pyproject.toml, .readthedocs.yaml, .gitignore - config files for pip,
   readthedocs, and git

Change Log
----------
Version 0.3.8
- Add support for UniverseMachine galaxy catalogs for Uchuu the simulations (docs forthcoming)
- Minor bug fixes

Version 0.3.7
- Development update - making a few timestream related features work better for beta testers

Version 0.3.6
- Correct critical bug in downloading TNG simulation data

Version 0.3.5
- Update ascii loader for UniverseMachine to use less memory and hopefully run faster
- Gridder for spectra
- Minor bug fixes and some tests

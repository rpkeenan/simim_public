The ``map`` Module
==================

The **SimIM** ``map`` module is used for converting the halo/galaxy 
catalogs contained in simulation and lightcone catalogs into pixelized
maps and/or cubes. It also provides a number of convenient utilities 
for interacting with these cubes once created.

The underlying functionality is implemented in the ``map.Grid`` class.
A number of sub-classes exist for conveniently generating grids for 
different types of objects (e.g. maps or data-cubes of astrophysical
sources, instrument PSFs).

.. currentmodule:: simim.map
.. autoclass:: Grid
    :members:
    :inherited-members:

Creating Data Cubes
-------------------

The ``map.Gridder`` method will grid properties from a catalog of objects
in an arbitrary of number of dimensions to create maps and data cubes.

.. autoclass:: Gridder
    :members:
    :inherited-members:

The ``map.gridder_function`` function works in the same way as ``map.Gridder``,
but instead of creating a ``map.Grid`` instance it simply returns the constructed
grid and axes.

.. autofunction:: gridder_function

PSFs
----

For creating representations of simple (Gaussian) instrumental point-spread
functions, the ``map.PSF`` and ``map.SpectralPSF`` classes are available:

.. autoclass:: PSF
    :members:

.. autoclass:: SpectralPSF
    :members:

For more sophisticated PSFs, the ``map.GridFromAxesAndFunction`` class may be 
useful:

.. autoclass:: GridFromAxesAndFunction
    :members:
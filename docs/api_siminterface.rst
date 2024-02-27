The ``siminterface`` Module
===========================

Interacting with Specific Simulations
-------------------------------------

These are tools for downloading and formatting simulation data from
various sources.

.. currentmodule:: simim.siminterface
.. autoclass:: illustris_catalogs
    :members:
    :inherited-members:

.. autoclass:: universemachine_catalogs
    :members:
    :inherited-members:

Under the Hood
--------------

These features are useful for building installers for new simulations,
which should work by extending the ``simim.siminterface._rawsiminterface.sim_catalogs``
class.

.. currentmodule:: simim.siminterface._rawsiminterface
.. autoclass:: sim_catalogs
    :members:
    :inherited-members:

.. autoclass:: snapshot
    :members:
    :inherited-members:
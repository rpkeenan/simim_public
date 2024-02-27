The ``siminterface`` Module
===========================

Interacting with Simulations Using SimIM
----------------------------------------

Once they are formatted, **SimIM** interfaces with all simulation data
through ``handler`` classes. The ``simhandler`` class provides an 
interface to all data from a simulation, while the ``snaphandler`` class
provides an interface with data from a single simulation snapshot.

.. currentmodule:: simim.siminterface
.. autoclass:: simhandler
    :members:
    :inherited-members:

.. autoclass:: snaphandler
    :members:
    :inherited-members:


Downloading and Formatting Simulations
--------------------------------------

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
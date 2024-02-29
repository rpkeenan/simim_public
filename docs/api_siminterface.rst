The ``siminterface`` Module
===========================

Interacting with Simulations Using SimIM
----------------------------------------

Once they are formatted, **SimIM** interfaces with all simulation data through
"Handler" classes. The ``SimHandler`` class provides an interface to all data
from a simulation, while the ``SnapHandler`` class provides an interface with
data from a single simulation snapshot.

.. currentmodule:: simim.siminterface
.. autoclass:: SimHandler
    :members:
    :inherited-members:

.. autoclass:: SnapHandler
    :members:
    :inherited-members:


Downloading and Formatting Simulations
--------------------------------------

These are tools for downloading and formatting simulation data from various
sources.

.. currentmodule:: simim.siminterface
.. autoclass:: IllustrisCatalogs
    :members:
    :inherited-members:

.. autoclass:: UniversemachineCatalogs
    :members:
    :inherited-members:


Under the Hood
--------------

These features are useful for building installers for new simulations, which
should work by extending the ``simim.siminterface._rawsiminterface.SimCatalogs``
class.

.. currentmodule:: simim.siminterface._rawsiminterface
.. autoclass:: SimCatalogs
    :members:
    :inherited-members:

.. autoclass:: Snapshot
    :members:
    :inherited-members:
The ``galprops`` Module
=======================

The ``galprops`` module is designed to support adding new properties to
simulation halo catalogs. This is done by defining functions which take a given
(existing) set of halo properties, and returns values for a new property derived
from those values. For example a function could return a star formation rate
based on the redshift and dark matter halo mass already in the catalog.

A number of such functions have been implemented and included in **SimIM** as
defaults. It is also possible to define new functions and add them.

Functions defined in the ``galprops`` module can either be used in a standalone
fashion, or can be wrapped using the ``Prop`` and ``MultiProp`` classes then
wrap these functions with instructions that allow ``SimHandler``,
``SnapHandler``, and ``LCHandler`` instances to easily apply them to galaxy
catalogs.

``Prop`` and ``GalProp`` Classes
================================

.. currentmodule:: simim.galprops
.. autoclass:: Prop
    :members:

.. currentmodule:: simim.galprops
.. autoclass:: MultiProp
    :members:

Pre-Packaged Galaxy Properties
==============================

A number prescriptions for assigning galaxy properties - most often SFR, stellar mass, 
or luminosity - are built into **SimIM**. These are drawn from various literature sources.
If these models are used in published work, please make appropriate reference to the 
original source - noted in the documentation for each model.

The prescriptions are implemented both as standalone functions, and as ``Prop`` instances.
The ``Prop`` instances are automatically available within the ``simim.galprops`` module. 
Functional versions must be imported from appropriate sub-modules.

Spectral Line Luminosities
--------------------------

Carbon Monoxide
^^^^^^^^^^^^^^^

.. currentmodule:: simim.galprops
.. automodule:: simim.galprops.line_co
    :members:

.. automodule:: simim.galprops.line_co_dutycycle
    :members:

HCN
^^^

.. automodule:: simim.galprops.line_densegas
    :members:

Far-Infrared
^^^^^^^^^^^^

.. automodule:: simim.galprops.line_fir
    :members:

.. automodule:: simim.galprops.line_fir_yang
    :members:

Star Formation Rates and Stellar Masses
---------------------------------------

.. automodule:: simim.galprops.sfr_behroozi13
    :members:

Helper Functions
----------------

.. automodule:: simim.galprops.log10normal
    :members:


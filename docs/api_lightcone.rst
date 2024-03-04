The ``lightcone`` Module
========================

The **SimIM** ``lightcone`` module is designed to construct light cones - tables
of galaxies with angular position and redshift and redshift dependent galaxy
properties comparable to real astrophysical surveys - out of simulation cubes.
The module has two main components ``lightcone.LCMaker`` which generates new
light cone files, and ``lightcone.LCHandler`` which provides a handler instance
for interacting with and analyzing the lightcones built by
``lightcone.LCMaker``. 

Making Light Cones
------------------

Light cones can inherit any property associated with their parent simulation,
except for those relating to position, which are transformed into coordinates
within the lightcone.

.. currentmodule:: simim.lightcone
.. autoclass:: LCMaker
    :members:
    :inherited-members:

The Light Cone Handler
----------------------

The handler for lightcones is almost entirely analogous to the hander for
simulation snapshots, making interfacing with data in "simulation space" and
"observer space" fairly similar.

.. autoclass:: LCHandler
    :members:
    :inherited-members:

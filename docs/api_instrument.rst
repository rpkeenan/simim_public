The ``instrument`` Module
===========================

Tools For Modeling Instruments
------------------------------

Basic instrument modeling, such as convolving a spectral cube with an instrument
PSF, can be achieved using utilities in **SimIM**'s ``map`` module. For more
sophisticated instrument modeling, the ``instrument`` module and
``simim.instrument.Instrument`` class exist.

The ``Instrument`` class defines an instrument with one or more detectors, which
can then be paired with a field or fields for simulated observations. The
detectors of an instrument are defined in terms of the spatial response (PSF)
and spectral response. Detector noise and pointing offsets between detectors can
also be included.

Once detectors and fields are added to an ``Instrument`` instance, the class has
methods to produce maps of the field as seen by each detector, as well as to
sample the field (and instrument noise) into a timestram at a specified set of
positions

.. currentmodule:: simim.instrument
.. autoclass:: Instrument
    :members:
    :inherited-members:

.. autoclass:: Detector
    :members:


Spatial Response Functions
--------------------------

There are built in functions for creating a Gaussian beam or a Gaussian beam
that varies in size along the frequency axis. Other beams can be specified by
the user by passing an appropriately defined function. 

A valid spatial response function will take as its arguments three 1D arrays -
specifying grids of 1) angular offsets in the RA direction, 2) angular offset in
the dec direction, and 3) frequency/wavelength, and will return a 3D grid
containing the beam evaluated at every 3D grid point from the three axes.
Spatial response functions may have additional parameters, which can then be
specified in the spatial_kwargs parameter when adding a detector to the
``Instrument`` instance.


Spectral Response Functions
---------------------------

There are built in functions for creating Gaussian and boxcar shaped spectral
responses. Other beams can be specified by the user by passing an appropriately
defined function. 

A valid spectral response function will take as it argument a 1D array of
frequency/wavelength coordinates, and return a 1D array containing the spectral
response at each coordinate. Spectral response functions may have additional
parameters, which can then be specified in the spectral_kwargs parameter when
adding a detector to the ``Instrument`` instance.

Noise Functions
---------------

Noise functions can be used to define the instrument noise of a detector which
will be added to the output timestream when generating timestreams. These
functions should take as arguments 1) a number of samples to draw and 2) a
timestep for each sample, and will return specified number of samples drawn from
a noise distribution appropriate for the timestep.

There are builtin functions for white noise or zero noise. Others can be defined
by the user.

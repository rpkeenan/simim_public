Installing ``SimIM``
====================

Installation of ``SimIM`` is simple. Clone the 
``SimIM`` github repo github repo <https://github.com/rpkeenan/simim_release>:

.. code-block:: console

    cd [location where you want to copy the repo]
    git clone https://github.com/rpkeenan/simim_release
    pip install .

which will create a local copy of the software package and install
it using ``pip``.

Once ``SimIM`` is installed it is necessary to specify paths in which
large data files will be stored. This only needs to be done once. A 
script for doing this is included in the github repo - ``setupsimim.py``:

.. code-block:: console

    python setupsimim.py


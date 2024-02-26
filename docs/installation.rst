Installing SimIM
================

From PyPI
---------

**SimIM** is available in the Python Package Index (PyPI) and 
can be installed via pip. 

.. code-block:: console

    $ pip install SimIM

Python 3 is required. The package
has been primarily developed on Python 3.11 and tested on 3.6;
compatibility with Python versions earlier than 3.6 is not 
guaranteed.

Most **SimIM** dependencies are common astronomical/scientific
computing or visualization packages.

Once **SimIM** is installed an additional step is required to 
set up paths for data storage. You should begin by creating a 
directory to hold large data files created and used by 
**SimIM**. Then, in a python interpreter run the following:

.. code-block:: python
    
    >>> from simim import setupsimim
    >>> setupsimim()
    Please specify a path to save data directories.
    Specifying no path will set the path to your home direcotry.
    Path: [path to data directory]
    Files will be saved in [path to data directory]/simim_resources
    Is this okay? y/n: [enter y to continue]

This only needs to be run once and the provided path will be used
by all subsequent uses of the **SimIM** package. Should your data
directory ever move, you'll need to re-run the ``setupsimim`` 
function.

From GitHub
-----------

The source code for **SimIM** is available on the 
`SimIM project GitHub <https://github.com/rpkeenan/simim_public>`_ 
and can be directly cloned and installed.

From the command line:

.. code-block:: console

    $ cd [location where you want to copy the repo]
    $ git clone https://github.com/rpkeenan/simim_public.git
    $ pip install .

Once **SimIM** is installed you will need to setup the path
to the data directory. This can be done in the same manner as
in the previous section or by running the ``setupsimim.py`` script
included in the source code distribution:

.. code-block:: console

    $ python setupsimim.py
    Please specify a path to save data directories.
    Specifying no path will set the path to your home direcotry.
    Path: [path to data directory]
    Files will be saved in [path to data directory]/simim_resources
    Is this okay? y/n: [enter y to continue]
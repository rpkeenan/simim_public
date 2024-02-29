Installing SimIM
================

Requirements
------------

Python 3 is required. The package has been primarily developed on Python 3.11
and tested on 3.6; compatibility with Python versions earlier than 3.6 is not
guaranteed.

Most **SimIM** dependencies are common astronomical/scientific computing or
visualization packages. The PyPI installation will automatically include these
files.

Some **SimIM** features rely on the GNU Wget application. Instructions for
installing it are available on the `GNU Wget webage
<https://www.gnu.org/software/wget/>`_. It can be installed on MacOS via
`homebrew <https://brew.sh/>`_. This is only necessary if you intend to use
**SimIM** to download Illustris/TNG data files.

From PyPI
---------

**SimIM** is available in the Python Package Index (PyPI) and can be installed
via pip. 

.. code-block:: console

    $ pip install simim

Once **SimIM** is installed an additional step is required to set up paths for
data storage. This can by done by running the ``setupsimim`` command in the 
command line:

.. code-block:: console

    $ setupsimim
    Please specify a path to save data directories.
    Specifying no path will set the path to your home directory.
    Path: [path to data directory]
    Files will be saved in [path to data directory]/simim_resources
    Is this okay? y/n: [enter y to continue]

Alternatively, within a Python interpreter this can be accomplished as follows:

.. code-block:: python
    
    >>> from simim import setupsimim
    >>> setupsimim()
    Please specify a path to save data directories.
    Specifying no path will set the path to your home directory.
    Path: [path to data directory]
    Files will be saved in [path to data directory]/simim_resources
    Is this okay? y/n: [enter y to continue]

This script only needs to be run once and the provided path will be used by all
subsequent uses of the **SimIM** package. Should your data directory ever move,
you'll need to re-run the ``setupsimim`` function.

From GitHub
-----------

The source code for **SimIM** is available on the `SimIM project GitHub
<https://github.com/rpkeenan/simim_public>`_ and can be directly cloned and
installed.

From the command line:

.. code-block:: console

    $ cd [location where you want to copy the repo]
    $ git clone https://github.com/rpkeenan/simim_public.git
    $ pip install .

Once **SimIM** is installed you will need to setup the path to the data
directory. This can be done in the same manner as in the previous section or by
running the ``setupsimim.py`` script included in the source code distribution.

.. warning::
    
    When ``setupsimim`` is run within the **SimIM** repo, it may cache the
    location of the data directory in the repo, rather than the installed 
    version of **SimIM** (unless the repo itself is the installed version).
    If you will be executing code using **SimIM** outside the repo, then 
    you should run ``setupsimim`` from a different location.

If you plan to contribute to the **SimIM** repository, a few additional
dependencies exist for building the documentation pages. These are listed in the
``docs/requirements.txt`` file.
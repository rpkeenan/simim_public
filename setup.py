from setuptools import setup, find_packages

setup(name='simim',
      version='2.0',
      description='Code for simulating the radio-submillimeter sky',
      url='https://github.com/rpkeenan/simim',
      author='R P Keenan',
      packages=['simim'],
      include_package_data=True,
      package_data={'':['simim/resources/*.txt']},
      install_requires=[
          'astropy',
          'numpy',
          'scipy',
          'h5py',
          'matplotlib',
          'importlib_resources',
          'requests'
      ]
      )

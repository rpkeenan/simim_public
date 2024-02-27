from setuptools import setup, find_packages

setup(name='simim',
      version='2.0',
      description='Code for simulating the radio-submillimeter sky',
      long_description=open('README.md').read(),
      url='https://github.com/rpkeenan/simim',
      author='R P Keenan',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      package_data={'':['simim/resources/*.txt']},
      python_requires='>3.0',
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
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SimIM'
copyright = '2024, R P Keenan'
author = 'R P Keenan'
release = '2.0'

root_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', # For generating automatic documentation
    'sphinx.ext.autosummary', # For collecting all of the documented objects
    'sphinx.ext.duration', # For timing builds
    'numpydoc', # This allows us to handle a few more docstring formats - specifically numpy's
    'nbsphinx', # This allows for Jupyter notebook integration 
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nbsphinx_execute = 'never' # Never execute Jupyter notebooks when compiling docs


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' # 'sphinxdoc' is similar to what matplotlib uses

html_theme_options = {
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': -1,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_logo = 'images/simim_logo.png'
html_favicon = 'images/simim_favicon.png'

# -- Options for autodoc -------------------------------------------------
autoclass_content = "both"
autodoc_member_order = "bysource"
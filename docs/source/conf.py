# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../xmca'))

vfile = {}
exec(open('../../xmca/version.py').read(), vfile)

# -- Project information -----------------------------------------------------

project = 'xmca'
copyright = '2021, Niclas Rieger'
author = 'Niclas Rieger'

# The full version, including alpha/beta/rc tags
release = vfile['__version__']


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary'
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Material theme options (see theme.conf for more information)
html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'xMCA',

    # Set you GA account ID to enable tracking
    'google_analytics_account': 'UA-XXXXX',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'base_url': 'https://github.com/nicrie/xmca',

    # Set the color and the accent color
    'theme_color': 'blue-grey',
    'color_primary': 'blue-grey',
    'color_accent': 'deep-orange',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/nicrie/xmca',
    'repo_name': 'xMCA',
    'repo_type': 'github',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 1,
    # If False, expand all TOC entries
    'globaltoc_collapse': True,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': False,
}

html_sidebars = {
    # exclude "localtoc.html"
    '**': [
        'logo-text.html', 'globaltoc.html',
        'localtoc.html', 'searchbox.html'
    ]
}

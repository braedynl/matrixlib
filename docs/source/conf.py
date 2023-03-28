# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.

import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Matrices-Py'
copyright = '2023, Braedyn L'
author = 'Braedyn L'
release = '0.3.0'

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []

# Autodoc configuration
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_member_order = "bysource"

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

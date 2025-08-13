# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'spamosaic'
author = 'Jinmiao Lab'
release = 'v1.0.3'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    'nbsphinx',
]
autosummary_generate = False
autosummary_generate_overwrite = False   
templates_path = ['_templates']

# autodoc_default_options = {
#     'members': True,
#     'undoc-members': True,
#     'show-inheritance': True,
# }

exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
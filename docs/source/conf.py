# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from pygments.lexers import PythonLexer

sys.path.insert(0, os.path.abspath("../../"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MAMUT"
copyright = '2025, "Igor Kołodziej, Hubert Kowalski, Julia Przybytniowska"'
author = '"Igor Kołodziej, Hubert Kowalski, Julia Przybytniowska"'
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

pygments_lexers = {
    "ipython3": PythonLexer,
}

nbsphinx_codecell_lexer = "python"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.webp"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]

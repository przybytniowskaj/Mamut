# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from pygments.lexers import PythonLexer

sys.path.insert(0, os.path.abspath("../../"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MAMUT"
copyright = "2025-2026, Igor Kołodziej, Hubert Kowalski, Julia Przybytniowska"
author = "Igor Kołodziej, Hubert Kowalski, Julia Przybytniowska"
try:
    release = package_version("mamut")
except PackageNotFoundError:
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
typehints_fully_qualified = False

pygments_lexers = {
    "ipython3": PythonLexer,
}

nbsphinx_codecell_lexer = "python"
nbsphinx_execute = "never"
language = "en"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_title = f"MAMUT {release} Documentation"
html_short_title = "MAMUT"
canonical_url = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://mamut.readthedocs.io/en/latest/",
).rstrip("/")
html_baseurl = f"{canonical_url}/"
html_logo = "_static/logo.webp"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_extra_path = ["robots.txt"]
html_theme_options = {
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
html_use_opensearch = html_baseurl.rstrip("/")

sitemap_url_scheme = "{link}"
sitemap_locales = [None]
sitemap_excludes = [
    "search.html",
    "genindex.html",
    "py-modindex.html",
    "opensearch.html",
    "_modules/*",
]
sitemap_indent = 2

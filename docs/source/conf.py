# Configuration file for the Sphinx documentation builder.

import os
import sys
import datetime

# Add the parent directory (where PyNetAlign is) to sys.path
sys.path.insert(0, os.path.abspath("../.."))

import PyNetAlign

# -- Project information
project = 'PyNetAlign'
copyright = f'{datetime.datetime.now().year}, Qi Yu'
author = 'Qi Yu'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

add_module_names = False
autosummary_generate = True

# typehints_use_rtype = True
# typehints_defaults = 'comma'

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


def rst_jinja_render(app, _, source):
    """Enable Jinja templating in .rst files."""
    if hasattr(app.builder, "templates"):
        rst_context = {"PyNetAlign": PyNetAlign,
                       "sys": sys}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    """Setup Sphinx to process Jinja templates."""
    app.connect("source-read", rst_jinja_render)

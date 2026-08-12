"""Sphinx configuration for the ADOPPTRS handover documentation.

Sources are Markdown, not reStructuredText, and deliberately so: the .md
files stay readable straight from GitHub, so the documentation exists even
where nobody has run a build. The HTML render is a bonus, not the carrier.

Build:
    pip install -r requirements.txt
    sphinx-build -b html . _build/html

The rendered site lands in _build/, which is gitignored. Note that docs/
already serves the Mapbox detection map at docs/index.html -- do not
build into docs/ itself.
"""

project = 'ADOPPTRS'
author = 'Thomas Brandt'
copyright = '2026, Thomas Brandt -- fork of the project by Francois Rozet'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

source_suffix = {'.md': 'markdown'}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'requirements.txt']

# Tables and admonitions carry most of the measured results, so the
# extensions that make them writable in Markdown are the ones that matter.
myst_enable_extensions = ['colon_fence', 'deflist', 'attrs_inline']

html_theme = 'furo'
html_title = 'ADOPPTRS -- status and handover'

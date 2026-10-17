from importlib.metadata import version as _version

project = "ReSPAIR"
author = "the ReSPAIR developers"
copyright = "2026, the ReSPAIR developers"
release = _version("respair")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
autodoc_mock_imports = ["sklearn"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "ReSPAIR"

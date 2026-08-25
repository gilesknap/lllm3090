"""Configuration file for the Sphinx documentation builder.

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from pathlib import Path
from subprocess import check_output

import lllm3090

project = "lllm3090"
copyright = "2026, Giles Knap"
author = "Giles Knap"

release = lllm3090.__version__
if "+" in release:
    # Not on a tag: use the branch name, which is more useful than a dev hash.
    root = Path(__file__).absolute().parent.parent
    version = check_output("git branch --show-current".split(), cwd=root).decode().strip()
else:
    version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

myst_enable_extensions = ["colon_fence"]

nitpicky = True
nitpick_ignore = [
    ("py:class", "NoneType"),
    ("py:class", "pathlib._local.Path"),
    ("py:class", "typer.models.Context"),
]

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False
autosummary_ignore_module_all = False

templates_path = ["_templates"]
default_role = "any"
master_doc = "index"
exclude_patterns = ["_build"]
pygments_style = "sphinx"

intersphinx_mapping = {"python": ("https://docs.python.org/3/", None)}

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version}"
html_theme_options = {
    "logo": {"text": project},
    "use_edit_page_button": False,
    "github_url": "https://github.com/gilesknap/lllm3090",
    "icon_links": [],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
}
html_show_sourcelink = True

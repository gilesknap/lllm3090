"""Local LLM serving for a single RTX 3090.

A llama.cpp engine, a browser control panel that starts and stops it, and a
curated model list sized against 24 GB of VRAM.
"""

from ._version import __version__

__all__ = ["__version__"]

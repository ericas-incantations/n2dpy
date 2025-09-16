"""Top-level package for normal2disp."""

from . import n2d as _n2d

__all__ = ["n2d", "__version__", "get_version"]

__version__ = _n2d.__version__
get_version = _n2d.get_version

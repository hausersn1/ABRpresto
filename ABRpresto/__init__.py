import json
from pathlib import Path

from . import XCsub


def get_version():
    try:
        from importlib import resources
        from setuptools_scm import get_version
        with resources.path("ABRpresto", "__init__.py") as path:
            repo = Path(path).parent.parent
            return get_version(repo)
    except:
        from ._version import __version__
        return __version__


version = get_version()

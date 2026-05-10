from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from mamut.wrapper import Mamut

try:
    __version__ = package_version("mamut")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0"

__all__ = ["Mamut", "__version__"]

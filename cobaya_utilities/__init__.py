from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cobaya_utilities")
except PackageNotFoundError:
    # package is not installed
    pass

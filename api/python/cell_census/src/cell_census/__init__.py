from importlib.metadata import PackageNotFoundError, version

from .experiment import get_experiment
from .get_anndata import get_anndata
from .open import download_source_h5ad, get_source_h5ad_uri, open_soma
from .presence_matrix import get_presence_matrix
from .release_directory import get_census_version_description, get_census_version_directory

try:
    __version__ = version("cell_census")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-unknown"

__all__ = [
    "download_source_h5ad",
    "get_anndata",
    "get_census_version_description",
    "get_census_version_directory",
    "get_experiment",
    "get_presence_matrix",
    "get_source_h5ad_uri",
    "open_soma",
]

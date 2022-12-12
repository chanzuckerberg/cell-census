import pathlib

import anndata
import cell_census
import numpy as np
import tiledbsoma as soma


def test_open_soma_latest() -> None:
    # There should _always_ be a 'latest'
    census = cell_census.open_soma(census_version="latest")
    assert census is not None
    assert isinstance(census, soma.Collection)

    # and it should always be the default
    assert cell_census.open_soma().uri == census.uri


def test_get_source_h5ad_uri() -> None:
    rng = np.random.default_rng()
    census = cell_census.open_soma(census_version="latest")
    census_datasets = census["census_info"]["datasets"].read_as_pandas_all()
    for idx in rng.choice(np.arange(len(census_datasets)), size=10, replace=False):
        a_dataset = census_datasets.iloc[idx]
        locator = cell_census.get_source_h5ad_uri(a_dataset.dataset_id)
        assert isinstance(locator, dict)
        assert "uri" in locator
        assert locator["uri"].endswith(a_dataset.dataset_h5ad_path)


def test_download_source_h5ad(tmp_path: pathlib.Path) -> None:
    census = cell_census.open_soma(census_version="latest")
    census_datasets = census["census_info"]["datasets"].read_as_pandas_all()
    small_dataset = census_datasets.nsmallest(1, "dataset_total_cell_count").iloc[0]

    adata_path = tmp_path / "adata.h5ad"
    cell_census.download_source_h5ad(small_dataset.dataset_id, adata_path.as_posix(), census_version="latest")
    assert adata_path.exists() and adata_path.is_file()

    ad = anndata.read_h5ad(adata_path.as_posix())
    assert ad is not None
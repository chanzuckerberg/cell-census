# Create a summary cube of Cell Census data, computing gene expression cell counts and sums across all of
# X, summarizing for a specified set of cell dimensions (obs attributes) and genes (var attribute). Processes X in
# batches, building the cube in memory. The cube will only contain rows for "coords" (distinct tuples of obs dimensions
# and gene) that have extant data in X.

# TODO: Make this into a notebook, if it has pedagogical value.

import sys

import cell_census
import pandas as pd
from cell_census.experiment_query import X_as_series, experiment_query

cube_dims_obs = [
    "tissue_ontology_term_id",
    "tissue_general_ontology_term_id",
    "cell_type_ontology_term_id",
    "dataset_id",
    "assay_ontology_term_id",
    "development_stage_ontology_term_id",
    "disease_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
    "sex_ontology_term_id",
]
cube_dims = ["gene_ontology_term_id"] + cube_dims_obs

if __name__ == "__main__":
    census_soma = cell_census.open_soma(uri=sys.argv[1] if len(sys.argv) > 1 else None)

    organisms_to_process = census_soma["census_data"].keys()
    # For each organism
    for organism_label in organisms_to_process:
        organism_census = census_soma["census_data"][organism_label]

        with experiment_query(organism_census, measurement_name="RNA") as query:
            if query.n_obs == 0:
                continue

            var_df = query.var().to_pandas().set_index("soma_joinid")
            obs_df = query.obs(column_names=["soma_joinid"] + cube_dims_obs).to_pandas().set_index("soma_joinid")

            X_stats_all: pd.DataFrame = None
            for X_tbl in query.X("raw"):
                print(f"Processing X batch size={X_tbl.shape[0]}")

                X_df = X_as_series(X_tbl).to_frame(name="raw_count")
                X_with_obs_df = X_df.join(obs_df[cube_dims_obs], on="soma_dim_0")
                X_with_obs_var_df = X_with_obs_df.join(var_df["feature_id"], on="soma_dim_1").rename(
                    columns={"feature_id": "gene_ontology_term_id"}
                )

                # This is the slow step.
                # TODO: Parallelize X batch processing, while ensuring thread-safe accumulation step
                X_stats_batch = X_with_obs_var_df.groupby(cube_dims, sort=False).agg(  # no sorting, for performance
                    ["size", "sum"]
                )
                X_stats_batch.columns = ["n_cells", "sum"]

                # Accumulate all summary stats in-memory, using a DataFrame
                if X_stats_all is None:
                    # first iteration, initialize accumulator
                    X_stats_all = X_stats_batch
                    X_stats_all["sum"] = X_stats_all["sum"].sparse.to_dense()
                else:
                    # Series.add() requires dense array, but maybe there's a better way to do this
                    X_stats_batch["sum"] = X_stats_batch["sum"].sparse.to_dense()
                    X_stats_all = X_stats_all.add(X_stats_batch, fill_value=0)

                print(f"Batch agg rows={X_stats_batch.shape[0]}")
                print(f"Total agg rows={X_stats_all.shape[0]}")

            X_stats_all.to_pickle("census_cube.pkl")
            print(X_stats_all)

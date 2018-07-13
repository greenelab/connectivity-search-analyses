import logging
import os

import pandas
import scipy.special
import scipy.stats

import hetmech.degree_group
import hetmech.degree_weight
import hetmech.hetmat


def compute_save_dwpc(graph, metapath, damping=0.5, dense_threshold=1, dtype='float64', approx_ok=True,
                      return_result=True):
    """
    Returns and saves a DWPC matrix. If already computed and return_result=True, will return saved result.
    Automatically ignores inverses, corrupt files, and previously existing files.
    """
    metapath = graph.metagraph.get_metapath(metapath)
    for inverse in True, False:
        mp = metapath.inverse if inverse else metapath
        try:
            return graph.read_path_counts(mp, 'dwpc', damping)
        except FileNotFoundError:
            continue
        except Exception as error:
            logging.info(
                f"{metapath}: Path count file read error - {error}"
            )
            # Catch all other file issues
            os.remove(graph.get_path_counts_path(mp, 'dwpc', damping, 'sparse.npz'))
    row, col, dwpc_matrix = hetmech.degree_weight.dwpc(graph, metapath, damping=damping,
                                                       dense_threshold=dense_threshold, dtype=dtype)
    path = graph.get_path_counts_path(metapath, 'dwpc', damping, None)
    hetmech.hetmat.save_matrix(dwpc_matrix, path)
    return row, col, dwpc_matrix


def compute_save_dgp(hetmat, metapath, damping=0.5, compression='gzip', delete_intermediates=True):
    """
    Compute summary file of combined degree-grouped permutations (DGP). Aggregates across permutations,
    deleting intermediates if delete_intermediates=True. Saves resulting files as compressed .tsv files
    using compression method given by compression.
    """
    for mp in (metapath.inverse, metapath):
        combined_path = hetmat.directory.joinpath(
          'adjusted-path-counts', 'dwpc-0.5', 'degree-grouped-permutations', f'{mp}.tsv')
        if combined_path.exists():
            return

    _, _, matrix = hetmat.read_path_counts(metapath, 'dwpc', damping)
    matrix_mean = matrix.mean()

    for name, permat in hetmat.permutations.items():
        path = permat.directory.joinpath('degree-grouped-path-counts', 'dwpc-0.5', f'{metapath}.tsv')
        if path.exists():
            pass
        else:
            degree_grouped_df = hetmech.degree_group.single_permutation_degree_group(
                permat, metapath, dwpc_mean=matrix_mean, damping=damping)
            path.parent.mkdir(parents=True, exist_ok=True)
            degree_grouped_df.to_csv(path, sep='\t')

    degree_stats_df = hetmech.degree_group.summarize_degree_grouped_permutations(
        hetmat, metapath, damping=damping, delete_intermediates=delete_intermediates)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    degree_stats_df.to_csv(combined_path, sep='\t', compression=compression)


def combine_dwpc_dgp(graph, metapath, damping):
    """
    Combine DWPC information with degree-grouped permutation summary metrics.
    Save resulting tables as one-per-metapath, compressed .tsv files.
    """
    stats_path = graph.directory.joinpath('adjusted-path-counts', f'dwpc-{float(damping)}',
                                          'degree-grouped-permutations', f'{metapath}.tsv')
    degree_stats_df = pandas.read_table(stats_path, compression='gzip')

    dwpc_row_generator = hetmech.degree_group.dwpc_to_degrees(graph, metapath)
    dwpc_df = pandas.DataFrame(dwpc_row_generator)
    df = (
        dwpc_df
        .merge(degree_stats_df, on=['source_degree', 'target_degree'])
        .drop(columns=['source_degree', 'target_degree'])
    )
    df['mean-nz'] = df['mean'] * df['n'] / df['nnz']
    df['beta'] = df['mean-nz'] / df['sd'] ** 2
    df['alpha'] = df['mean-nz'] * df['beta']
    df['p-value'] = df['nnz'] / df['n'] * scipy.special.gammaincc(df['alpha'], df['beta'] * df['dwpc'])
    return df

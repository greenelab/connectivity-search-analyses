import pathlib
import zipfile
import pandas

import hetmech.degree_group
import hetmech.degree_weight
import hetmech.matrix


def compute_save_dwpc(graph, metapath, damping=0.5, dense_threshold=1, dtype='float64', approx_ok=True):
    """
    Returns and saves a DWPC matrix. If already computed, will return saved result.
    """
    path = graph.get_path_counts_path(metapath, 'dwpc', damping, None)
    for inverse in (True, False):
        mp = metapath
        if inverse:
            mp = metapath.inverse
        for ext in ('.sparse.npz', '.npy'):
            path = pathlib.Path(str(path) + ext)
            if path.exists():
                try:
                    return graph.read_path_counts(mp, 'dwpc', damping)
                except zipfile.BadZipfile:
                    continue
    row, col, dwpc_matrix = hetmech.degree_weight.dwpc(graph, metapath, damping=damping,
                                                       dense_threshold=dense_threshold, dtype=dtype)
    path = graph.get_path_counts_path(metapath, 'dwpc', damping, None)
    hetmech.hetmat.save_matrix(dwpc_matrix, path)
    return row, col, dwpc_matrix


def combine_dwpc_dgp(graph, metapath, damping):
    """
    Combine DWPC information with degree-grouped permutation summary metrics.
    Save resulting tables as one-per-metapath, compressed .tsv files.
    """
    stats_path = graph.directory.joinpath('adjusted-path-counts', f'dwpc-{float(damping)}',
                                          'degree-grouped-permutations', f'{metapath}.tsv')
    degree_stats_df = pandas.read_table(stats_path)

    dwpc_row_generator = hetmech.degree_group.dwpc_to_degrees(graph, metapath)
    dwpc_df = pandas.DataFrame(dwpc_row_generator)
    df = (
        dwpc_df
        .merge(degree_stats_df, on=['source_degree', 'target_degree'])
        .drop(columns=['source_degree', 'target_degree'])
        .rename(columns={'mean': 'p-dwpc', 'sd': 'sd-dwpc'})
    )
    df['r-dwpc'] = df['dwpc'] - df['p-dwpc']
    df['z-dwpc'] = df['r-dwpc'] / df['sd-dwpc']
    return df

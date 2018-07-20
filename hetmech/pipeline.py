import pandas
import scipy.special
import scipy.stats

import hetmech.degree_group
import hetmech.degree_weight
import hetmech.hetmat


def combine_dwpc_dgp(graph, metapath, damping, ignore_zeros=False):
    """
    Combine DWPC information with degree-grouped permutation summary metrics.
    Includes gamma-hurdle significance estimates.
    """
    stats_path = graph.get_running_degree_group_path(metapath, 'dwpc', damping, extension='.tsv.gz')
    dgp_df = pandas.read_table(stats_path)
    dgp_df['mean_nz'] = dgp_df['sum'] / dgp_df['nnz']
    dgp_df['sd_nz'] = ((dgp_df['sum_of_squares'] - dgp_df['sum'] ** 2 / dgp_df['nnz']) / (dgp_df['nnz'] - 1)) ** 0.5
    dgp_df['beta'] = dgp_df['mean_nz'] / dgp_df['sd_nz'] ** 2
    dgp_df['alpha'] = dgp_df['mean_nz'] * dgp_df['beta']
    dwpc_row_generator = hetmech.degree_group.dwpc_to_degrees(
        graph, metapath, damping=damping, ignore_zeros=ignore_zeros)
    dwpc_df = pandas.DataFrame(dwpc_row_generator)
    dwpc_df = dwpc_df.merge(dgp_df)
    dwpc_df['p_value'] = (
        dwpc_df['nnz'] / dwpc_df['n']
        * (1 - scipy.special.gammainc(dwpc_df['alpha'], dwpc_df['beta'] * dwpc_df['dwpc']))
    ).where(cond=dwpc_df['dwpc'] > 0, other=1)
    dwpc_df.drop(columns=['sum', 'sum_of_squares', 'beta', 'alpha'], inplace=True)
    dwpc_df.sort_values(['source_id', 'target_id'], inplace=True)
    return dwpc_df

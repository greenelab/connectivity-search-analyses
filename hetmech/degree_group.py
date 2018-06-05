import numpy
import scipy.sparse

from hetmech.matrix import metaedge_to_adjacency_matrix


def degrees_to_degree_to_ind(degrees):
    degree_to_indices = dict()
    for i, degree in sorted(enumerate(degrees), key=lambda x: x[1]):
        degree_to_indices.setdefault(degree, []).append(i)
    return degree_to_indices


def metapath_to_degree_dicts(graph, metapath):
    metapath = graph.metagraph.get_metapath(metapath)
    _, _, source_adj_mat = metaedge_to_adjacency_matrix(graph, metapath[0], dense_threshold=0.7)
    _, _, target_adj_mat = metaedge_to_adjacency_matrix(graph, metapath[-1], dense_threshold=0.7)
    source_degrees = source_adj_mat.sum(axis=1).flat
    target_degrees = target_adj_mat.sum(axis=0).flat
    source_degree_to_ind = degrees_to_degree_to_ind(source_degrees)
    target_degree_to_ind = degrees_to_degree_to_ind(target_degrees)
    return source_degree_to_ind, target_degree_to_ind


def generate_degree_group_stats(source_degree_to_ind, target_degree_to_ind, matrix, scale=False, scaler=1):
    """
    Yield dictionaries with degree grouped stats
    """
    if scipy.sparse.issparse(matrix) and not scipy.sparse.isspmatrix_csr(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)
    for source_degree, row_inds in source_degree_to_ind.items():
        if source_degree > 0:
            row_matrix = matrix[row_inds, :]
            if scipy.sparse.issparse(row_matrix):
                row_matrix = row_matrix.toarray()
                # row_matrix = scipy.sparse.csc_matrix(row_matrix)
        for target_degree, col_inds in target_degree_to_ind.items():
            row = {
                'source_degree': source_degree,
                'target_degree': target_degree,
            }
            row['n'] = len(row_inds) * len(col_inds)
            if source_degree == 0 or target_degree == 0:
                row['sum'] = 0
                row['nnz'] = 0
                row['sum_of_squares'] = 0
                yield row
                continue

            slice_matrix = row_matrix[:, col_inds]
            values = slice_matrix.data if scipy.sparse.issparse(slice_matrix) else slice_matrix
            if scale:
                values = numpy.arcsinh(values / scaler)
            row['sum'] = values.sum()
            row['sum_of_squares'] = (values ** 2).sum()
            if scipy.sparse.issparse(slice_matrix):
                row['nnz'] = slice_matrix.nnz
            else:
                row['nnz'] = numpy.count_nonzero(slice_matrix)
            yield row


# def compute(graph, metapaths, )
# rows = hetmech.degree_group.generate_degree_group_stats(source_deg_to_ind, target_deg_to_ind,
#                                                  dwpc_matrix, scale=True, scaler=dwpc_matrix.mean())
# degree_stats_df = (
#     pd.DataFrame(rows)
#     .set_index(['source_degree', 'target_degree'])
#     .assign(n_perms=1)
# )
# degree_stats_df.head(2)


def compute_summary_metrics(df):
    df['mean'] = df['sum'] / df['n']
    df['sd'] = ((df['sum_of_squares'] - df['sum'] ** 2 / df['n']) / (df['n'] - 1)) ** 0.5
    return df

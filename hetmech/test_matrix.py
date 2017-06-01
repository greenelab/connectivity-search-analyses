import hetio.readwrite
import numpy
import pytest
from scipy import sparse

from .matrix import metaedge_to_adjacency_matrix

gig_rows = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
gig_cols = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
gig_adj = numpy.array([[0, 0, 1, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0],
                       [1, 1, 0, 1, 0, 0, 1],
                       [0, 0, 1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0]], dtype=numpy.bool_)
gig_adj_csc = sparse.csc_matrix(gig_adj)
gad_rows = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
gad_cols = ["Crohn's Disease", 'Multiple Sclerosis']
gad_adj = numpy.array([[0, 1],
                       [0, 1],
                       [1, 0],
                       [0, 1],
                       [0, 0],
                       [1, 1],
                       [0, 0]], dtype=numpy.float64)
gad_adj_csc = sparse.csc_matrix(gad_adj, dtype=numpy.bool_)
dlt_rows = ["Crohn's Disease", 'Multiple Sclerosis']
dlt_cols = ['Leukocyte', 'Lung']
dlt_adj = numpy.array([[0, 0],
                       [1, 0]], dtype=numpy.bool_)
dlt_adj_csc = sparse.csc_matrix(dlt_adj)
tld_rows = ['Leukocyte', 'Lung']
tld_cols = ["Crohn's Disease", 'Multiple Sclerosis']
tld_adj = numpy.array([[0, 1],
                       [0, 0]])
tld_adj_csc = sparse.csc_matrix(tld_adj, dtype=numpy.int64)


@pytest.mark.parametrize("test_edge,exp_row,exp_col,exp_adj,mat_type,dtype", [
    ('GiG', gig_rows, gig_cols, gig_adj, numpy.array, numpy.bool_),
    ('GiG', gig_rows, gig_cols, gig_adj_csc, sparse.csc_matrix, numpy.int32),

    ('GaD', gad_rows, gad_cols, gad_adj, numpy.array, numpy.float64),
    ('GaD', gad_rows, gad_cols, gad_adj_csc, sparse.csc_matrix, numpy.bool_),

    ('DlT', dlt_rows, dlt_cols, dlt_adj, numpy.array, numpy.bool_),
    ('DlT', dlt_rows, dlt_cols, dlt_adj_csc, sparse.csc_matrix, numpy.int32),

    ('TlD', tld_rows, tld_cols, tld_adj, numpy.array, numpy.int32),
    ('TlD', tld_rows, tld_cols, tld_adj_csc, sparse.csc_matrix, numpy.int64)
])
def test_metaedge_to_adjacency_matrix(test_edge, exp_row, exp_col, exp_adj,
                                      mat_type, dtype):
    """
    Test the functionality of metaedge_to_adjacency_matrix in generating
    numpy arrays. Uses same test data as in test_degree_weight.py
    Figure 2D of Himmelstein & Baranzini (2015) PLOS Comp Bio.
    https://doi.org/10.1371/journal.pcbi.1004259.g002
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    row_names, col_names, adj_mat = \
        metaedge_to_adjacency_matrix(graph, test_edge, matrix_type=mat_type,
                                     dtype=dtype)

    assert numpy.array_equal(row_names, exp_row)
    assert numpy.array_equal(col_names, exp_col)
    assert not type((exp_adj != adj_mat)).max(exp_adj != adj_mat)
    assert adj_mat.dtype == dtype
    assert isinstance(adj_mat, type(exp_adj))

import hetio.readwrite
import numpy
import pytest
from scipy import sparse

from .matrix import metaedge_to_adjacency_matrix


def get_arrays(edge, mat_type, dtype):
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    adj_dict = {
        'GiG': [[0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0]],
        'GaD': [[0, 1],
                [0, 1],
                [1, 0],
                [0, 1],
                [0, 0],
                [1, 1],
                [0, 0]],
        'DlT': [[0, 0],
                [1, 0]],
        'TlD': [[0, 1],
                [0, 0]]
    }
    row_names = node_dict[edge[0]]
    col_names = node_dict[edge[-1]]
    adj_matrix = mat_type(adj_dict[edge], dtype=dtype)
    return row_names, col_names, adj_matrix


@pytest.mark.parametrize("test_edge", ['GiG', 'GaD', 'DlT', 'TlD'])
@pytest.mark.parametrize("mat_type", [numpy.array, sparse.csc_matrix,
                                      sparse.csr_matrix, numpy.matrix])
@pytest.mark.parametrize("dtype", [numpy.bool_, numpy.int64, numpy.float64])
def test_metaedge_to_adjacency_matrix(test_edge, mat_type, dtype):
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
        metaedge_to_adjacency_matrix(graph, test_edge, dtype=dtype,
                                     matrix_type=mat_type)
    exp_row, exp_col, exp_adj = get_arrays(test_edge, mat_type, dtype)

    assert row_names == exp_row
    assert col_names == exp_col
    assert type(adj_mat) == type(exp_adj)
    assert adj_mat.dtype == dtype
    assert adj_mat.shape == exp_adj.shape
    assert (adj_mat != exp_adj).sum() == 0


@pytest.mark.parametrize('mat_type', [numpy.ndarray, sparse.csc_matrix])
@pytest.mark.parametrize("auto", [True, False])
@pytest.mark.parametrize("test_edge", ['GiG', 'GaD', 'DlT', 'TlD'])
def test_meta_auto(test_edge, mat_type, auto):
    """
    Test the functionality of metaedge_to_adjacency_matrix in generating
    arrays with automatic type. If the percent nonzero is above 30% of the
    matrix, then the matrix will be a numpy.ndarray. Otherwise, the matrix
    will be a sparse.csc_matrix.
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    row, col, adj = metaedge_to_adjacency_matrix(graph, test_edge, auto=auto,
                                                 matrix_type=mat_type)

    # Define a dict with adjacency matrices having known percent nonzero
    edge_to_nnz = {'GiG': sparse.csc_matrix,
                   'GaD': numpy.ndarray,
                   'DlT': sparse.csc_matrix,
                   'TlD': sparse.csc_matrix}
    auto_to_mat_type = {True: edge_to_nnz[test_edge],
                        False: mat_type}

    assert type(adj) == auto_to_mat_type[auto]

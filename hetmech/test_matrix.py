import hetio.readwrite
import numpy
import pytest
from scipy import sparse

from .matrix import metaedge_to_adjacency_matrix


def get_arrays(edge, mat_type, dtype, threshold):
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    adj_dict = {
        'GiG': ([[0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0]], 0.204),
        'GaD': ([[0, 1],
                [0, 1],
                [1, 0],
                [0, 1],
                [0, 0],
                [1, 1],
                [0, 0]], 0.429),
        'DlT': ([[0, 0],
                [1, 0]], 0.25),
        'TlD': ([[0, 1],
                [0, 0]], 0.25)
    }
    row_names = node_dict[edge[0]]
    col_names = node_dict[edge[-1]]

    if threshold:
        if adj_dict[edge][1] <= threshold:
            adj_matrix = sparse.csc_matrix(adj_dict[edge][0], dtype=dtype)
        else:
            adj_matrix = numpy.array(adj_dict[edge][0], dtype=dtype)
    elif mat_type:
        adj_matrix = mat_type(adj_dict[edge][0], dtype=dtype)
    else:
        adj_matrix = numpy.array(adj_dict[edge][0], dtype=dtype)
    return row_names, col_names, adj_matrix


@pytest.mark.parametrize('threshold', [0, 0.5, 1])
@pytest.mark.parametrize("test_edge", ['GiG', 'GaD', 'DlT', 'TlD'])
@pytest.mark.parametrize("mat_type", [numpy.array, sparse.csc_matrix,
                                      sparse.csr_matrix, numpy.matrix, None])
@pytest.mark.parametrize("dtype", [numpy.bool_, numpy.int64, numpy.float64])
def test_metaedge_to_adjacency_matrix(test_edge, mat_type, dtype, threshold):
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
    row_names, col_names, adj_mat = metaedge_to_adjacency_matrix(
            graph, test_edge, dtype=dtype, matrix_type=mat_type,
            sparse_threshold=threshold
        )
    exp_row, exp_col, exp_adj = get_arrays(test_edge, mat_type, dtype,
                                           threshold)

    assert row_names == exp_row
    assert col_names == exp_col
    assert type(adj_mat) == type(exp_adj)
    assert adj_mat.dtype == dtype
    assert adj_mat.shape == exp_adj.shape
    assert not (adj_mat != exp_adj).sum()


@pytest.mark.parametrize('mat_type', [numpy.ndarray, sparse.csc_matrix, None])
@pytest.mark.parametrize("threshold", [1, 0])
@pytest.mark.parametrize("test_edge", ['GiG', 'GaD', 'DlT', 'TlD'])
def test_meta_auto(test_edge, mat_type, threshold):
    """
    Test the functionality of metaedge_to_adjacency_matrix in generating
    arrays with automatic type. If the percent nonzero is above threshold,
    then the matrix will be a numpy.ndarray. Otherwise, the matrix will
    be a sparse.csc_matrix.
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    row, col, adj = metaedge_to_adjacency_matrix(
        graph, test_edge, sparse_threshold=threshold, matrix_type=mat_type)

    auto_to_mat_type = {1: sparse.csc_matrix,
                        0: mat_type if mat_type else numpy.ndarray}

    assert type(adj) == auto_to_mat_type[threshold]

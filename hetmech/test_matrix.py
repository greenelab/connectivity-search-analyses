import hetio.readwrite
from .matrix import metaedge_to_adjacency_matrix
import numpy as np


def test_metaedge_to_adjacency_matrix():
    """
    Test the functionality of metaedge_to_adjacency_matrix in
    generating sparse matrices vs numpy arrays. Uses same test
    data as in test_degree_weight.py Figure 2D of Himmelstein &
    Baranzini (2015) PLOS Comp Bio.
    https://doi.org/10.1371/journal.pcbi.1004259.g002
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)

    mat_gig = metaedge_to_adjacency_matrix(graph, 'GiG')
    assert np.array_equal(mat_gig[2], [[0, 0, 1, 0, 1, 0, 0],
                                       [0, 0, 1, 0, 0, 0, 0],
                                       [1, 1, 0, 1, 0, 0, 1],
                                       [0, 0, 1, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0, 0]])

    mat_gad = metaedge_to_adjacency_matrix(graph, 'GaD')
    assert np.array_equal(mat_gad[2], [[0, 1],
                                       [0, 1],
                                       [1, 0],
                                       [0, 1],
                                       [0, 0],
                                       [1, 1],
                                       [0, 0]])

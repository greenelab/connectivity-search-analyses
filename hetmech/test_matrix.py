import hetio.readwrite
from .matrix import metaedge_to_adjacency_matrix

'''
Test the functionality of metaedge_to_adjacency_matrix in
generating sparse matrices vs numpy arrays. Uses same test
data as in test_degree_weight.py Figure 2D of Himmelstein &
Baranzini (2015) PLOS Comp Bio.
https://doi.org/10.1371/journal.pcbi.1004259.g002
'''

url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
    '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
    'test/data/disease-gene-example-graph.json',
)
graph = hetio.readwrite.read_graph(url)


def test_sparse_matrix():
    mat = metaedge_to_adjacency_matrix(graph, 'GiG')
    assert all([mat[2][0, 4], mat[2][0, 2], mat[2][6, 2], mat[2][2, 0]])
    assert not any([mat[2][0, 0], mat[2][0, 1], mat[2][6, 0], mat[2][6, 1]])


def test_sparse_matrix_2():
    mat = metaedge_to_adjacency_matrix(graph, 'GaD')
    assert all([mat[2][0, 1], mat[2][1, 1], mat[2][2, 0], mat[2][3, 1]])
    assert not any([mat[2][0, 0], mat[2][1, 0], mat[2][2, 1], mat[2][3, 0]])

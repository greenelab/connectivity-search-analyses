import numpy

from .dwpc import remove_diag, degree_weight
from .matrix import metaedge_to_adjacency_matrix


def dwpc_baab(graph, metapath, damping=0.5):
    """
    A function to handle metapath (segments) of the form BAAB.
    This function will handle arbitrary lengths of this repeated
    pattern. For example, ABCCBA, ABCDDCBA, etc. all work with this
    function. As it stands, random single inserts are not supported.
    """
    assert len(metapath) % 2
    assert len(set(metapath)) % 2
    for i, metaedge in enumerate(metapath):
        assert metaedge.inverse == metapath[-1 - i]

    # Find the center of the metapath
    for i, v in enumerate(metapath.edges[:-2]):
        if v.inverse == metapath.edges[i + 2]:
            middle_edge, index = metapath[i + 1], i

    # Start with the center
    row, col, dwpc_matrix = metaedge_to_adjacency_matrix(
        graph, middle_edge, dtype=numpy.float64, sparse_threshold=0)
    dwpc_matrix = degree_weight(dwpc_matrix, damping, damping)

    # Move backwards from the center, multiplying around the existing matrix
    while index >= 0:
        r, c, adj = metaedge_to_adjacency_matrix(
            graph, metapath[index], dtype=numpy.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping, damping)
        dwpc_matrix = remove_diag(adj @ dwpc_matrix @ adj.T)
        index -= 1
    return row, col, dwpc_matrix

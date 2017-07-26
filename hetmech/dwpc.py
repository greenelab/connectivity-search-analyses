import functools
import operator

import numpy

from .matrix import metaedge_to_adjacency_matrix, normalize, copy_array


def remove_diag(mat):
    """Set the main diagonal of a square matrix to zeros."""
    assert len(set(mat.shape)) == 1  # must be square
    return mat - numpy.diag(mat.diagonal())


def degree_weight(matrix, damping, copy=True):
    """Normalize an adjacency matrix by the in and out degree."""
    matrix = copy_array(matrix, copy)
    row_sums = numpy.array(matrix.sum(axis=1)).flatten()
    column_sums = numpy.array(matrix.sum(axis=0)).flatten()
    matrix = normalize(matrix, row_sums, 'rows', damping)
    matrix = normalize(matrix, column_sums, 'columns', damping)

    return matrix


def dwpc_no_repeats(graph, metapath, damping=0.5):
    assert len(set(metapath.edges)) == len(metapath)

    parts = list()
    for edge in metapath:
        rows, cols, adj = metaedge_to_adjacency_matrix(
            graph, edge, dtype=numpy.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping)

        if edge == metapath[0]:
            row_names = rows
        if edge == metapath[-1]:
            col_names = cols
        parts.append(adj)

    dwpc_matrix = functools.reduce(operator.matmul, parts)
    return row_names, col_names, dwpc_matrix


def dwpc_baab(graph, metapath, damping=0.5):
    """
    A function to handle metapath (segments) of the form BAAB.
    This function will handle arbitrary lengths of this repeated
    pattern. For example, ABCCBA, ABCDDCBA, etc. all work with this
    function.
    """
    raise NotImplementedError("See PR #61")


def dwpc_baba(graph, metapath, damping=0.5):
    """
    Computes the degree-weighted path count for overlapping metanode
    repeats of the form B-A-B-A. Note that this does NOT yet support
    B-A-C-B-A, or any sort of metapath wherein there are metanodes
    within the overlapping region. This is the biggest priority to add.
    """
    raise NotImplementedError("See PR #61")


def dwpc_short_repeat(graph, metapath, damping=0.5):
    """
    One metanode repeated 3 or fewer times (A-A-A), not (A-A-A-A)
    This can include other random inserts, so long as they are not
    repeats. Must start with the repeated node. Acceptable examples:
    A-B-A-A) (A-B-A-C-D-E-F-A) (A-B-A-A-C), etc.
    """
    start_node = metapath.get_nodes()[0]
    dwpc_matrix = None
    for edge in metapath:
        rows, cols, adj = metaedge_to_adjacency_matrix(
            graph, edge, dtype=numpy.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping)
        if dwpc_matrix is None:
            row_names = rows
            dwpc_matrix = adj
        else:
            dwpc_matrix = dwpc_matrix@adj
        if edge.target == start_node:
            dwpc_matrix = remove_diag(dwpc_matrix)

    return row_names, cols, dwpc_matrix


def dwpc_long_repeat(graph, metapath, damping=0.5):
    """One metanode repeated 4 or more times. Considerably slower than
    dwpc_short_repeat, so should only be used if necessary. This
    function uses history vectors that split the computation into more
    tasks."""
    raise NotImplementedError("See PR #59")


def get_segments(metagraph, metapath):
    """Should categorize things into more than just the five categories
    in PR # 60. We want to segment the metapath into long-repeats, short-
    repeats, BABA, (which can not at the moment include other intermediates),
    BAAB (which can have intermediates as long as the whole thing is
    symmetrical), and other, non-segment-able regions."""
    raise NotImplementedError("Will integrate PR #60")


def dwpc(graph, metapath, damping=0.5):
    """This function will call get_segments, then the appropriate function"""
    raise NotImplementedError

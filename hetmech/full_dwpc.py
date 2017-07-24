import functools
import operator

import numpy

from .matrix import metaedge_to_adjacency_matrix, normalize, copy_array


def remove_diag(mat):
    """Set the main diagonal of a square matrix to zeros."""
    return mat - np.diag(mat.diagonal())


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
        adj = degree_weight(adj, 0.5)
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
    assert len(metapath) % 2
    assert len(set(metapath)) % 2
    for i, metaedge in enumerate(metapath):
        assert metaedge.inverse == metapath[-1-i]

    # Find the center of the metapath
    for i, v in enumerate(metapath.edges[:-2]):
        if v.inverse == metapath.edges[i+2]:
            middle_edge, index = metapath[i+1], i

    # Start with the center
    row, col, dwpc_matrix = metaedge_to_adjacency_matrix(
        graph, middle_edge, dtype=np.float64, sparse_threshold=0)
    dwpc_matrix = degree_weight(dwpc_matrix, damping)

    # Move backwards from the center, multiplying around the existing matrix
    while index >= 0:
        r, c, adj = metaedge_to_adjacency_matrix(
            graph, metapath[index], dtype=np.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping)
        dwpc_matrix = remove_diag(adj@dwpc_matrix@adj.T)
        index -= 1
    return row, col, dwpc_matrix


def index_to_baba(index, adjacency):
    """
    Takes an index and adjacency matrix and outputs an array of the child
    nodes a distance away. The function finds the children for one source
    node. If a weighted adjacency matrix is provided, this function computes
    the degree-weighted path count (DWPC). An unweighted matrix returns
    simply the path count.
    Parameters
    ----------
    index : int
        row index of the source node
    adjacency : numpy.ndarray
        adjacency matrix corresponding to the metaedge
    Returns
    -------
    numpy.ndarray
        1 dimensional array corresponding to the (PC/DWPC) of target nodes
    """
    # Create a vector for the given index
    b0 = numpy.zeros(adjacency.shape[0])
    b0[index] = 1

    # Find the child nodes and add the previous node's index at the
    #  zero position, elongating the vector by one.
    a0 = [numpy.insert(i, 0, index) for i in numpy.diag(b0 @ adjacency) if
          i.any()]

    b1 = []
    for i in a0:
        # Keep the nonzero element's index for history purposes
        ind = (i[1:] != 0).tolist().index(1)
        # Find the next set of child nodes (ignore history element)
        child = i[1:] @ adjacency.T
        # Set the element corresponding to the previous node equal to zero
        child[int(i[0])] = 0
        # Split the list of child nodes and add a history position again
        nodes = [numpy.insert(i, 0, ind) for i in numpy.diag(child) if i.any()]
        b1 += nodes
    # Roughly repeat the procedure
    a1 = numpy.zeros(adjacency.shape[1])
    for i in b1:
        child = i[1:] @ adjacency
        child[int(i[0])] = 0
        a1 += child
    return a1


def dwpc_baba(graph, metapath, damping=0.5):
    """
    Computes the degree-weighted path count for overlapping metanode
    repeats of the form B-A-B-A. Note that this does NOT yet support
    B-A-C-B-A, or any sort of metapath wherein there are metanodes
    within the overlapping region. This is the biggest priority to add.
    """
    metaedges = metapath.edges
    assert len(set(metaedges)) == 2

    # Get and normalize the adjacency matrix
    row, col, adjacency = metaedge_to_adjacency_matrix(
        graph, metaedges[0], dtype=numpy.float64, sparse_threshold=0)
    adjacency = degree_weight(adjacency, damping)

    ret = [index_to_baba(i, adjacency) for i in range(adjacency.shape[0])]
    ret = numpy.array(ret, dtype=numpy.float64)
    return row, col, ret


def dwpc_short_repeat(graph, metapath, damping=0.5):
    """One metanode repeated 3 or fewer times (A-A-A), not (A-A-A-A)"""
    rows, cols, dwpc_matrix = dwpc_no_repeats(graph, metapath, damping)
    dwpc_matrix = remove_diag(dwpc_matrix)
    return rows, cols, dwpc_matrix


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
    symmetrical), and non-segmentable."""
    raise NotImplementedError("Will integrate PR #60")


def dwpc(graph, metapath, damping=0.5):
    """This function will call get_segments, then the appropriate function"""
    raise NotImplementedError

import numpy

from .degree_weight import dwwc_step
from .matrix import metaedge_to_adjacency_matrix


def remove_diag(mat):
    """Set the main diagonal of a square matrix to zeros."""
    return mat - numpy.diag(mat.diagonal())


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
    within the overlapping region.
    """
    metaedges = metapath.edges
    assert len(set(metaedges)) == 2

    # Get and normalize the adjacency matrix
    row, col, adjacency = metaedge_to_adjacency_matrix(
        graph, metaedges[0], dtype=numpy.float64, sparse_threshold=0)
    adjacency = dwwc_step(adjacency, damping, damping)

    ret = [index_to_baba(i, adjacency) for i in range(adjacency.shape[0])]
    ret = numpy.array(ret, dtype=numpy.float64)
    return row, col, ret


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
        assert metaedge.inverse == metapath[-1 - i]

    # Find the center of the metapath
    for i, v in enumerate(metapath.edges[:-2]):
        if v.inverse == metapath.edges[i + 2]:
            middle_edge, index = metapath[i + 1], i

    # Start with the center
    row, col, dwpc_matrix = metaedge_to_adjacency_matrix(
        graph, middle_edge, dtype=numpy.float64, sparse_threshold=0)
    dwpc_matrix = dwwc_step(dwpc_matrix, damping, damping)

    # Move backwards from the center, multiplying around the existing matrix
    while index >= 0:
        r, c, adj = metaedge_to_adjacency_matrix(
            graph, metapath[index], dtype=numpy.float64, sparse_threshold=0)
        adj = dwwc_step(adj, damping, damping)
        dwpc_matrix = remove_diag(adj @ dwpc_matrix @ adj.T)
        index -= 1
    return row, col, dwpc_matrix

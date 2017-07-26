import functools

import numpy

from .degree_weight import dwwc_step
from .matrix import metaedge_to_adjacency_matrix


def remove_diag(mat):
    """Set the main diagonal of a square matrix to zeros."""
    return mat - numpy.diag(mat.diagonal())


def index_to_baba(graph, metapath, index, damping=0.5):
    """
    Takes an index and outputs an array (vector) of the child nodes
    along the metapath. Metapath must start with the first repeated
    node.

    Useful for metapaths of the general form (B-A-B-A). Can include
    any number of random metanode inserts, so long as the only repeats
    are of the above form.


    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    index : int
    damping : float

    Returns
    -------
    numpy.ndarray
        The target nodes down the metapath for the source node indicated
        by the given index
    """
    metanodes = list(metapath.get_nodes())
    repeated_nodes = {v for i, v in enumerate(metanodes) if
                      v in metanodes[i + 1:]}

    node_b = metapath[0].source  # B_0 in BABA
    other_repeats = (repeated_nodes - {node_b})

    assert node_b in repeated_nodes  # starts with a repeated
    assert len(other_repeats) == 1  # two repeats total

    node_a = other_repeats.pop()
    # create search vector corresponding to the given index
    node = numpy.zeros(
        len(metaedge_to_adjacency_matrix(graph, metapath[0])[0]) + 1)
    node[index + 1] = 1
    # set the history (for A_0) aside as NaN
    node[0] = numpy.nan
    vectors = [node]

    for edge in metapath:
        row, col, adj = metaedge_to_adjacency_matrix(
            graph, edge, dtype=numpy.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping)

        # multiply each vector by the adjacency matrix and transfer the history
        vectors = [numpy.insert(vec[1:] @ adj, 0, vec[0]) for vec in vectors]
        # split each child vector into vectors for each individual child node
        vectors = [numpy.insert(i, 0, vec[0]) for vec in vectors for i in
                   numpy.diag(vec[1:]) if i.any()]

        if edge.target == node_b:
            for vec in vectors:
                vec[1:][index] = 0
            # remove empty vectors
            vectors = [vec for vec in vectors if vec[1:].any()]
        elif edge.target == node_a:
            # initialize history by accounting for node A_0 in BABA
            if any([numpy.isnan(vec[0]) for vec in vectors]):
                for vec in vectors:
                    # the index of the nonzero element
                    for i, v in enumerate(vec[1:]):
                        if v:
                            vec[0] = i
                # remove empty vectors
                vectors = [vec for vec in vectors if vec[1:].any()]
            # enforce history for node A_1 in BABA
            else:
                for vec in vectors:
                    vec[1:][int(vec[0])] = 0
    targets = sum([vec[1:] for vec in vectors])
    if type(targets) != numpy.ndarray:
        targets = numpy.zeros(adj.shape[1], dtype=numpy.float64)
    return targets


def dwpc_baba(graph, metapath, damping=0.5):
    """
    Computes the degree-weighted path count for overlapping metanode
    repeats of the form B-A-B-A. Supports random inserts of metanodes,
    for example (B-C-A-D-B-A), so long as these inserts are not repeated.
    """
    row_names, col, adj = metaedge_to_adjacency_matrix(graph, metapath[0])
    number_source_nodes = adj.shape[0]
    row, col_names, adj = metaedge_to_adjacency_matrix(graph, metapath[-1])
    baba = functools.partial(index_to_baba, graph=graph, metapath=metapath,
                             damping=damping)
    dwpc_matrix = [baba(index=i) for i in range(number_source_nodes)]
    dwpc_matrix = numpy.array(dwpc_matrix, dtype=numpy.float64)
    return row_names, col_names, dwpc_matrix


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
    dwpc_matrix = dwwc_step(dwpc_matrix, damping, damping)

    # Move backwards from the center, multiplying around the existing matrix
    while index >= 0:
        r, c, adj = metaedge_to_adjacency_matrix(
            graph, metapath[index], dtype=numpy.float64, sparse_threshold=0)
        adj = dwwc_step(adj, damping, damping)
        dwpc_matrix = remove_diag(adj @ dwpc_matrix @ adj.T)
        index -= 1
    return row, col, dwpc_matrix

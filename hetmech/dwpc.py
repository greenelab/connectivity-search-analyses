import functools
import operator

import numpy

from .matrix import metaedge_to_adjacency_matrix, normalize, copy_array


def remove_diag(mat):
    """Set the main diagonal of a square matrix to zeros."""
    assert mat.shape[0] == mat.shape[1]  # must be square
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
    for metaedge in metapath:
        rows, cols, adj = metaedge_to_adjacency_matrix(
            graph, metaedge, dtype=numpy.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping)

        if metaedge == metapath[0]:
            row_names = rows
        if metaedge == metapath[-1]:
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
        vectors = [numpy.insert(vec[1:] @ adj, 0, vec[0]) for vec in
                   vectors]
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


def dwpc_short_repeat(graph, metapath, damping=0.5):
    """
    One metanode repeated 3 or fewer times (A-A-A), not (A-A-A-A)
    This can include other random inserts, so long as they are not
    repeats. Must start and end with the repeated node. Acceptable
    examples: (A-B-A-A), (A-B-A-C-D-E-F-A), (A-B-A-A), etc.
    """
    start_metanode = metapath.source()
    assert start_metanode == metapath.target()

    dwpc_matrix = None
    dwpc_tail = None
    index_of_repeats = [i for i, v in enumerate(metapath.get_nodes()) if
                        v == start_metanode]

    for metaedge in metapath[:index_of_repeats[1]]:
        row, col, adj = metaedge_to_adjacency_matrix(
            graph, metaedge, dtype=numpy.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping)
        if dwpc_matrix is None:
            row_names = col_names = row
            dwpc_matrix = adj
        else:
            dwpc_matrix = dwpc_matrix @ adj

    dwpc_matrix = remove_diag(dwpc_matrix)

    if len(index_of_repeats) == 3:
        for metaedge in metapath[index_of_repeats[1]:]:
            row, col, adj = metaedge_to_adjacency_matrix(
                graph, metaedge, dtype=numpy.float64, sparse_threshold=0)
            adj = degree_weight(adj, damping)
            if dwpc_tail is None:
                dwpc_tail = adj
            else:
                dwpc_tail = dwpc_tail @ adj
        dwpc_tail = remove_diag(dwpc_tail)
        dwpc_matrix = dwpc_matrix @ dwpc_tail
        dwpc_matrix = remove_diag(dwpc_matrix)

    return row_names, col_names, dwpc_matrix


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

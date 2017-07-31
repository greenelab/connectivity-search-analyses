import collections
import functools
import itertools
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
    function. Random non-repeat inserts are supported. The metapath
    must start and end with a repeated node, though.

    Covers all variants of symmetrically repeated metanodes with
    support for random non-repeat metanode inserts at any point.
    Metapath must start and end with a repeated metanode.


    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    damping : float

    Examples
    --------
    Acceptable metapaths forms include the following:
    B-A-A-B
    B-C-A-A-B
    B-C-A-D-A-E-B
    B-C-D-E-A-F-A-B
    """
    # Segment the metapath
    seg = get_segments(graph.metagraph, metapath)
    # Start with the middle group (A-A or A-...-A in BAAB)
    mid_ind = len(seg) // 2
    mid_seg = seg[mid_ind]
    row, col, dwpc_mid = dwpc_no_repeats(graph, mid_seg, damping=damping)
    dwpc_mid = remove_diag(dwpc_mid)

    # Get two indices for the segments ahead of and behind the middle region
    head_ind = mid_ind
    tail_ind = mid_ind
    while head_ind > 0:
        head_ind -= 1
        tail_ind += 1
        head = seg[head_ind]
        tail = seg[tail_ind]
        row, c, dwpc_head = dwpc_no_repeats(graph, head, damping=damping)
        r, col, dwpc_tail = dwpc_no_repeats(graph, tail, damping=damping)
        dwpc_mid = remove_diag(dwpc_head @ dwpc_mid @ dwpc_tail)
    return row, col, dwpc_mid


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


def categorize(metapath):
    """
    Returns the classification of a given metapath as one of
    a set of metapath types which we approach differently.
    Parameters
    ----------
    metapath : hetio.hetnet.MetaPath

    Returns
    -------
    classification : string
        One of ['no_repeats', 'disjoint', 'short_repeat',
                'long_repeat', 'BAAB', 'BABA', 'other']
    Examples
    --------
    GbCtDlA -> 'no_repeats'
    GiGiG   -> 'short_repeat'
    GiGiGcG -> 'long_repeat'
    GiGcGiG -> 'long_repeat'
    GiGbCrC -> 'disjoint'
    GbCbGbC -> 'BABA'
    GbCrCbG -> 'BAAB'
    DaGiGbCrC -> 'disjoint'
    GiGaDpCrC -> 'disjoint'
    GiGbCrCpDrD -> 'disjoint'
    GbCpDaGbCpD -> NotImplementedError
    GbCrCrCrCrCbG -> NotImplementedError
    """
    metanodes = list(metapath.get_nodes())
    repeated_nodes = {v for i, v in enumerate(metanodes) if
                      v in metanodes[i + 1:]}

    if not repeated_nodes:
        return 'no_repeats'

    repeats_only = [node for node in metanodes if node in repeated_nodes]

    # Group neighbors if they are the same
    grouped = [list(v) for k, v in itertools.groupby(repeats_only)]

    # Handle multiple disjoint repeats, any number, ie. AA,BB,CC,DD,...
    if len(grouped) == len(repeated_nodes):
        # Identify if there is only one metanode
        if len(set(repeated_nodes)) == 1:
            freq = collections.Counter(metanodes)
            if max(freq.values()) < 4:
                return 'short_repeat'
            else:
                return 'long_repeat'

        return 'disjoint'

    # Group [A, BB, A] or [A, B, A, B] into one
    if len(repeats_only) - len(grouped) <= 1:
        grouped = [repeats_only]

    # Categorize the reformatted metapath
    if len(grouped) == 1 and len(grouped[0]) == 4:
        if grouped[0][0] == grouped[0][-1]:
            return 'BAAB'
        else:
            return 'BABA'
    else:
        # Multi-repeats that aren't disjoint, eg. ABCBAC
        if len(repeated_nodes) > 2:
            raise NotImplementedError(
                "Only two overlapping repeats currently supported")

        if len(metanodes) > 5:
            raise NotImplementedError(
                "Complex metapaths of length > 4 are not yet supported")
        return 'other'


def get_segments(metagraph, metapath):
    """
    Split a metapath into segments of recognized groups and non-repeated
    nodes. Groups include BAAB, BABA, disjoint short- and long-repeats.
    Returns an error for categorization 'other'.

    Parameters
    ----------
    metagraph : hetio.hetnet.MetaGraph
    metapath : hetio.hetnet.Metapath

    Returns
    -------
    list
        list of metapaths. If the metapath is not segmentable or is already
        fully simplified (eg. GaDaGaD), then the list will have only one
        element.

    Examples
    --------
    'CbGaDaGaD' -> ['CbG', 'GaDaGaD']
    'GbCpDaGaD' -> ['GbCpDaGaD']
    'CrCbGiGaDrD' -> ['CrCbG', 'GiGaD', 'DrD']
    """
    def add_head_tail(metapath, indices):
        # handle non-duplicated on the front
        if indices[0][0] != 0:
            indices = [[0, indices[0][0]]] + indices
        # handle non-duplicated on the end
        if indices[-1][-1] != len(metapath):
            indices = indices + [[indices[-1][-1], len(metapath)]]
        return indices

    category = categorize(metapath)
    metanodes = metapath.get_nodes()
    freq = collections.Counter(metanodes)
    repeated_nodes = {i for i in freq.keys() if freq[i] > 1}

    if category == 'other':
        raise NotImplementedError("Incompatible metapath")

    elif category in ('disjoint', 'short_repeat', 'long_repeat'):
        indices = sorted([[metanodes.index(i), len(metapath) - list(
            reversed(metanodes)).index(i)] for i in repeated_nodes])
        indices = add_head_tail(metapath, indices)
        # handle middle cases with non-repeated nodes between disjoint regions
        # Eg. [[0,2], [3,4]] -> [[0,2],[2,3],[3,4]]
        inds = []
        for i, v in enumerate(indices[:-1]):
            inds.append(v)
            if v[-1] != indices[i + 1][0]:
                inds.append([v[-1], indices[i + 1][0]])
        indices = inds + [indices[-1]]

    elif category in ('BAAB', 'BABA'):
        assert len(repeated_nodes) == 2
        indices_of_repeats = [i for i, v in enumerate(metanodes)
                              if v in repeated_nodes]
        indices_of_next = indices_of_repeats[1:] + [len(metanodes) + 1]
        indices = [i for i in zip(indices_of_repeats, indices_of_next)]
        indices = add_head_tail(metapath, indices)

    segments = [metapath[i[0]:i[1]] for i in indices]
    segments = [i for i in segments if i]
    segments = [metagraph.get_metapath(metaedges) for metaedges in segments]
    return segments


def dwpc(graph, metapath, damping=0.5):
    """This function will call get_segments, then the appropriate function"""
    raise NotImplementedError

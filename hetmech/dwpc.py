import collections
import functools
import itertools
import logging
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
    metanodes = list(metapath.get_nodes())
    repeated_nodes = [v for i, v in enumerate(metanodes) if
                      v in metanodes[i + 1:]]
    # Find the indices of the innermost repeat (eg. BACAB -> 1,3)
    first_inner, second_inner = [i for i, metanode in enumerate(metanodes) if
                                 metanode == repeated_nodes[-1]]
    dwpc_inner = None

    # Traverse between and including innermost repeated metanodes
    inner_metapath = graph.metagraph.get_metapath(
        metapath[first_inner:second_inner])
    dwpc_inner = dwpc_short_repeat(graph, inner_metapath, damping=damping)[2]

    def next_outer(first_ind, last_ind, inner_array):
        """
        A recursive function. Works outward from the middle of a
        metapath. Multiplies non-repeat metanodes as appropriate and
        builds outward. When identical metanodes are ahead of and
        behind the middle segment being worked with, this function
        multiplies by both and subtracts the main diagonal.

        Parameters
        ----------
        first_ind : int
            index at the beginning of the middle segment
        last_ind : int
            index at the end of the middle segment
        inner_array : numpy.ndarray
            The working dwpc_matrix, which is multiplied from the front
            and back depending on which side has a duplicated metanode
            at the closest position
        """
        # case where node at the end is a repeated metanode
        if metanodes[last_ind + 1] in repeated_nodes:
            # if middle segment surrounded by repeated metanodes
            if metanodes[first_ind - 1] == metanodes[last_ind + 1]:
                adj1 = metaedge_to_adjacency_matrix(
                    graph, metapath[first_ind - 1])[2]
                adj2 = metaedge_to_adjacency_matrix(
                    graph, metapath[last_ind])[2]
                adj1 = degree_weight(adj1, damping)
                adj2 = degree_weight(adj2, damping)

                inner_array = adj1 @ (inner_array @ adj2)
                inner_array = remove_diag(inner_array)
                first_ind, last_ind = first_ind - 1, last_ind + 1
            # only trailing metanode is a repeat
            else:
                adj = metaedge_to_adjacency_matrix(
                    graph, metapath[first_ind - 1])[2]
                adj = degree_weight(adj, damping)
                inner_array = adj @ inner_array
                first_ind -= 1
        # trailing metanode is not a repeated
        else:
            adj = metaedge_to_adjacency_matrix(graph, metapath[last_ind])[2]
            adj = degree_weight(adj, damping)
            inner_array = inner_array @ adj
            last_ind += 1
        # the middle segment spans the entire metapath
        if len(metapath) == last_ind - first_ind:
            return inner_array
        else:
            return next_outer(first_ind, last_ind, inner_array)

    # get source and target ID arrays
    row_names = metaedge_to_adjacency_matrix(
        graph, metapath[0], dtype=numpy.float64)[0]
    col_names = metaedge_to_adjacency_matrix(
        graph, metapath[-1], dtype=numpy.float64)[1]
    dwpc_matrix = next_outer(first_inner, second_inner, dwpc_inner)
    return row_names, col_names, dwpc_matrix


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
    GbCpDaGbCpD -> 'other'
    GbCrCrCrCrCbG -> 'other'
    """
    metanodes = list(metapath.get_nodes())
    freq = collections.Counter(metanodes)
    repeated = {metanode for metanode, count in freq.items() if count > 1}

    if not repeated:
        return 'no_repeats'

    repeats_only = [node for node in metanodes if node in repeated]

    # Group neighbors if they are the same
    grouped = [list(v) for k, v in itertools.groupby(repeats_only)]

    # Handle multiple disjoint repeats, any number, ie. AA,BB,CC,DD,...
    if len(grouped) == len(repeated):
        # Identify if there is only one metanode
        if len(repeated) == 1:
            if max(freq.values()) < 4:
                return 'short_repeat'
            else:
                return 'long_repeat'

        return 'disjoint'

    assert len(repeats_only) > 3

    # Categorize the reformatted metapath
    if len(repeats_only) == 4:
        if repeats_only[0] == repeats_only[-1]:
            assert repeats_only[1] == repeats_only[2]
            return 'BAAB'
        else:
            assert repeats_only[0] == repeats_only[2] and \
                   repeats_only[1] == repeats_only[3]
            return 'BABA'
    elif len(repeats_only) == 5 and max(map(len, grouped)) == 3:
        if repeats_only[0] == repeats_only[-1]:
            return 'BAAB'
    else:
        # Multi-repeats that aren't disjoint, eg. ABCBAC
        if len(repeated) > 2:
            logging.info(
                f"{metapath}: Only two overlapping repeats currently supported"
            )

        if len(metanodes) > 4:
            logging.info(
                f"{metapath}: Complex metapaths of length > 4 are not yet "
                f"supported")
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
    category_to_function = {'no_repeats': dwpc_no_repeats,
                            'short_repeat': dwpc_short_repeat,
                            'long_repeat': dwpc_long_repeat,
                            'BAAB': dwpc_baab,
                            'BABA': dwpc_baba}

    category = categorize(metapath)
    if category in ('long_repeat', 'other', 'BABA'):
        raise NotImplementedError

    segments = get_segments(graph.metagraph, metapath)

    row_names = None

    dwpc_matrices = []
    for subpath in segments:
        print(subpath)
        subcat = categorize(subpath)
        row, col, mat = category_to_function[subcat](graph, subpath, damping)
        dwpc_matrices.append(mat)
        if row_names is None:
            row_names = row

    col_names = col
    dwpc_matrix = functools.reduce(operator.matmul, dwpc_matrices)

    return row_names, col_names, dwpc_matrix

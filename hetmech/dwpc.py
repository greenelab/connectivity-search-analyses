import collections
import copy
import functools
import itertools
import logging
import operator

import numpy

from .matrix import copy_array, metaedge_to_adjacency_matrix, normalize


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
    repeats of the form B-A-B-A. Supports random inserts.
    Segment must start with B and end with A. AXBYAZB
    """
    seg = get_segments(graph.metagraph, metapath)

    row_names, col, axb = dwpc_no_repeats(graph, seg[0], damping=damping)
    row, col, bya = dwpc_no_repeats(graph, seg[1], damping=damping)
    row, col_names, azb = dwpc_no_repeats(graph, seg[2], damping=damping)

    correction_a = numpy.diag((axb@bya).diagonal())@azb
    correction_b = axb@numpy.diag((bya@azb).diagonal())
    correction_c = axb*bya.T*azb

    dwpc_matrix = (axb@bya@azb - correction_a - correction_b
                   + correction_c)

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


def node_to_children(graph, metapath, node, metapath_index, damping=0,
                     history=None):
    """
    Returns a history adjusted list of child nodes.

    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    node : numpy.ndarray
    metapath_index : int
    damping : float
    history : numpy.ndarray

    Returns
    -------
    dict
        List of child nodes and a single numpy.ndarray of the newly
        updated history vector.
    """
    metaedge = metapath[metapath_index]
    metanodes = list(metapath.get_nodes())
    freq = collections.Counter(metanodes)
    repeated = {i for i in freq.keys() if freq[i] > 1}

    if history is None:
        history = {
            i.target: numpy.ones(
                len(metaedge_to_adjacency_matrix(graph, i)[1]
                    ), dtype=numpy.float64)
            for i in metapath if i.target in repeated
        }
    history = history.copy()
    if metaedge.source in history:
        history[metaedge.source] -= numpy.array(node != 0, dtype=numpy.float64)

    row, col, adj = metaedge_to_adjacency_matrix(graph, metaedge,
                                                 dtype=numpy.float64)
    adj = degree_weight(adj, damping)
    vector = node @ adj

    if metaedge.target in history:
        vector *= history[metaedge.target]

    children = [i for i in numpy.diag(vector) if i.any()]
    return {'children': children, 'history': history,
            'next_index': metapath_index + 1}


def dwpc_general_case(graph, metapath, damping=0):
    """
    A slow but general function to compute the degree-weighted
    path count. Works by splitting the metapath at junctions
    where one node is joined to multiple nodes over a metaedge.

    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    damping : float
    """
    dwpc_step = functools.partial(node_to_children, graph=graph,
                                  metapath=metapath, damping=damping)

    start_nodes, col, adj = metaedge_to_adjacency_matrix(graph, metapath[0])
    row, fin_nodes, adj = metaedge_to_adjacency_matrix(graph, metapath[-1])
    number_start = len(start_nodes)
    number_end = len(fin_nodes)

    dwpc_matrix = []
    if len(metapath) > 1:
        for i in range(number_start):
            search = numpy.zeros(number_start, dtype=numpy.float64)
            search[i] = 1
            step1 = [dwpc_step(node=search, metapath_index=0, history=None)]
            k = 1
            while k < len(metapath):
                k += 1
                step2 = []
                for group in step1:
                    for child in group['children']:
                        hist = copy.deepcopy(group['history'])
                        out = dwpc_step(node=child,
                                        metapath_index=group['next_index'],
                                        history=hist)
                        if out['children']:
                            step2.append(out)
                    step1 = step2

            final_children = [group for group in step2
                              if group['children'] != []]

            end_nodes = sum(
                [child for group in final_children
                 for child in group['children']])
            if type(end_nodes) not in (list, numpy.ndarray):
                end_nodes = numpy.zeros(number_end)
            dwpc_matrix.append(end_nodes)
    else:
        dwpc_matrix = degree_weight(adj, damping=damping)
    dwpc_matrix = numpy.array(dwpc_matrix, dtype=numpy.float64)
    return start_nodes, fin_nodes, dwpc_matrix


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
    elif repeats_only == list(reversed(repeats_only)) and \
            not len(repeats_only) % 2:
        return 'BAAB'

    else:
        # Multi-repeats that aren't disjoint, eg. ABCBAC
        if len(repeated) > 2:
            logging.info(
                f"{metapath}: Only two overlapping repeats currently supported"
            )
            return 'other'

        if len(metanodes) > 4:
            logging.info(
                f"{metapath}: Complex metapaths of length > 4 are not yet "
                f"supported")
            return 'other'
        assert False


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
    'CbGaDaGaD' -> ['CbG', 'GaD', 'GaG', 'GaD']
    'GbCpDaGaD' -> ['GbCpD', 'DaG', 'GaD']
    'CrCbGiGaDrD' -> ['CrC', 'CbG', 'GiG', 'GaD', 'DrD']
    """
    def add_head_tail(metapath, indices):
        # handle non-duplicated on the front
        if indices[0][0] != 0:
            indices = [(0, indices[0][0])] + indices
        # handle non-duplicated on the end
        if indices[-1][-1] != len(metapath):
            indices = indices + [(indices[-1][-1], len(metapath))]
        return indices

    category = categorize(metapath)
    metanodes = metapath.get_nodes()
    freq = collections.Counter(metanodes)
    repeated = {i for i in freq.keys() if freq[i] > 1}

    # if category == 'other':
    #     raise NotImplementedError("Incompatible metapath")

    if category in ('disjoint', 'short_repeat', 'long_repeat'):
        indices = sorted([[metanodes.index(i), len(metapath) - list(
            reversed(metanodes)).index(i)] for i in repeated])
        indices = add_head_tail(metapath, indices)
        # handle middle cases with non-repeated nodes between disjoint regions
        # Eg. [[0,2], [3,4]] -> [[0,2],[2,3],[3,4]]
        inds = []
        for i, v in enumerate(indices[:-1]):
            inds.append(v)
            if v[-1] != indices[i + 1][0]:
                inds.append([v[-1], indices[i + 1][0]])
        indices = inds + [indices[-1]]

    elif category in ('BAAB', 'BABA', 'other'):
        nodes = set(metanodes)
        repeat_indices = (
            [[i for i, v in enumerate(metanodes)
              if v == metanode] for metanode in nodes])
        repeat_indices = [i for i in repeat_indices if len(i) > 1]
        simple_repeats = [i for group in repeat_indices for i in group]
        inds = []
        for i in repeat_indices:
            if len(i) == 2:
                inds += i
            if len(i) > 2:
                inds.append(i[0])
                inds.append(i[-1])
                for j in i[1:-1]:
                    if (j - 1 in simple_repeats and j + 1 in simple_repeats) \
                            and not (j - 1 in i and j + 1 in i):
                        inds.append(j)
        inds = sorted(inds)
        seconds = inds[1:] + [inds[-1]]
        indices = list(zip(inds, seconds))
        indices = [i for i in indices if len(set(i)) == 2]
        indices = add_head_tail(metapath, indices)
    segments = [metapath[i[0]:i[1]] for i in indices]
    segments = [i for i in segments if i]
    segments = [metagraph.get_metapath(metaedges) for metaedges in segments]
    return segments


def dwpc(graph, metapath, damping=0.5):
    """This function will call get_segments, then the appropriate function"""
    category_to_function = {'no_repeats': dwpc_no_repeats,
                            'short_repeat': dwpc_short_repeat,
                            'long_repeat': dwpc_general_case,
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

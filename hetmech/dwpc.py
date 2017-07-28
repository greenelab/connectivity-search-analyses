import collections
import copy
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
            graph, metaedge, dtype=numpy.float64)
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
            graph, metaedge, dtype=numpy.float64)
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
                graph, metaedge, dtype=numpy.float64)
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
    metaedge = metapath[metapath_index]
    metanodes = list(metapath.get_nodes())
    freq = collections.Counter(metanodes)
    repeated_nodes = {i for i in freq.keys() if freq[i] > 1}

    if history is None:
        history = {
            i.target: numpy.ones(
                len(metaedge_to_adjacency_matrix(graph, i)[1]
                    ), dtype=numpy.float64)
            for i in metapath if i.target in repeated_nodes
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

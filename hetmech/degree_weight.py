import collections
import copy
import functools
import itertools
import logging
import operator
import time

import numpy
from scipy import sparse

from .matrix import auto_convert, copy_array, metaedge_to_adjacency_matrix, \
    normalize


def dwpc(graph, metapath, damping=0.5, sparse_threshold=0, use_general=False):
    """
    A unified function to compute the degree-weighted path count.
    This function will call get_segments, then the appropriate
    specialized (or generalized) DWPC function.

    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    damping : float
    sparse_threshold : float (0 <= sparse_threshold <= 1)
        sets the density threshold above which a sparse matrix will be
        converted to a dense automatically.
    use_general : bool
        if True, dwpc will call _dwpc_general_case and give a warning
        on metapaths which are categorized 'other' and 'long_repeat'.
        If False, an exception is raised when such a metapath is given.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray or
    scipy.sparse.csc_matrix, float
        row labels, column labels, the DWPC matrix, and the computation time
    """
    start_time = time.perf_counter()
    category_to_function = {'no_repeats': _dwpc_no_repeats,
                            'short_repeat': _dwpc_short_repeat,
                            'long_repeat': _dwpc_general_case,
                            'BAAB': _dwpc_baab,
                            'BABA': _dwpc_baba,
                            'complex': dwpc,
                            'interior_complete_group': _dwpc_baba}

    category = categorize(metapath)
    segments = get_segments(graph.metagraph, metapath)
    if category == 'disjoint':
        row = None
        col = None
        dwpc_matrix = None
        for segment in segments:
            r, c, seg_matrix = category_to_function[categorize(segment)](
                graph, segment, damping=damping,
                sparse_threshold=sparse_threshold)
            if row is None:
                row = r
            if segment is segments[-1]:
                col = c

            if dwpc_matrix is None:
                dwpc_matrix = seg_matrix
            else:
                dwpc_matrix = dwpc_matrix @ seg_matrix

    elif category == 'repeat_around':
        mid = dwpc(graph, segments[1], damping=damping)[2]
        row, c, adj0 = _dwpc_no_repeats(graph, segments[0], damping=damping)
        r, col, adj1 = _dwpc_no_repeats(graph, segments[-1], damping=damping)
        dwpc_matrix = _remove_diag(adj0 @ mid @ adj1)

    elif category == 'short_repeat' and len(segments) != 1:
        dwpc_matrix = None
        for i in segments:
            if categorize(i) == 'short_repeat':
                row_names, col, mat = _dwpc_short_repeat(
                    graph, i, damping, sparse_threshold=sparse_threshold)
            else:
                row_names, col, mat = _dwpc_no_repeats(
                    graph, i, damping, sparse_threshold=sparse_threshold)
            if dwpc_matrix is None:
                row = row_names
                dwpc_matrix = mat
            else:
                dwpc_matrix = dwpc_matrix @ mat

    elif category in ('long_repeat', 'other'):
        if use_general:
            row, col, dwpc_matrix = _dwpc_general_case(
                graph, metapath, damping)
        else:
            raise Exception('Metapath category will use _dwpc_general_case')

    else:
        row, col, dwpc_matrix = category_to_function[category](
            graph, metapath, damping, sparse_threshold=sparse_threshold)
    total_time = time.perf_counter() - start_time
    return row, col, dwpc_matrix, total_time


def dwwc(graph, metapath, damping=0.5, sparse_threshold=0):
    """
    Compute the degree-weighted walk count (DWWC) in which nodes can be
    repeated within a path.

    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    damping : float
    sparse_threshold : float (0 <= sparse_threshold <= 1)
        sets the density threshold above which a sparse matrix will be
        converted to a dense automatically.
    """
    dwwc_matrix = None
    row_names = None
    for metaedge in metapath:
        rows, cols, adj_mat = metaedge_to_adjacency_matrix(
            graph, metaedge, sparse_threshold=sparse_threshold)
        adj_mat = _degree_weight(adj_mat, damping)
        if dwwc_matrix is None:
            row_names = rows
            dwwc_matrix = adj_mat
        else:
            dwwc_matrix = dwwc_matrix @ adj_mat
            dwwc_matrix = auto_convert(dwwc_matrix, sparse_threshold)
    return row_names, cols, dwwc_matrix


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
                'long_repeat', 'BAAB', 'BABA', 'repeat_around',
                 'interior_complete_group', 'disjoint_groups', 'other']
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
    # 6 node paths with 3x2 repeats
    elif len(repeated) == 3 and len(metapath) == 5:
        if repeats_only[0] == repeats_only[-1]:
            return 'repeat_around'
        # AABCCB or AABCBC
        elif len(grouped[0]) == 2 or len(grouped[-1]) == 2:
            return 'disjoint_groups'
        # ABA CC B
        elif len(repeats_only) - len(grouped) == 1:
            return 'interior_complete_group'

        # most complicated len 6
        else:
            return 'other'

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

    if category == 'no_repeats':
        return [metapath]

    elif category == 'repeat_around':
        indices = [[0, 1], [1, 4], [4, 5]]

    elif category == 'disjoint_groups':
        # CCBABA or CCBAAB or BABACC or BAABCC -> [CC, BABA], etc.
        metanodes = list(metapath.get_nodes())
        grouped = [list(v) for k, v in itertools.groupby(metanodes)]
        indices = [[0, 1], [1, 2], [2, 5]] if len(grouped[0]) == 2 else [
            [0, 3], [3, 4], [4, 5]]

    elif category in ('disjoint', 'short_repeat', 'long_repeat'):
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

    elif category in ('BAAB', 'BABA', 'other', 'interior_complete_group'):
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
    # eg: B CC ABA
    if category == 'interior_complete_group':
        segs = []
        for i, v in enumerate(segments[:-1]):
            if segments[i + 1].source() == segments[i + 1].target():
                edges = v.edges + segments[i + 1].edges + segments[i + 2].edges
                segs.append(metagraph.get_metapath(edges))
            elif v.source() == v.target():
                pass
            elif segments[i - 1].source() == segments[i - 1].target():
                pass
            else:
                segs.append(v)
        segs.append(segments[-1])
        segments = segs
    return segments


def _remove_diag(mat):
    """Set the main diagonal of a square matrix to zeros."""
    mattype = type(mat)
    if mattype == numpy.ndarray:
        mattype = numpy.array
    assert mat.shape[0] == mat.shape[1]  # must be square
    return mat - mattype(numpy.diag(mat.diagonal()), dtype=numpy.float64)


def _degree_weight(matrix, damping, copy=True):
    """Normalize an adjacency matrix by the in and out degree."""
    matrix = copy_array(matrix, copy)
    row_sums = numpy.array(matrix.sum(axis=1)).flatten()
    column_sums = numpy.array(matrix.sum(axis=0)).flatten()
    matrix = normalize(matrix, row_sums, 'rows', damping)
    matrix = normalize(matrix, column_sums, 'columns', damping)

    return matrix


def _dwpc_no_repeats(graph, metapath, damping=0.5, sparse_threshold=0):
    assert len(set(metapath.edges)) == len(metapath)
    parts = []
    row_names = None
    for metaedge in metapath:
        rows, cols, adj = metaedge_to_adjacency_matrix(
            graph, metaedge, dtype=numpy.float64,
            sparse_threshold=sparse_threshold)
        adj = _degree_weight(adj, damping)

        if row_names is None:
            row_names = rows
        if metaedge == metapath[-1]:
            col_names = cols
        parts.append(adj)

    dwpc_matrix = functools.reduce(operator.matmul, parts)
    return row_names, col_names, dwpc_matrix


def _dwpc_baab(graph, metapath, damping=0.5, sparse_threshold=0):
    """
    A function to handle metapath (segments) of the form BAAB.
    This function will handle arbitrary lengths of this repeated
    pattern. For example, ABCCBA, ABCDDCBA, etc. all work with this
    function. Random non-repeat inserts are supported. The metapath
    must start and end with a repeated node, though.

    Covers all variants of symmetrically repeated metanodes with
    support for random non-repeat metanode inserts at any point.

    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    damping : float
    sparse_threshold : float (0 <= sparse_threshold <= 1)
        sets the density threshold above which a sparse matrix will be
        converted to a dense automatically.

    Examples
    --------
    Acceptable metapaths forms include the following:
    B-A-A-B
    B-C-A-A-B
    B-C-A-D-A-E-B
    B-C-D-E-A-F-A-B
    C-B-A-A-B-D-E
    """
    # Segment the metapath
    segments = get_segments(graph.metagraph, metapath)
    # Start with the middle group (A-A or A-...-A in BAAB)
    for i, s in enumerate(segments):
        if s.source() == s.target():
            mid_seg = s
            mid_ind = i
    row, col, dwpc_mid, seconds = dwpc(
        graph, mid_seg, damping=damping, sparse_threshold=sparse_threshold)
    dwpc_mid = _remove_diag(dwpc_mid)

    # Get two indices for the segments ahead of and behind the middle region
    head_ind = mid_ind
    tail_ind = mid_ind
    while head_ind > 0 or tail_ind < len(segments):
        head_ind -= 1
        tail_ind += 1
        head = segments[head_ind] if head_ind >= 0 else None
        tail = segments[tail_ind] if tail_ind < len(segments) else None
        # Multiply on the head
        if head is not None:
            row, c, dwpc_head, seconds = dwpc(
                graph, head, damping=damping,
                sparse_threshold=sparse_threshold)
            dwpc_mid = dwpc_head @ dwpc_mid
        # Multiply on the tail
        if tail is not None:
            r, col, dwpc_tail, seconds = dwpc(
                graph, tail, damping=damping,
                sparse_threshold=sparse_threshold)
            dwpc_mid = dwpc_mid @ dwpc_tail
        # Remove the diagonal if the head and tail are repeats
        if head and tail:
            if head.source() == tail.target():
                dwpc_mid = _remove_diag(dwpc_mid)

    return row, col, dwpc_mid


def _dwpc_baba(graph, metapath, damping=0.5, sparse_threshold=0):
    """
    Computes the degree-weighted path count for overlapping metanode
    repeats of the form B-A-B-A. Supports random inserts.
    Segment must start with B and end with A. AXBYAZB
    """
    segments = get_segments(graph.metagraph, metapath)
    seg_axb = None
    for i, s in enumerate(segments[:-2]):
        if s.source() == segments[i + 2].source() and not seg_axb:
            seg_axb = s
            seg_bya = segments[i + 1]
            seg_azb = segments[i + 2]
            seg_cda = segments[0] if i == 1 else None
            seg_bed = segments[-1] if segments[-1] != seg_azb else None
    # Collect segment DWPC and corrections
    row_names, col, axb, seconds = dwpc(graph, seg_axb, damping=damping,
                                        sparse_threshold=sparse_threshold)
    row, col, bya, seconds = dwpc(graph, seg_bya, damping=damping,
                                  sparse_threshold=sparse_threshold)
    row, col_names, azb, seconds = dwpc(graph, seg_azb, damping=damping,
                                        sparse_threshold=sparse_threshold)

    correction_a = numpy.diag((axb @ bya).diagonal()) @ azb
    correction_b = axb @ numpy.diag((bya @ azb).diagonal()) if \
        not sparse.issparse(bya) else \
        axb @ sparse.csc_matrix(numpy.diag((bya @ azb).diagonal()))
    correction_c = axb * bya.T * azb if not sparse.issparse(bya) else \
        (axb.multiply(bya.T)).multiply(azb)
    # Apply the corrections
    dwpc_matrix = (axb @ bya @ azb - correction_a - correction_b
                   + correction_c)
    # Account for possible head and tail segments outside the BABA group
    if seg_cda is not None:
        row_names, col, cda, seconds = dwpc(graph, seg_cda, damping=damping,
                                            sparse_threshold=sparse_threshold)
        dwpc_matrix = cda @ dwpc_matrix
    if seg_bed is not None:
        row, col_names, bed, seconds = dwpc(graph, seg_bed, damping=damping,
                                            sparse_threshold=sparse_threshold)
        dwpc_matrix = dwpc_matrix @ bed
    return row_names, col_names, dwpc_matrix


def _dwpc_short_repeat(graph, metapath, damping=0.5, sparse_threshold=0):
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
            graph, metaedge, dtype=numpy.float64,
            sparse_threshold=sparse_threshold)
        adj = _degree_weight(adj, damping)
        if dwpc_matrix is None:
            row_names = col_names = row
            dwpc_matrix = adj
        else:
            dwpc_matrix = dwpc_matrix @ adj

    dwpc_matrix = _remove_diag(dwpc_matrix)
    # Extra correction for random metanodes in the repeat segment
    if len(index_of_repeats) == 3:
        for metaedge in metapath[index_of_repeats[1]:]:
            row, col, adj = metaedge_to_adjacency_matrix(
                graph, metaedge, dtype=numpy.float64,
                sparse_threshold=sparse_threshold)
            adj = _degree_weight(adj, damping)
            if dwpc_tail is None:
                dwpc_tail = adj
            else:
                dwpc_tail = dwpc_tail @ adj
        dwpc_tail = _remove_diag(dwpc_tail)
        dwpc_matrix = dwpc_matrix @ dwpc_tail
        dwpc_matrix = _remove_diag(dwpc_matrix)
    return row_names, col_names, dwpc_matrix


def _node_to_children(graph, metapath, node, metapath_index, damping=0,
                      history=None):
    """
    Returns a history adjusted list of child nodes. Used in _dwpc_general_case.

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
    adj = _degree_weight(adj, damping)
    vector = node @ adj

    if metaedge.target in history:
        vector *= history[metaedge.target]

    children = [i for i in numpy.diag(vector) if i.any()]
    return {'children': children, 'history': history,
            'next_index': metapath_index + 1}


def _dwpc_general_case(graph, metapath, damping=0):
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
    dwpc_step = functools.partial(_node_to_children, graph=graph,
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
        dwpc_matrix = _degree_weight(adj, damping=damping)
    dwpc_matrix = numpy.array(dwpc_matrix, dtype=numpy.float64)
    return start_nodes, fin_nodes, dwpc_matrix

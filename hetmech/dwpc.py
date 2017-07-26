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


class TraverseLongRepeat:
    """
    Class is run by the following:

    var_name = TraverseLongRepeat(search_node, adjacency_matrix)
    var_name.go_to_depth(var_name.start, path_length)
    var_name.paths  # --> Gives vector of actual path targets
    """
    def __init__(self, start_node, adjacency_matrix):
        """
        Parameters
        ----------
        start_node : numpy.ndarray, dtype=numpy.float64
            vector of the start node
        adjacency_matrix : numpy.ndarray, dtype=numpy.float64
        """
        self.start = start_node
        self.adj = adjacency_matrix
        self.paths = numpy.zeros(len(start_node), dtype=numpy.float64)

    @staticmethod
    def node_to_children(node, adjacency, history=None):
        """
        Returns a dictionary of history-accounted child nodes and a history
        vector

        Parameters
        ----------
        node : numpy.ndarray
            vector with only the node to query
        adjacency : numpy.ndarray
            square adjacency matrix
        history : numpy.ndarray or None
            A vector with zero corresponding to a node which has already been
            traversed, and one corresponding to a yet-untouched node.

        Returns
        -------
        Dictionary of child nodes and history vector. Will not include any
        nodes which were given zero in the history vector.
        """
        if history is None:
            history = numpy.ones(len(node), dtype=numpy.float64)
        else:
            history = numpy.array(history != 0, dtype=numpy.float64)
        history -= numpy.array(node != 0)
        vector = node @ adjacency
        vector *= history
        history = numpy.array(history != 0, dtype=numpy.float64)
        children = [i for i in numpy.diag(vector) if i.any()]
        if not children:
            children = [numpy.zeros(len(node), dtype=numpy.float64)]
        return {'children': children, 'history': history}

    def two_step(self, node, history=None):
        """
        Returns the child nodes two steps from a node. A simple method
        to ensure that there is no backtracking.

        Parameters
        ----------
        node : numpy.ndarray, dtype=numpy.float64
        history : numpy.ndarray, dtype=numpy.float64

        Returns
        -------
        list of numpy.ndarray, dtype=numpy.float64
        """
        returned = []
        node_info = self.node_to_children(node, self.adj, history)
        for child_node in node_info['children']:
            nxt = self.node_to_children(child_node, self.adj,
                                        node_info['history'])
            if nxt['children']:
                returned.append(nxt)
        return returned

    def go_to_depth(self, node, depth, history=None):
        """
        Returns a vector of the potential path end nodes given a start
        node, adjacency matrix, and number of edges to traverse.

        Parameters
        ----------
        node : numpy.ndarray, dtype=numpy.float64
        depth : scalar
            number of edges to traverse in potential paths
        history : numpy.ndarray, dtype=numpy.float64

        Returns
        -------
        numpy.ndarray, dtype=numpy.float64
            target nodes where ones indicate a path connection of the
            correct length from the queried start node to the end node.
        """
        assert depth <= 10  # We don't need arbitrarily high depths.
        if depth == 0:
            self.paths += node
        elif depth == 2:
            for i in self.two_step(node, history):
                for j in i['children']:
                    self.paths += j
        else:
            nodes = self.node_to_children(node, self.adj, history)
            for child in nodes['children']:
                self.go_to_depth(child, depth - 1, nodes['history'])


def index_to_nodes(adj, index, depth):
    """
    Traces a single node through the graph to a given depth
    and returns the nodes which can be reached in exactly
    the number of steps specified.

    Parameters
    ----------
    adj : numpy.ndarray
        an adjacency matrix will give the path counts, while
        a normalized adjacency matrix will return the degree-
        weighted path counts
    index : int
        index of the start node
    depth : int
        number of edges in the metapath

    Returns
    -------
    1-dimensional numpy.ndarray
        the nodes which can be reached within the exact number
        of steps specified and the DWPC for each
    """
    search = numpy.zeros(adj.shape[0])
    search[index] = 1
    a = TraverseLongRepeat(search, adj)
    a.go_to_depth(a.start, depth)
    return a.paths


def dwpc_long_repeat(graph, metapath, damping=0.5):
    """
    One metanode repeated 4 or more times. Considerably slower than
    dwpc_short_repeat, so should only be used if necessary. This
    function uses history vectors that split the computation into more
    tasks.

    Computes the degree-weighted path count when a single
    metanode/metaedge is repeated an arbitrary number of times.

    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    damping : float
        damping=0 returns the unweighted path counts

    Returns
    -------
    nodes, nodes, path_matrix : Tuple[list, list, numpy.ndarray]

    Notes
    -----
    The metapath given must be either a metaedge-repeat only
    metapath or a segment which follows this criterion. It is
    only useful to use this function over dwpc_duplicated_metanode
    in metapaths and metapath segments which have three or more
    edges. The purpose of this DWPC method is to eliminate the
    counting of (within metapath A-A-A-A) the path a-b-c-b.
    This method is nonspecific to length, and will eliminate node
    repeats for any length metanode repeats. However, this is
    a very slow method compared to others, and should be used
    sparingly, only for the few metapaths that demand its
    application.
    """

    # Check that the metapath is just one repeated metaedge
    metanodes = set(metapath.get_nodes())
    metaedges = set(metapath.edges)
    assert len(metanodes) == 1
    assert len(metaedges) == 1

    depth = len(metapath)
    metaedge = metaedges.pop().get_abbrev()

    row, col, adjacency_matrix = metaedge_to_adjacency_matrix(
        graph, metaedge, dtype=numpy.float64, sparse_threshold=0)
    nnode = adjacency_matrix.shape[0]  # number of nodes

    # Weight the adjacency matrix
    weighted_adj = degree_weight(adjacency_matrix, damping, damping)

    source_to_destinations = functools.partial(index_to_nodes,
                                               adj=weighted_adj,
                                               depth=depth)

    # Perform the actual path counting
    full_array = [source_to_destinations(index=i) for i in range(nnode)]
    full_array = numpy.array(full_array, dtype=numpy.float64)
    return row, col, full_array


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

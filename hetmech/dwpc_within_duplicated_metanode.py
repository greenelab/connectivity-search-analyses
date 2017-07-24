import functools

import numpy

from .degree_weight import dwwc_step
from .matrix import metaedge_to_adjacency_matrix


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


class Traverse:
    """
    Class is run by the following:

    var_name = Traverse(search_node, adjacency_matrix)
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
        node_info = node_to_children(node, self.adj, history)
        for child_node in node_info['children']:
            nxt = node_to_children(child_node, self.adj, node_info['history'])
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
            nodes = node_to_children(node, self.adj, history)
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
    a = Traverse(search, adj)
    a.go_to_depth(a.start, depth)
    return a.paths


def dwpc_same_metanode(graph, metapath, damping=0.5):
    """
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
    weighted_adj = dwwc_step(adjacency_matrix, damping, damping)

    source_to_destinations = functools.partial(index_to_nodes,
                                               adj=weighted_adj,
                                               depth=depth)

    # Perform the actual path counting
    full_array = [source_to_destinations(index=i) for i in range(nnode)]
    full_array = numpy.array(full_array, dtype=numpy.float64)
    return row, col, full_array

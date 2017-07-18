import numpy


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
    history -= node
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
        numpy.ndarray, dtype=numpy.float64 of target nodes where
        ones indicate a path connection of the correct length from
        the queried start node to the end node.
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

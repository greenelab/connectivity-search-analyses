import numpy
import collections

from .matrix import metaedge_to_adjacency_matrix
from .degree_weight import dwwc_step


def dwpc(graph, metapath, damping=1.0):
    """
    Compute the degree-weighted path count (DWPC).
    First checks that no more than one metanode type is repeated.
    """
    dwpc_matrix = None
    # First read through sequence of metanode types
    nodetype_sequence = collections.defaultdict(int)
    for metaedge in metapath:
        metanodes = metaedge.get_nodes()
        nodetype_sequence[metanodes[0]] += 1
        nodetype_sequence[metanodes[1]] += 1

    # Identify the repeated node type
    # and check that no more than 1 metanodetype is repeated
    repeated_node = filter(lambda x: nodetype_sequence[x] > 2,
                           nodetype_sequence)
    if len(repeated_node) > 1:
        print("Input metapath repeats more than one nodetype")
        return dwpc_matrix

    elif len(repeated_node) == 0:
        # Now perform multiplications
        for metaedge in metapath:
            adj_mat = metaedge_to_adjacency_matrix(graph, metaedge)
            adj_mat = dwwc_step(adj_mat, damping, damping)
            if dwpc_matrix is None:
                dwpc_matrix = adj_mat
            else:
                dwpc_matrix = dwpc_matrix @ adj_mat
    else:
        # Determine start/endpoints for left,loop,right
        first_appearance = None
        last_appearance = None
        for idx, metaedge in enumerate(metapath):
            metanodes = metaedge.get_nodes()
            if repeated_node[0] == metanodes[0]:
                if first_appearance is None:
                    first_appearance = idx
            if repeated_node[0] == metanodes[1]:
                last_appearance = idx

        # Handle head
        left_matrix = None
        for idx, metaedge in enumerate(metapath):
            adj_mat = metaedge_to_adjacency_matrix(graph, metaedge)
            adj_mat = dwwc_step(adj_mat, damping, damping)
            metanodes = metaedge.get_nodes()
            if idx < first_appearance:  # haven't seen repeatednode yet
                if left_matrix is None:
                    left_matrix = adj_mat
                else:
                    left_matrix = left_matrix @ adj_mat
            elif idx < last_appearance:  # i.e. metanodes[0] = repeatednode
                if dwpc_matrix is None:
                    dwpc_matrix = adj_mat
                else:
                    dwpc_matrix = dwpc_matrix @ adj_mat
                    # if endpoints are same type, subtract diagonal after mult
                    if metanodes[1] == repeated_node[0]:
                        dwpc_matrix -= numpy.diag(numpy.diag(dwpc_matrix))
            else:
                dwpc_matrix = dwpc_matrix @ adj_mat

    return left_matrix @ dwpc_matrix

import numpy
import collections
from scipy.sparse import csr_matrix
import scipy.sparse

from .matrix import metaedge_to_adjacency_matrix
from .degree_weight import dwwc_step


def dwpc(graph, metapath, damping=1.0, verbose=False):
    """
    Compute the degree-weighted path count (DWPC).
    First checks that no more than one metanode type is repeated.
    """
    print("Calling DWPC")
    dwpc_matrix = None
    # First read through sequence of metanode types
    nodetype_sequence = collections.defaultdict(int)
    for metaedge in metapath:
        # metanodes = metaedge.get_nodes()
        nodetype_sequence[metaedge.source] += 1
        nodetype_sequence[metaedge.target] += 1
    # Identify the repeated node type
    # and check that no more than 1 metanodetype is repeated
    repeated_node = filter(lambda x: nodetype_sequence[x] > 2,
                           nodetype_sequence)
    repeated_node = list(repeated_node)
    if len(repeated_node) > 1:
        if verbose:
            print("Input metapath repeats more than one nodetype")
        return dwpc_matrix

    elif len(repeated_node) == 0:
        if verbose:
            print("Input metapath has no repeated nodetypes")
        # Now perform multiplications
        for metaedge in metapath:
            adj_mat = metaedge_to_adjacency_matrix(graph, metaedge)
            adj_mat = dwwc_step(adj_mat, damping, damping)
            if dwpc_matrix is None:
                dwpc_matrix = csr_matrix(adj_mat)
            else:
                dwpc_matrix = dwpc_matrix @ adj_mat
    else:
        # Determine start/endpoints for left,loop,right
        if verbose:
            print("Input metapath repeats exactly one nodetype")
            print("Repeated node list: {}, type {}".format(repeated_node, repeated_node[0]))
        
        first_appearance = None
        last_appearance = None
        for idx, metaedge in enumerate(metapath):
            if repeated_node[0] == metaedge.source:
                if first_appearance is None:
                    first_appearance = idx
            if repeated_node[0] == metaedge.target:
                last_appearance = idx
        if verbose:
            print("metapath has {} nodes, and {} are the repeated ones"
                  "".format(len(metapath),[first_appearance, last_appearance]))

        # Handle head
        left_matrix = None
        for idx, metaedge in enumerate(metapath):
            if verbose:
                print("\tWorking on {}".format(idx))
            adj_mat = metaedge_to_adjacency_matrix(graph, metaedge)
            adj_mat = dwwc_step(adj_mat, damping, damping)
            adj_mat = csr_matrix(adj_mat)
            # metanodes = metaedge.get_nodes()
            if idx < first_appearance:  # haven't seen repeatednode yet
                if verbose:
                    print("\t\tleft_matrix {}".format(idx))
                if left_matrix is None:
                    if verbose:
                        print("\t\tleft_matrix {} initialize; shape adj {}".format(idx, adj_mat.shape))
                    left_matrix = csr_matrix(adj_mat)
                else:
                    if verbose:
                        print("\t\tleft_matrix {} multiply; shape left {} ; shape adj {}".format(idx, left_matrix.shape, adj_mat.shape))
                    if verbose:
                        print("\t\tType of left_matrix before = {}".format(type(left_matrix)))
                    left_matrix = left_matrix @ adj_mat
            elif idx <= last_appearance:  # i.e. metanodes[0] = repeatednode
                if verbose:
                    print("\t\tloop_matrix {}".format(idx))
                if dwpc_matrix is None:
                    if verbose:
                        print("\t\tloop_matrix {} initialize; shape adj {}".format(idx, adj_mat.shape))
                    dwpc_matrix = csr_matrix(adj_mat)
                else:
                    if verbose:
                        print("\t\tloop_matrix {} multiply; shape loop {} ; shape adj {}".format(idx, dwpc_matrix.shape, adj_mat.shape))
                        print("\t\tType of loop_matrix before = {}".format(type(dwpc_matrix)))
                    adj_mat = csr_matrix(adj_mat)
                    dwpc_matrix = dwpc_matrix @ adj_mat
                    # if endpoints are same type, subtract diagonal after mult
                    if metaedge.target == repeated_node[0]:
                        if verbose:
                            print("\tsubtracting diag {} ".format(idx))
                            # np.diag(diagonals[0], offsets[0])
                            # scipy version here
                            print(dwpc_matrix.diagonal().shape)
                        dwpc_matrix -= scipy.sparse.diags( [dwpc_matrix.diagonal()], [0] )
            else:
                if verbose:
                    print("\t\ttail_matrix {}".format(idx))
                dwpc_matrix = dwpc_matrix @ adj_mat

    return left_matrix @ dwpc_matrix
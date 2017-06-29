def get_segments_multi(metagraph, metapath):
    """
    The intention is to call this method from "get_segments()"
    in the case that overlapping segments with duplicate nodes are detected.

    This function will then check for two possibilities:
        (1) ABAB or BABA --- our code will handle these cases
        (2) Anything else, i.e. BACBAC --- such cases raise an error

    For ABAB
        In the most general version of this, it is possible that there are
        additional segments around and between the repeated ones, i.e.
        UAXBYAZBV. Note that U, X, Y, Z, V can each represent entire segments
        consisting of nodes that are not repeated anywhere else. A given
        segment can include a single nodetype repeated, as long as no node in
        any of these segments gets repeated in some other segment. So the
        segment U could be nodes "DFGD" as long as D,F,G are not repeated
        anywhere else in the original metapath.

    Output: Extract the segments UA, BV, AXB, BYA, AZB

    For BAAB  (more generally, UBXAYAZBV)
        Work in progress, will fill in later.

    """


def dwpc_baba(graph, metapath, duplicates=None, damping=0.5,
              sparse_threshold=0):
    """
    Compute the degree-weighted path count (DWPC) when two metanodes are
    duplicated and interleaved as BABA.
    User must specify the duplicated metanodes.
    This should be called from inside the larger dwpc function

    Parameters
    ==========
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    duplicates : hetio.hetnet.MetaNode or None
    damping : float
    sparse_threshold : float (0 <= sparse_threshold <= 1)
        sets the density threshold above which a sparse matrix will be
        converted to a dense automatically.

    [1] Use get_segments_multi above to obtain the appropriate segments
    [2] Get the matrices for the individual edges of each metpath segment,
        multiply together to get a single matrix for each segment in 1.1 (while
        weighting appropriately for the DWPC). For a given segment S, let P_S
         be the DWPC matrix for S. So after this step, we should have computed
         P_(UA), (P_(AXB), P_(BYA), P_(AZB), P_(BV)
    [3] Set M = diag(P_(AXB) * P_(BYA)) * P_(AZB)
          Set N = P_(AXB) * diag(P_(BYA) * P_(AZB))
    [4] Now we have "P_(AXB) * P_(BYA) * P_(AZB) - M - N " gives all the
        paths in segment "AXB,BYA,AZB" (and weights them appropriately), except
        we have *over*compensated by subtracting a bit too much. We need to add
        back into the computation a quantity accounting for all paths in the
        metapath that go through a loop in metanode A as well as a loop in
        metanode B.

        For each node r in A and each ndoe c in B, we design a new matrix R
        as follows:
            R_{r,c} = P_(AXB)_{r,c} * P_(BYA)_{c,r} * P_(AZB)_{r,c}

        This can be computed rapidly via the Hadamard product, which can be
        accomplished via numpy.multiply:
        https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.multiply.html

        R = P_(AXB) (hadamard product) (P_(BYZ))^T (hadamard product) P_(AZB)
        R = numpy.multiply(P_(AXB), numpy.multiply(P_(BYZ).transpose, P_(AZB)))

        Then the full DWPC matrix is
        P_(UAXBYAZBV) =
                P_(UA) * (P_(AXB) * P_(BYA) * P_(AZB) - M - N + R) * P_(BV)
    """

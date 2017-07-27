import hetio.readwrite
import numpy
import pytest

from .dwpc import TraverseLongRepeat, index_to_nodes, dwpc_long_repeat


def get_node(index):
    """Return a vector with a one at the given index"""
    diag = numpy.eye(5, dtype=numpy.float64)
    return diag[index]


def get_adj(which):
    adj1 = numpy.array([[0., 1., 1., 0., 0.],
                        [1., 0., 0., 1., 0.],
                        [1., 0., 0., 1., 0.],
                        [0., 1., 1., 0., 1.],
                        [0., 0., 0., 1., 0.]], dtype=numpy.float64)
    adj2 = numpy.array([[0, 1, 1, 1, 0],
                        [1, 0, 1, 0, 0],
                        [1, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0]], dtype=numpy.float64)

    select_adjacency = {1: adj1, 2: adj2}
    return select_adjacency[which]


@pytest.mark.parametrize('node,solution', [(0, [1, 2]),
                                           (1, [0, 3]),
                                           (2, [0, 3]),
                                           (3, [1, 2, 4]),
                                           (4, [3])])
def test_node_to_children(node, solution):
    """Test the basic functionality of node_to_children to output
    child nodes and an updated history vector"""
    adj = get_adj(1)

    children = TraverseLongRepeat.node_to_children(
        get_node(node), adj)['children']
    solution = [get_node(i) for i in solution]
    diff = [v - solution[i] for i, v in enumerate(children)]
    those_equal = [not i.any() for i in diff]
    assert all(those_equal)


def get_step_solutions(index, step, whole=False):
    step0 = {i: get_node(i) for i in range(5)}
    step1 = {0: [0, 1, 1, 1, 0], 1: [1, 0, 1, 0, 0], 2: [1, 1, 0, 1, 0],
             3: [1, 0, 1, 0, 1], 4: [0, 0, 0, 1, 0]}
    step2 = {0: [0, 1, 2, 1, 1], 1: [1, 0, 1, 2, 0], 2: [2, 1, 0, 1, 1],
             3: [1, 2, 1, 0, 0], 4: [1, 0, 1, 0, 0]}
    step3 = {0: [0, 1, 0, 1, 1], 1: [1, 0, 1, 2, 2], 2: [0, 1, 0, 1, 1],
             3: [1, 2, 1, 0, 0], 4: [1, 2, 1, 0, 0]}
    step4 = {0: [0, 0, 0, 0, 1], 1: [0, 0, 0, 0, 2], 2: [0, 0, 0, 0, 1],
             3: [0, 0, 0, 0, 0], 4: [1, 2, 1, 0, 0]}
    step5 = {i: [0, 0, 0, 0, 0] for i in range(5)}
    which_step = {0: step0, 1: step1, 2: step2, 3: step3, 4: step4, 5: step5}
    if whole:
        return [list(i.values()) for i in list(which_step.values())]
    return which_step[step][index]


@pytest.mark.parametrize('step', [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize('node', [0, 1, 2, 3, 4])
def test_traverse(node, step):
    """Test the ability of Traverse to give path counts for the
    depth given."""
    adj = get_adj(2)

    start_node = get_node(node)
    solution = numpy.array(get_step_solutions(node, step), dtype=numpy.float64)

    a = TraverseLongRepeat(start_node, adj)
    a.go_to_depth(a.start, step)
    output = a.paths
    assert numpy.array_equal(output, solution)


def get_matrices(abbrev, depth):
    """
    Return a path-count matrix for a chosen example adjacency
    matrix and chosen depth
    """
    A = get_step_solutions(None, None, whole=True)
    B = {
        0: list(numpy.eye(4, dtype=numpy.float64)),
        1: [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]],
        2: [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
        3: [[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
        4: 4*[[0, 0, 0, 0]],
        5: 4*[[0, 0, 0, 0]]}
    C = {
        0: list(numpy.eye(8, dtype=numpy.float64)),
        1: [[0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0]],
        2: [[0, 0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 1], [1, 1, 1, 1, 0, 1, 1, 0]],
        3: [[0, 0, 1, 1, 1, 0, 1, 1], [0, 0, 0, 2, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 2, 0, 0], [1, 2, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 2], [0, 0, 2, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 2, 1, 1, 0]],
        4: [[0, 0, 0, 2, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 0], [2, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 2], [0, 1, 0, 1, 1, 0, 2, 1],
            [1, 0, 0, 0, 1, 2, 0, 0], [0, 0, 0, 1, 2, 1, 0, 0]],
        5: [[0, 0, 1, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 1, 0, 0]]}
    matrix_dict = {'A': A, 'B': B, 'C': C}
    return matrix_dict[abbrev][depth]


@pytest.mark.parametrize('depth', [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize('matrix', ['A', 'B', 'C'])
def test_index_to_nodes(matrix, depth):
    """
    Test the ability to get path-count vectors from the
    index_to_nodes function
    """
    solution = numpy.array(get_matrices(matrix, depth), dtype=numpy.float64)
    adj = numpy.array(get_matrices(matrix, 1), dtype=numpy.float64)

    output = [index_to_nodes(adj, i, depth) for i, v in enumerate(adj)]
    output = numpy.array(output, dtype=numpy.float64)

    assert numpy.array_equal(solution, output)


def get_expected(length):
    """
    Return degree-weighted path counts for the genes in the
    example graph below.
    """
    gig1 = [[0., 0., 0.35355339, 0., 0.70710678, 0., 0.],
            [0., 0., 0.5, 0., 0., 0., 0.],
            [0.35355339, 0.5, 0., 0.5, 0., 0., 0.5],
            [0., 0., 0.5, 0., 0., 0., 0.],
            [0.70710678, 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0.5, 0., 0., 0., 0.]]
    gig2 = [[0., 0.1767767, 0., 0.1767767, 0., 0., 0.1767767],
            [0.1767767, 0., 0., 0.25, 0., 0., 0.25],
            [0., 0., 0., 0., 0.25, 0., 0.],
            [0.1767767, 0.25, 0., 0., 0., 0., 0.25],
            [0., 0., 0.25, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0.1767767, 0.25, 0., 0.25, 0., 0., 0.]]
    gig3 = [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0],
            [0, 0.125, 0, 0.125, 0, 0, 0.125],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0]]
    gig4 = gig5 = gig6 = numpy.zeros((7, 7))
    mat_dict = {1: gig1, 2: gig2, 3: gig3, 4: gig4,
                5: gig5, 6: gig6}
    genes = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
    exp_mat = numpy.array(mat_dict[length], dtype=numpy.float64)
    return genes, genes, exp_mat


@pytest.mark.parametrize('length', list(range(1, 6)))
def test_dwpc_same_metanode(length):
    """
    Test the functionality of dwpc_same_metanode to find DWPC
    within a metapath (segment) of metanode and metaedge repeats.
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metagraph = graph.metagraph
    m_path = 'GiG' + length*'iG'
    metapath = metagraph.metapath_from_abbrev(m_path)
    rows, cols, dwpc_mat = dwpc_long_repeat(graph, metapath, damping=0.5)
    exp_row, exp_col, exp_dwpc = get_expected(length)

    # Test matrix, row, and column label output
    assert pytest.approx((dwpc_mat - exp_dwpc).sum(), 0, abs=1e-7)
    assert rows == exp_row
    assert cols == exp_col

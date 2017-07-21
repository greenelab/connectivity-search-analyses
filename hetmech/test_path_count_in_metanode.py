import hetio.readwrite
import numpy
import pytest

from .dwpc_within_duplicated_metanode import node_to_children, Traverse, \
    dwpc_same_metanode


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


@pytest.mark.parametrize('node', [(0, [1, 2]),
                                  (1, [0, 3]),
                                  (2, [0, 3]),
                                  (3, [1, 2, 4]),
                                  (4, [3])])
def test_node_to_children(node):
    """Test the basic functionality of node_to_children to output
    child nodes and an updated history vector"""
    adj = get_adj(1)

    children = node_to_children(get_node(node[0]), adj)['children']
    solution = [get_node(i) for i in node[1]]
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

    a = Traverse(start_node, adj)
    a.go_to_depth(a.start, step)
    output = a.paths
    assert numpy.array_equal(output, solution)


def get_expected(m_path):
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
    mat_dict = {'GiG': gig1, 'GiGiG': gig2, 'GiGiGiG': gig3, 'GiGiGiGiG': gig4,
                'GiGiGiGiGiG': gig5, 'GiGiGiGiGiGiG': gig6}
    genes = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
    exp_mat = numpy.array(mat_dict[m_path], dtype=numpy.float64)
    return genes, genes, exp_mat


@pytest.mark.parametrize('m_path', ['GiG' + i*'iG' for i in range(6)])
def test_dwpc_same_metanode(m_path):
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
    metapath = metagraph.metapath_from_abbrev(m_path)
    rows, cols, dwpc_mat = dwpc_same_metanode(graph, metapath, damping=0.5)
    exp_row, exp_col, exp_dwpc = get_expected(m_path)

    # Test matrix, row, and column label output
    assert pytest.approx((dwpc_mat - exp_dwpc).sum(), 0, abs=1e-7)
    assert rows == exp_row
    assert cols == exp_col

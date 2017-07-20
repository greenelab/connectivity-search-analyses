import pytest

from .dwpc_within_duplicated_metanode import *


def get_node(index):
    """Return a vector with a one at the given index"""
    diag = numpy.eye(5, dtype=numpy.float64)
    return diag[index]


@pytest.mark.parametrize('node', [(0, [1, 2, 3]),
                                  (1, [0, 2]),
                                  (2, [0, 1, 3]),
                                  (3, [0, 2, 4]),
                                  (4, [3])])
def test_node_to_children(node):
    """Test the basic functionality of node_to_children to output
    child nodes and an updated history vector"""
    adj = numpy.array([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ], dtype=numpy.float64)

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
    adj = numpy.array([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ], dtype=numpy.float64)

    start_node = get_node(node)
    solution = numpy.array(get_step_solutions(node, step), dtype=numpy.float64)

    a = Traverse(start_node, adj)
    a.go_to_depth(a.start, step)
    output = a.paths
    assert numpy.array_equal(output, solution)


def get_matrices(abbrev, depth):
    A = get_step_solutions(None, None, whole=True)
    B = {
        0: list(numpy.eye(4, dtype=numpy.float64)),
        1: [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]],
        2: [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
        3: [[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
        4: [[0, 0, 0, 0] for i in range(4)],
        5: [[0, 0, 0, 0] for i in range(4)]}
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


@pytest.mark.parametrize('matrix', ['A', 'B', 'C'])
@pytest.mark.parametrize('depth', [0, 1, 2, 3, 4, 5])
def test_all_paths(matrix, depth):
    """Test the ability to get full path-count matrices from the
    AllPaths class."""
    solution = numpy.array(get_matrices(matrix, depth), dtype=numpy.float64)
    adj = numpy.array(get_matrices(matrix, 1), dtype=numpy.float64)

    a = PathCount(adj, depth, None)
    output = a.iterate_rows()

    assert numpy.array_equal(solution, output)

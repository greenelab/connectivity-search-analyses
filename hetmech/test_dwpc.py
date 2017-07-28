import hetio.readwrite
import numpy
import pytest

from .dwpc import dwpc_baab, dwpc_general_case


@pytest.mark.parametrize('metapath,expected', [
    ('DaGiGaD', [[0., 0.47855339],
                 [0.47855339, 0.]]),
    ('TeGiGeT', [[0, 0],
                 [0, 0]]),
    ('DaGiGeTlD', [[0, 0],
                   [0, 0]]),
    ('DaGeTeGaD', [[0, 0],
                   [0, 0]]),
    ('TlDaGiGeT', [[0., 0.47855339],
                   [0., 0.]])
])
def test_dwpc_baab(metapath, expected):
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    exp_row = node_dict[metapath[0]]
    exp_col = node_dict[metapath[-1]]
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metapath = graph.metagraph.metapath_from_abbrev(metapath)

    row, col, dwpc_matrix = dwpc_baab(graph, metapath, damping=0.5)

    expected = numpy.array(expected, dtype=numpy.float64)

    assert abs(dwpc_matrix - expected).sum() == pytest.approx(0, abs=1e-7)
    assert exp_row == row
    assert exp_col == col


def get_general_solutions(length):
    genes = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
    mat_dict = {
        0: [[0, 0, 0.35355339, 0, 0.70710678, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0],
            [0.35355339, 0.5, 0, 0.5, 0, 0, 0.5],
            [0, 0, 0.5, 0, 0, 0, 0],
            [0.70710678, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0]],
        1: [[0, 0.1767767, 0, 0.1767767, 0, 0, 0.1767767],
            [0.1767767, 0, 0, 0.25, 0, 0, 0.25],
            [0, 0, 0, 0, 0.25, 0, 0],
            [0.1767767, 0.25, 0, 0, 0, 0, 0.25],
            [0, 0, 0.25, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0.1767767, 0.25, 0, 0.25, 0, 0, 0]],
        2: [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0],
            [0, 0.125, 0, 0.125, 0, 0, 0.125],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0]],
        3: numpy.zeros((7, 7)),
        4: numpy.zeros((7, 7)),
        5: numpy.zeros((7, 7))
    }
    return genes, genes, mat_dict[length]


@pytest.mark.parametrize('length', list(range(6)))
def test_dwpc_general_case(length):
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
    rows, cols, dwpc_mat = dwpc_general_case(graph, metapath, damping=0.5)
    exp_row, exp_col, exp_dwpc = get_general_solutions(length)

    # Test matrix, row, and column label output
    assert abs(dwpc_mat - exp_dwpc).sum() == pytest.approx(0, abs=1e-7)
    assert rows == exp_row
    assert cols == exp_col


@pytest.mark.parametrize('damping', [0, 0.5, 1])
@pytest.mark.parametrize('metapath,category', [
    ('DaGiGaD', 'BAAB'), ('TeGiGeT', 'BAAB'), ('DaGiGeTlD', 'BAAB'),
    ('DaGeTeGaD', 'BAAB'), ('TlDaGiGeT', 'BAAB')
])
def test_general_equals_other(metapath, category, damping):
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metapath = graph.metagraph.metapath_from_abbrev(metapath)

    cat_dict = {'BAAB': dwpc_baab}
    exp_row, exp_col, exp_dwpc = cat_dict[category](graph, metapath, damping)
    row, col, dwpc = dwpc_general_case(graph, metapath, damping)

    assert row == exp_row
    assert col == exp_col
    assert abs(dwpc - exp_dwpc).sum() == pytest.approx(0, abs=1e-7)

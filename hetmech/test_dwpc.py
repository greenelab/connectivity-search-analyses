import hetio.readwrite
import numpy
import pytest

from .dwpc import dwpc_baab, dwpc_baba


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


def get_baba_matrices(metapath):
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    edge_dict = {
        0: [[0.08838835, 0],
            [0.08838835, 0],
            [0, 0.125],
            [0.08838835, 0],
            [0, 0],
            [0, 0],
            [0, 0]],
        1: [[0, 0],
            [0, 0]],
        2: [[0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]],
        3: [[0.25, 0.],
            [0.25, 0.],
            [0., 0.],
            [0.25, 0.],
            [0., 0.],
            [0.1767767, 0.],
            [0., 0.]],
        4: [[0., 0.],
            [0., 0.],
            [0.125, 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]],
        5: [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.25],
            [0., 0.]]
    }
    mat_dict = {
        'GaDaGaD': (0, 0),
        'DaGaDaG': (0, 1),
        'DlTlDlT': (1, 0),
        'TlDlTlD': (1, 1),
        'GeTeGeT': (2, 0),
        'TeGeTeG': (2, 1),
        'GaDlTeGaD': (3, 0),
        'DaGeTlDaG': (3, 1),
        'GeTlDaGaD': (4, 0),
        'DaGaDlTeG': (4, 1),
        'GaDaGeTlD': (5, 0),
        'DlTeGaDaG': (5, 1)
    }
    first = node_dict[metapath[0]]
    last = node_dict[metapath[-1]]
    edge = mat_dict[metapath]
    adj = numpy.array(edge_dict[edge[0]], dtype=numpy.float64)
    if edge[1]:
        adj = adj.transpose()
    return first, last, adj


@pytest.mark.parametrize('m_path', ('GaDaGaD', 'DaGaDaG', 'DlTlDlT',
                                    'TlDlTlD', 'GeTeGeT', 'TeGeTeG',
                                    'GaDlTeGaD', 'GeTlDaGaD', 'GaDaGeTlD'))
def test_dwpc_baba(m_path):
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metagraph = graph.metagraph
    metapath = metagraph.metapath_from_abbrev(m_path)

    row_sol, col_sol, adj_sol = get_baba_matrices(m_path)
    row, col, dwpc = dwpc_baba(graph, metapath, damping=0.5)

    assert row_sol == row
    assert col_sol == col
    assert numpy.max(adj_sol - dwpc) == pytest.approx(0, abs=1e-8)

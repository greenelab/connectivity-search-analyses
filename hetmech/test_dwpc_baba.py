import hetio.readwrite
import numpy
import pytest

from .dwpc_baba import dwpc_baba


def get_matrices(metapath):
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    edge_dict = {
        'GaD': [[0.08838835, 0],
                [0.08838835, 0],
                [0, 0.125],
                [0.08838835, 0],
                [0, 0],
                [0, 0],
                [0, 0]],
        'DlT': [[0, 0],
                [0, 0]],
        'TeG': [[0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]],
        'c': [[1, 0],
              [1, 0],
              [0, 0],
              [1, 0],
              [0, 0],
              [1, 0],
              [0, 0]]
    }
    mat_dict = {
        'GaDaGaD': ('GaD', 0),
        'DaGaDaG': ('GaD', 1),
        'DlTlDlT': ('DlT', 0),
        'TlDlTlD': ('DlT', 0),
        'GeTeGeT': ('TeG', 0),
        'TeGeTeG': ('TeG', 1),
        'GaDlTeGaD': ('c', 0)
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
                                    'GaDlTeGaD'))
def test_dwpc_baba(m_path):
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metagraph = graph.metagraph
    metapath = metagraph.metapath_from_abbrev(m_path)

    row_sol, col_sol, adj_sol = get_matrices(m_path)
    row, col, dwpc = dwpc_baba(graph, metapath, damping=0.5)

    assert row_sol == row
    assert col_sol == col
    assert pytest.approx(adj_sol, dwpc)

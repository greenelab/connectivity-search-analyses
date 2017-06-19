import hetio.readwrite
import numpy
import pytest
from scipy import sparse

from .degree_weight import dwwc, dwpc_duplicated_metanode


def test_disease_gene_example_dwwc():
    """
    Test the PC & DWWC computations in Figure 2D of Himmelstein & Baranzini
    (2015) PLOS Comp Bio. https://doi.org/10.1371/journal.pcbi.1004259.g002
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metagraph = graph.metagraph

    # Compute GiGaD path count and DWWC matrices
    metapath = metagraph.metapath_from_abbrev('GiGaD')
    rows, cols, wc_matrix = dwwc(graph, metapath, damping=0)
    rows, cols, dwwc_matrix = dwwc(graph, metapath, damping=0.5)

    # Check row and column name assignment
    assert rows == ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
    assert cols == ["Crohn's Disease", 'Multiple Sclerosis']

    # Check concordance with https://doi.org/10.1371/journal.pcbi.1004259.g002
    i = rows.index('IRF1')
    j = cols.index('Multiple Sclerosis')

    # Warning: the WC (walk count) and PC (path count) are only equivalent
    # because none of the GiGaD paths contain duplicate nodes. Since, GiGaD
    # contains duplicate metanodes, WC and PC are not guaranteed to be the
    # same. However, they happen to be equivalent for this example.
    assert wc_matrix[i, j] == 3
    assert dwwc_matrix[i, j] == pytest.approx(0.25 + 0.25 + 32 ** -0.5)


def get_expect(m_path, auto):
    if m_path in ('GiGaD', 'GaDaG') and auto:
        mattype = 'dense'
    else:  # Others never reach threshold density
        mattype = 'sparse'
    mat_dict = {
        'GiGaD': [[0.25, 0.],
                  [0.35355339, 0.],
                  [0., 0.6767767],
                  [0.35355339, 0.],
                  [0., 0.35355339],
                  [0., 0.],
                  [0.35355339, 0.]],
        'GaDaG': [[0.25, 0.25, 0., 0.25, 0., 0.1767767, 0.],
                  [0.25, 0.25, 0., 0.25, 0., 0.1767767, 0.],
                  [0., 0., 0.5, 0., 0., 0.35355339, 0.],
                  [0.25, 0.25, 0., 0.25, 0., 0.1767767, 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0.1767767, 0.1767767, 0.35355339, 0.1767767, 0., 0.375, 0.],
                  [0., 0., 0., 0., 0., 0., 0.]],
        'GeTlD': [[0., 0.],
                  [0., 0.],
                  [0., 0.70710678],
                  [0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [0., 0.]],
        'GiG': [[0., 0., 0.35355339, 0., 0.70710678, 0., 0.],
                [0., 0., 0.5, 0., 0., 0., 0.],
                [0.35355339, 0.5, 0., 0.5, 0., 0., 0.5],
                [0., 0., 0.5, 0., 0., 0., 0.],
                [0.70710678, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0.5, 0., 0., 0., 0.]],
        'GaDaG_dwpc': [[0., 0.25, 0., 0.25, 0., 0.1767767, 0.],
                       [0.25, 0., 0., 0.25, 0., 0.1767767, 0.],
                       [0., 0., 0., 0., 0., 0.35355339, 0.],
                       [0.25, 0.25, 0., 0., 0., 0.1767767, 0.],
                       [0., 0., 0., 0., 0., 0., 0.],
                       [0.1767767, 0.1767767, 0.35355339, 0.1767767, 0.,
                        0., 0.],
                       [0., 0., 0., 0., 0., 0., 0.]]
    }
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    if m_path == 'GaDaG':
        dwwc_mat = numpy.array(mat_dict[m_path])
        dwpc_mat = numpy.array(mat_dict['GaDaG_dwpc'])
    else:
        dwwc_mat = dwpc_mat = numpy.array(mat_dict[m_path])
    row_names = node_dict[m_path[0]]
    col_names = node_dict[m_path[-1]]
    return mattype, dwwc_mat, dwpc_mat, row_names, col_names


@pytest.mark.parametrize('auto', [True, False])
@pytest.mark.parametrize('m_path', ('GiGaD', 'GaDaG', 'GeTlD', 'GiG'))
def test_dwpc_duplicated_metanode(m_path, auto):
    """
    Test the ability of dwwc to convert dwwc_matrix to a dense array when the
    percent nonzero goes above the 1/3 threshold. If auto is off, the matrices
    will start as sparse and stay sparse throughout. If auto is on, the
    matrices start sparse and will be converted to dense arrays when their
    densities exceeds 1/3.
    Also tests the matrix output in generating degree-weighted paths.
    """

    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metagraph = graph.metagraph
    metapath = metagraph.metapath_from_abbrev(m_path)
    dup = metapath.get_nodes()[0]
    rows, cols, dwwc_mat = dwpc_duplicated_metanode(
        graph, metapath, damping=0.5, auto=auto, mat_type=sparse.csc_matrix,
        duplicate=None)
    rows, cols, dwpc_mat = dwpc_duplicated_metanode(
        graph, metapath, damping=0.5, auto=auto, mat_type=sparse.csc_matrix,
        duplicate=dup)

    exp_type, exp_dwwc, exp_dwpc, exp_row, exp_col = get_expect(m_path, auto)

    # Test AUTO mode
    if exp_type == 'dense':
        assert isinstance(dwwc_mat, numpy.ndarray)
        assert isinstance(dwpc_mat, numpy.ndarray)
    else:
        assert sparse.issparse(dwwc_mat)
        assert sparse.issparse(dwpc_mat)

    # Test row and column label output
    assert (dwwc_mat - exp_dwwc).sum() < 0.00001  # Assert equal
    assert (dwpc_mat - exp_dwpc).sum() < 0.00001  # Assert equal
    assert rows == exp_row
    assert cols == exp_col

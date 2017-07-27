import hetio.readwrite
import numpy
import pytest
from scipy import sparse

from .matrix import metaedge_to_adjacency_matrix, categorize, get_segments


def get_arrays(edge, dtype, threshold):
    # Dictionary with tuples of matrix and percent nonzero
    adj_dict = {
        'GiG': ([[0, 0, 1, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [1, 1, 0, 1, 0, 0, 1],
                 [0, 0, 1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0]], 0.204),
        'GaD': ([[0, 1],
                 [0, 1],
                 [1, 0],
                 [0, 1],
                 [0, 0],
                 [1, 1],
                 [0, 0]], 0.429),
        'DlT': ([[0, 0],
                 [1, 0]], 0.25),
        'TlD': ([[0, 1],
                 [0, 0]], 0.25)
    }
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    row_names = node_dict[edge[0]]
    col_names = node_dict[edge[-1]]

    if adj_dict[edge][1] < threshold:
        adj_matrix = sparse.csc_matrix(adj_dict[edge][0], dtype=dtype)
    else:
        adj_matrix = numpy.array(adj_dict[edge][0], dtype=dtype)

    return row_names, col_names, adj_matrix


@pytest.mark.parametrize('threshold', [0, 0.25, 0.5, 1])
@pytest.mark.parametrize("dtype", [numpy.bool_, numpy.int64, numpy.float64])
@pytest.mark.parametrize("test_edge", ['GiG', 'GaD', 'DlT', 'TlD'])
def test_metaedge_to_adjacency_matrix(test_edge, dtype, threshold):
    """
    Test the functionality of metaedge_to_adjacency_matrix in generating
    numpy arrays. Uses same test data as in test_degree_weight.py
    Figure 2D of Himmelstein & Baranzini (2015) PLOS Comp Bio.
    https://doi.org/10.1371/journal.pcbi.1004259.g002
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    row_names, col_names, adj_mat = metaedge_to_adjacency_matrix(
        graph, test_edge, dtype=dtype, sparse_threshold=threshold)
    exp_row, exp_col, exp_adj = get_arrays(test_edge, dtype, threshold)

    assert row_names == exp_row
    assert col_names == exp_col
    assert type(adj_mat) == type(exp_adj)
    assert adj_mat.dtype == dtype
    assert adj_mat.shape == exp_adj.shape
    assert (adj_mat != exp_adj).sum() == 0


@pytest.mark.parametrize('metapath,solution', [
    ('GiG', 'short_repeat'),
    ('GiGiGiG', 'long_repeat'),
    ('G' + 10 * 'iG', 'long_repeat'),
    ('GiGiGcGcG', 'long_repeat'),  # iicc
    ('GiGcGcGiG', 'long_repeat'),  # icci
    ('GcGiGcGaDrD', 'disjoint'),  # cicDD
    ('GcGiGaDrDrD', 'disjoint'),  # ciDDD
    ('CpDaG', 'no_repeats'),  # ABC
    ('DaGiGaDaG', 'other'),  # ABBAB
    ('DaGiGbC', 'disjoint'),  # ABBC
    ('DaGiGaD', 'BAAB'),  # ABBA
    ('GeAlDlAeG', 'BAAB'),  # ABCBA
    ('CbGaDrDaGeA', 'BAAB'),  # ABCCBD
    ('AlDlAlD', 'BABA'),  # ABAB
    ('CrCbGbCbG', 'other'),  # BBABA
    ('CbGiGbCrC', 'other'),
    ('CbGaDaGeAlD', 'BABA'),  # ABCBDC
    ('AlDaGiG', 'disjoint'),  # ABCC
    ('AeGaDaGiG', 'disjoint'),  # ABCB
    ('CbGaDpCbGaD', 0),  # ABCABC
    ('DaGiGiGiGiGaD', None),  # ABBBBBA
    ('CbGaDrDaGbC', 0),  # ABCCBA
    ('DlAuGcGpBPpGaDlA', 0),  # ABCCDCAB
    ('CrCbGiGaDrD', 'disjoint'),  # AABBCC
    ('CbGbCbGbC', 'other')])  # ABABA
def test_categorize(metapath, solution):
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/hetionet-v1.0-metagraph.json',
    )
    metagraph = hetio.readwrite.read_metagraph(url)
    metapath = metagraph.metapath_from_abbrev(metapath)
    if not solution:
        err_dict = {
            0: "Only two overlapping repeats currently supported",
            None: "Complex metapaths of length > 4 are not yet supported"}
        with pytest.raises(NotImplementedError) as err:
            categorize(metapath)
        assert str(err.value) == err_dict[solution]
    else:
        assert categorize(metapath) == solution


@pytest.mark.parametrize('metapath,solution', [
    ('AeGiGaDaG', '[AeG, GiGaDaG]'),  # short_repeat
    ('AeGiGeAlD', '[AeGiGeA, AlD]'),  # BAABC
    ('DaGaDaG', '[DaGaDaG]'),  # BABA
    ('DlAeGaDaG', '[DlAeGaDaG]'),  # BCABA
    ('GaDlAeGaD', '[GaDlAeGaD]'),  # BACBA
    ('GiGiG', '[GiGiG]'),  # short_repeat
    ('GiGiGiG', '[GiGiGiG]'),  # long_repeat
    ('CrCbGiGiGaDrDlA', '[CrC, CbG, GiGiG, GaD, DrD, DlA]'),
    ('CrCrCbGiGeAlDrD', '[CrCrC, CbG, GiG, GeAlD, DrD]'),
    ('SEcCrCrCbGiGeAlDrDpS', '[SEcC, CrCrC, CbG, GiG, GeAlD, DrD, DpS]'),
    ('SEcCrCrCrC', '[SEcC, CrCrCrC]')
])
def test_get_segments(metapath, solution):
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/hetionet-v1.0-metagraph.json',
    )
    metagraph = hetio.readwrite.read_metagraph(url)
    metapath = metagraph.metapath_from_abbrev(metapath)
    output = str(get_segments(metagraph, metapath))
    assert output == solution

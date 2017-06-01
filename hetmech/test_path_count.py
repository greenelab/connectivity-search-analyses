import hetio.pathtools
import hetio.readwrite
import pytest

from .degree_weight import dwpc


def get_bupropion_subgraph():
    """
    Read the bupropion and nicotine dependence Hetionet v1.0 subgraph.
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        'c1d47a942f2387cde4ee40c67b7dc5bd93b598f8',
        'test/data/bupropion-CbGpPWpGaD-subgraph.json.xz',
    )
    return hetio.readwrite.read_graph(url)


def test_CbGpPWpGaD_traversal():
    """
    Test path counts and degree-weighted path counts for the CbGpPWpGaD
    metapath between bupropion and nicotine dependence. Expected values from
    the network traversal methods at https://git.io/vHBh2.
    """
    graph = get_bupropion_subgraph()
    compound = 'DB01156'  # Bupropion
    disease = 'DOID:0050742'  # nicotine dependences
    metapath = graph.metagraph.metapath_from_abbrev('CbGpPWpGaD')
    rows, cols, pc_matrix = dwpc(graph, metapath, damping=0)
    rows, cols, dwpc_matrix = dwpc(graph, metapath, damping=0.4)
    i = rows.index(compound)
    j = cols.index(disease)
    assert pc_matrix[i, j] == 142
    assert dwpc_matrix[i, j] == pytest.approx(0.03287590886921623)


def test_CbGiGiGaD_traversal():
    """
    Test path counts and degree-weighted path counts for the CbGpPWpGaD
    metapath between bupropion and nicotine dependence. Expected values from
    the network traversal methods at https://git.io/vHBh2.
    """
    graph = get_bupropion_subgraph()
    compound = 'DB01156'  # Bupropion
    disease = 'DOID:0050742'  # nicotine dependences
    metapath = graph.metagraph.metapath_from_abbrev('CbGiGiGaD')
    paths = hetio.pathtools.paths_between(
        graph,
        source=('Compound', compound),
        target=('Disease', disease),
        metapath=metapath,
        duplicates=False,
    )
    hetio_dwpc = hetio.pathtools.DWPC(paths, damping_exponent=0.4)
    
    rows, cols, pc_matrix = dwpc(graph, metapath, damping=0)
    rows, cols, dwpc_matrix = dwpc(graph, metapath, damping=0.4)
    i = rows.index(compound)
    j = cols.index(disease)

    assert pc_matrix[i, j] == len(paths)
    assert dwpc_matrix[i, j] == pytest.approx(hetio_dwpc)

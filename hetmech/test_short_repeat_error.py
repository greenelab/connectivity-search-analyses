from .degree_weight import dwpc, _degree_weight, remove_diag, get_segments, \
    dwwc
from .matrix import metaedge_to_adjacency_matrix
import hetio.readwrite
import pytest
import numpy
from scipy import sparse


@pytest.mark.parametrize('metapath,cypher', [
    ('CpDrDdGdD', 0.012299)
])
def test_short_error(metapath, cypher):
    commit = '59c448fd912555f84b9822b4f49b431b696aea15'
    url = (f'https://github.com/dhimmel/hetionet/raw/{commit}/hetnet/json/'
           f'hetionet-v1.0.json.bz2')
    graph = hetio.readwrite.read_graph(url)
    compounds, diseases, dwpc_matrix, seconds = dwpc(
        graph, graph.metagraph.metapath_from_abbrev(metapath), damping=0.4,
        sparse_threshold=1)

    compound_index = compounds.index('DB00860')
    disease_index = diseases.index('DOID:3083')
    difference = dwpc_matrix[compound_index, disease_index] - cypher

    print(difference)
    assert difference == pytest.approx(0, abs=1e-6)

import hetio.readwrite
import pytest

from .degree_weight import dwpc


@pytest.mark.slow
@pytest.mark.parametrize('error_compound,error_disease,metapath,cypher', [
    ('DB00860', 'DOID:3083', 'CpDrDdGdD', 0.012299),
    ('DB00860', 'DOID:3083', 'CpDrDuGuD', 0.006209),
    ('DB00321', 'DOID:1826', 'CtDrDaGaD', 0.000522),
    ('DB00945', 'DOID:3393', 'CtDrDdGdD', 0.000000),
    ('DB00945', 'DOID:3393', 'CtDrDuGuD', 0.000000),
    ('DB00945', 'DOID:3393', 'CtDrDaGaD', 0.005130),
    ('DB00945', 'DOID:9008', 'CpDrDuGuD', 0.011211),
    ('DB00331', 'DOID:11612', 'CtDrDuGuD', 0.002222),
    ('DB00661', 'DOID:1826', 'CtDrDaGaD', 0.008234),
    ('DB00571', 'DOID:1826', 'CtDrDaGaD', 0.008234),
    ('DB00996', 'DOID:1826', 'CtDrDaGaD', 0.000395),
    ('DB00313', 'DOID:1826', 'CtDrDaGaD', 0.000395)
])
def test_short_error(error_compound, error_disease, metapath, cypher):
    commit = '59c448fd912555f84b9822b4f49b431b696aea15'

    url = (f'https://github.com/dhimmel/hetionet/raw/{commit}/hetnet/json/'
           f'hetionet-v1.0.json.bz2')
    graph = hetio.readwrite.read_graph(url)

    compounds, diseases, dwpc_matrix, seconds = dwpc(
        graph, graph.metagraph.metapath_from_abbrev(metapath), damping=0.4,
        sparse_threshold=1)

    compound_index = compounds.index(error_compound)
    disease_index = diseases.index(error_disease)
    difference = dwpc_matrix[compound_index, disease_index] - cypher

    assert difference == pytest.approx(0, abs=1e-6)

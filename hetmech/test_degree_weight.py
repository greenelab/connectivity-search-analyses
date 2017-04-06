import hetio.readwrite
import pytest

from .degree_weight import dwwc
from .matrix import get_node_to_position


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
    metapath = metagraph.metapath_from_abbrev('GiGaD')

    # Check that metapath does not contain duplicate metanodes,
    # i.e. the special case applies where DWWC is equivalent to DWPC.
    metanodes = metapath.get_nodes()
    assert len(metanodes) == len(set(metanodes))

    # Compute GiGaD path count and DWWC matrices
    pc_matrix = dwwc(graph, metapath, damping=0)
    dwwc_matrix = dwwc(graph, metapath, damping=0.5)

    # Check concordance with https://doi.org/10.1371/journal.pcbi.1004259.g002
    gene_index = get_node_to_position(graph, 'Gene')
    disease_index = get_node_to_position(graph, 'Disease')
    i = gene_index[graph.node_dict['Gene', 'IRF1']]
    j = disease_index[graph.node_dict['Disease', 'Multiple Sclerosis']]
    assert pc_matrix[i, j] == 3
    assert dwwc_matrix[i, j] == pytest.approx(0.25 + 0.25 + 32**-0.5)

import numpy

import hetmech.hetmat
import hetmech.matrix
from hetmech.tests.hetnets import get_graph


def test_disease_gene_example_converstion_to_hetmet(tmpdir):
    """
    Test the PC & DWWC computations in Figure 2D of Himmelstein & Baranzini
    (2015) PLOS Comp Bio. https://doi.org/10.1371/journal.pcbi.1004259.g002
    """
    graph = get_graph('disease-gene-example')
    hetmat = hetmech.hetmat.hetmat_from_graph(graph, tmpdir)
    assert list(graph.metagraph.get_nodes()) == list(hetmat.metagraph.get_nodes())

    # Test GaD adjacency matrix
    hetio_adj = hetmech.matrix.metaedge_to_adjacency_matrix(graph, 'GaD', dense_threshold=0)
    hetmat_adj = hetmech.matrix.metaedge_to_adjacency_matrix(hetmat, 'GaD', dense_threshold=0)
    assert hetio_adj[0] == hetmat_adj[0]  # row identifiers
    assert hetio_adj[1] == hetmat_adj[1]  # column identifiers
    assert numpy.array_equal(hetio_adj[2], hetmat_adj[2])  # adj matrices

    # Test DaG adjacency matrix (hetmat only stores GaD and must transpose)
    hetio_adj = hetmech.matrix.metaedge_to_adjacency_matrix(graph, 'DaG', dense_threshold=0)
    hetmat_adj = hetmech.matrix.metaedge_to_adjacency_matrix(hetmat, 'DaG', dense_threshold=0)
    assert hetio_adj[0] == hetmat_adj[0]  # row identifiers
    assert hetio_adj[1] == hetmat_adj[1]  # column identifiers
    assert numpy.array_equal(hetio_adj[2], hetmat_adj[2])  # adj matrices

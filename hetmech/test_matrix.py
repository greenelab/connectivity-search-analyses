import numpy
from scipy import sparse
import pytest
import hetio.readwrite
from collections import OrderedDict
import hetio.hetnet

from .matrix import get_node_to_position, metaedge_to_adjacency_matrix

def test_sparse_matrix():
    '''
    Test the functionality of metaedge_to_adjacency_matrix in 
    generating sparse matrices vs numpy arrays. Uses same test
    data as in test_degree_weight.py Figure 2D of Himmelstein & 
    Baranzini (2015) PLOS Comp Bio. 
    https://doi.org/10.1371/journal.pcbi.1004259.g002
    '''
    
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    
    def inefficient_metaedge_to_adjacency_matrix(graph, metaedge, dtype=numpy.bool_):
        """
        Returns a sparse matrix where source nodes are rows and target
        nodes are columns by converting a numpy array.
        """
        if not isinstance(metaedge, hetio.hetnet.MetaEdge):
            # metaedge is an abbreviation
            metaedge = graph.metagraph.metapath_from_abbrev(metaedge)[0]
        source_nodes = list(get_node_to_position(graph, metaedge.source))
        target_node_to_position = get_node_to_position(graph, metaedge.target)
        shape = len(source_nodes), len(target_node_to_position)
        adjacency_matrix = numpy.zeros(shape, dtype=dtype)
        for i, source_node in enumerate(source_nodes):
            for edge in source_node.edges[metaedge]:
                j = target_node_to_position[edge.target]
                adjacency_matrix[i, j] = 1
        adjacency_matrix = sparse.csr_matrix(adjacency_matrix) ## Inefficient but effective conversion
        row_names = [node.identifier for node in source_nodes]
        column_names = [node.identifier for node in target_node_to_position]
        return row_names, column_names, adjacency_matrix

    mat_ineff = inefficient_metaedge_to_adjacency_matrix(graph, 'GiG')
    mat_eff = metaedge_to_adjacency_matrix(graph, 'GiG')
    print("matrix.py should be giving a sparse array.")
    assert numpy.array_equal(mat_ineff[2].toarray(), mat_eff[2].toarray())
    
def test_matrix_not_sparse():
    with pytest.raises(AttributeError):
        print("Returns true if matrix.py is still giving numpy array")     
        test_sparse_matrix()

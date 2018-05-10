import pathlib
import hetio.hetnet
import hetio.readwrite
import hetio.matrix

def hetmat_from_graph(graph):
    """
    Create a hetmat.HetMat from a hetio.hetnet.Graph.
    """
    assert isinstance(graph, hetio.hetnet.Graph)
    metagraph = graph.metagraph
    metaedges = list(metagraph.get_edges(exclude_inverts=True))
    for metaedge in metaedges:
        matrix = hetio.matrix.metaedge_to_adjacency_matrix(metaedge)
    hetmat = HetMat(metagraph)


class HetMat:

    def __init__(self, directory, initialize=False, metagraph=None):
        """
        Initialize a HetMat with its MetaGraph.
        """
        self.directory = pathlib.Path(directory)
        self.metagraph_path = directory.joinpath('metagraph.json')
        self.nodes_directory = directory.joinpath('nodes')
        self.edges_directory = directory.joinpath('edges')
        if initialize:
            assert metagraph is not None
            self.metagraph = metagraph
            self.initialize()
        self.metagraph = hetio.readwrite.read_metagraph(self.metagraph_path)
        return self

    def initialize(self):
        """
        Initialize a directory.
        """
        # Create directories
        directories = [
            self.metagraph_path,
            self.nodes_directory,
            self.edges_directory,
        ]
        for directory in directories:
            if not directory.isdir():
                directory.mkdir()
        # Write metagraph
        hetio.readwrite.write_metagraph(self.metagraph_path)

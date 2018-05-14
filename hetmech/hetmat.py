import functools
import pathlib

import hetio.hetnet
import hetio.readwrite
import hetio.matrix


def hetmat_from_graph(graph, path):
    """
    Create a hetmat.HetMat from a hetio.hetnet.Graph.
    """
    assert isinstance(graph, hetio.hetnet.Graph)
    hetmat = HetMat(path, initialize=True)
    hetmat.metagraph = graph.metagraph
    metagraph = hetmat.metagraph
    metaedges = list(metagraph.get_edges(exclude_inverts=True))
    for metaedge in metaedges:
        matrix = hetio.matrix.metaedge_to_adjacency_matrix(metaedge)
    return hetmat


class HetMat:

    def __init__(self, directory, initialize=False):
        """
        Initialize a HetMat with its MetaGraph.
        """
        self.directory = pathlib.Path(directory)
        self.metagraph_path = directory.joinpath('metagraph.json')
        self.nodes_directory = directory.joinpath('nodes')
        self.edges_directory = directory.joinpath('edges')
        if initialize:
            self.initialize()
        return self

    def initialize(self):
        """
        Initialize the directory structure. This function is intended to be
        called when creating new HetMat instance on disk.
        """
        # Create directories
        directories = [
            self.directory,
            self.metagraph_path,
            self.nodes_directory,
            self.edges_directory,
        ]
        for directory in directories:
            if not directory.isdir():
                directory.mkdir()

    @property
    @functools.lru_cache()
    def metagraph(self):
        """
        HetMat.metagraph is a cached property. Hence reading the metagraph from
        disk should only occur once, the first time the metagraph property is
        accessed. See https://stackoverflow.com/a/19979379/4651668. If this
        method has issues, consider using cached_property from
        https://github.com/pydanny/cached-property.
        """
        return hetio.readwrite.read_metagraph(self.metagraph_path)

    @metagraph.setter
    def metagraph(self, metagraph):
        """
        Set the metagraph property by writing the metagraph to disk.
        """
        hetio.readwrite.write_metagraph(metagraph, self.metagraph_path)

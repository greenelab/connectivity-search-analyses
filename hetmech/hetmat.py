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
    # metaedges = list(metagraph.get_edges(exclude_inverts=True))
    # for metaedge in metaedges:
    #     matrix = hetio.matrix.metaedge_to_adjacency_matrix(metaedge)
    return hetmat


class HetMat:

    nodes_formats = {
        'tsv',
        'feather',
        'pickle',
        'json',
    }

    edges_formats = {
        'npy',
        'sparse.npz',
        'tsv',
    }

    def __init__(self, directory, initialize=False):
        """
        Initialize a HetMat with its MetaGraph.
        """
        self.directory = pathlib.Path(directory)
        self.metagraph_path = self.directory.joinpath('metagraph.json')
        self.nodes_directory = self.directory.joinpath('nodes')
        self.edges_directory = self.directory.joinpath('edges')
        if initialize:
            self.initialize()

    def initialize(self):
        """
        Initialize the directory structure. This function is intended to be
        called when creating new HetMat instance on disk.
        """
        # Create directories
        directories = [
            self.directory,
            self.nodes_directory,
            self.edges_directory,
        ]
        for directory in directories:
            if not directory.is_dir():
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

    def get_nodes_path(self, metanode, file_format='tsv'):
        """
        Potential file_formats are TSV, feather, JSON, and pickle.
        """
        return self.nodes_directory.joinpath(f'{metanode}.{file_format}')

    def get_edges_path(self, metaedge, file_format='npy'):
        """
        Get path to edges file
        """
        if isinstance(metaedge, hetio.hetnet.MetaEdge):
            metaedge = metaedge.get_abbrev()
        else:
            # Ensure that metaedge is a valid abbreviation
            _metaedge, = self.metagraph.metapath_from_abbrev(metaedge)
            assert _metaedge.get_abbrev() == metaedge
        return self.nodes_directory.joinpath(f'{metaedge}.{file_format}')

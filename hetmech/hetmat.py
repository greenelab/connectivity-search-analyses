import functools
import itertools
import pathlib
import shutil

import hetio.hetnet
import hetio.matrix
import hetio.permute
import hetio.readwrite
import numpy
import pandas
import scipy.sparse

import hetmech.degree_weight


def hetmat_from_graph(graph, path, save_metagraph=True, save_nodes=True, save_edges=True):
    """
    Create a hetmat.HetMat from a hetio.hetnet.Graph.
    """
    assert isinstance(graph, hetio.hetnet.Graph)
    hetmat = HetMat(path, initialize=True)
    hetmat.metagraph = graph.metagraph

    # Save metanodes
    metanodes = list(graph.metagraph.get_nodes())
    for metanode in metanodes:
        path = hetmat.get_nodes_path(metanode)
        rows = list()
        node_to_position = hetio.matrix.get_node_to_position(graph, metanode)
        for node, position in node_to_position.items():
            rows.append((position, node.identifier, node.name))
        node_df = pandas.DataFrame(rows, columns=['position', 'identifier', 'name'])
        path = hetmat.get_nodes_path(metanode)
        node_df.to_csv(path, index=False, sep='\t')

    # Save metaedges
    metaedges = list(graph.metagraph.get_edges(exclude_inverts=True))
    for metaedge in metaedges:
        rows, cols, matrix = hetio.matrix.metaedge_to_adjacency_matrix(graph, metaedge, dense_threshold=1)
        path = hetmat.get_edges_path(metaedge, file_format='sparse.npz')
        scipy.sparse.save_npz(str(path), matrix, compressed=True)
    return hetmat


def hetmat_from_permuted_graph(hetmat, permutation_id, permuted_graph):
    """
    Assumes subdirectory structure and that permutations inherit nodes but not
    edges.
    """
    if not hetmat.permutations_directory.is_dir():
        hetmat.permutations_directory.mkdir()
    directory = hetmat.permutations_directory.joinpath(f'{permutation_id}.hetmat')
    if directory.is_dir():
        # If directory exists, back it up using a .bak extension
        backup_directory = directory.with_name(directory.name + '.bak')
        if backup_directory.is_dir():
            shutil.rmtree(backup_directory)
        shutil.move(directory, backup_directory)
    permuted_hetmat = HetMat(directory, initialize=True)
    permuted_hetmat.is_permutation = True
    permuted_hetmat.metagraph_path.symlink_to('../../metagraph.json')
    permuted_hetmat.nodes_directory.rmdir()
    permuted_hetmat.nodes_directory.symlink_to('../../nodes', target_is_directory=True)
    permuted_hetmat = hetmat_from_graph(
        permuted_graph, directory, save_metagraph=False, save_nodes=False)
    return permuted_hetmat


def read_matrix(path, file_format='infer'):
    path = str(path)
    if file_format == 'infer':
        if path.endswith('.sparse.npz'):
            file_format = 'sparse.npz'
        if path.endswith('.npy'):
            file_format = 'npy'
    if file_format == 'infer':
        raise ValueError('Could not infer file_format for {path}')
    if file_format == 'sparse.npz':
        # https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.sparse.load_npz.html
        return scipy.sparse.load_npz(path)
    if file_format == 'npy':
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.load.html
        return numpy.load(path)
    raise ValueError(f'file_format={file_format} is not supported.')


def read_first_matrix(specs):
    """
    Attempt to read each path provided by specs, until one exists. If none of
    the specs point to an existing path, raise a FileNotFoundError.
    specs should be a list where each element is a dictionary specifying a
    potential path from which to read a matrix. Currently, the spec dictionary
    supports the following keys:
    - path: path to the file
    - transpose: whether to tranpose the file after reading it. If omitted,
      then False.
    - file_format: format of the matrix. If omitted, then infer.
    """
    paths = list()
    for spec in specs:
        path = pathlib.Path(spec['path'])
        paths.append(str(path))
        if not path.is_file():
            continue
        transpose = spec.get('transpose', False)
        file_format = spec.get('file_format', 'infer')
        matrix = read_matrix(path, file_format=file_format)
        if transpose:
            matrix = matrix.transpose()
        return matrix
    raise FileNotFoundError(
        f'No matrix files found at the specified paths:\n' +
        '\n'.join(paths))


def _permute_matrix(adjacency_matrix, directed=False, multiplier=10,
                    excluded_pair_set=set(), seed=0, log=False):

    edge_list = list(zip(*adjacency_matrix.nonzero()))
    permuted_edges, logs = hetio.permute.permute_pair_list(
        edge_list, directed=directed, multiplier=multiplier,
        excluded_pair_set=excluded_pair_set, seed=seed, log=log)

    edges = numpy.array(permuted_edges)
    ones = numpy.ones(len(edges), dtype=adjacency_matrix.dtype)
    permuted_adjacency = scipy.sparse.csc_matrix((ones, (edges[:, 0], edges[:, 1])),
                                                 shape=adjacency_matrix.shape)

    # Keep the same sparse type as adjacency_matrix
    permuted_adjacency = type(adjacency_matrix)(permuted_adjacency)

    # Ensure node degrees have been preserved
    assert all(permuted_adjacency.sum(axis=1) == adjacency_matrix.sum(axis=1))
    assert all(permuted_adjacency.sum(axis=0) == adjacency_matrix.sum(axis=0))

    return permuted_adjacency, logs


class HetMat:

    # Supported formats for nodes files
    nodes_formats = {
        'tsv',
        # 'feather',
        # 'pickle',
        # 'json',
    }

    # Supported formats for edges files
    edges_formats = {
        'npy',
        'sparse.npz',
        # 'tsv',
    }

    def __init__(self, directory, initialize=False):
        """
        Initialize a HetMat with its MetaGraph.
        """
        self.directory = pathlib.Path(directory)
        self.metagraph_path = self.directory.joinpath('metagraph.json')
        self.nodes_directory = self.directory.joinpath('nodes')
        self.edges_directory = self.directory.joinpath('edges')
        self.path_counts_directory = self.directory.joinpath('path-counts')
        # Permutations should set is_permutation=True
        self.is_permutation = False
        self.permutations_directory = self.directory.joinpath('permutations')
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
    def permutations(self):
        """
        Return a dictionary of permutation name to permutation directory.
        Assumes permutation name is the directory name minus its .hetmat
        extension.
        """
        permutations = {}
        for directory in sorted(self.permutations_directory.glob('*.hetmat')):
            if not directory.is_dir():
                continue
            permutation = HetMat(directory)
            permutation.is_permutation = True
            name, _ = directory.name.rsplit('.', 1)
            permutations[name] = permutation
        return permutations

    def permute_graph(self, num_new_permutations=None, namer=None, start_from=None,
                      multiplier=10, excluded_pair_set=set(), seed=0, log=True):
        """
        Generate and save permutations of the HetMat adjacency matrices.

        Parameters
        ----------
        num_new_permutations : int
            The number of new, permuted HetMats to generate
        namer : generator
            Yields the names of new permutations. Cannot pass names of existing permutations
        start_from : str
            Name of permutation to use as starting point. For multiple permutations,
            the first permutation starts from start_from, and future permutations
            continue from the previous one.
        multiplier : int
            How many attempts to make when cross-swapping edges.
        excluded_pair_set : set
            Pairs of nodes which are not to be changed
        seed : int
            Random seed for generating new permutations
        log : bool
            When log=True, give information about permutation attempts
        """
        if namer is None:
            # If no namer given, continue increasing names by one for new permutations
            def namer():
                i = 1
                while True:
                    yield str(i).zfill(3)
                    i += 1
            namer = namer()

        logging_stats = list()
        for _ in range(num_new_permutations):
            permutation_name = next(namer)
            name = f'{permutation_name}.hetmat/'
            new_permutation_path = self.permutations_directory.joinpath(name)

            new_hetmat = HetMat(new_permutation_path, initialize=True)
            new_hetmat.is_permutation = True
            new_hetmat.metagraph_path.symlink_to('../../metagraph.json')
            new_hetmat.nodes_directory.rmdir()
            new_hetmat.nodes_directory.symlink_to('../../nodes', target_is_directory=True)

            if start_from is not None:
                original_hetmat = self.permutations[start_from]
            else:
                original_hetmat = self

            metaedges = list(self.metagraph.get_edges(exclude_inverts=True))
            for metaedge in metaedges:
                rows, cols, original_matrix = original_hetmat.metaedge_to_adjacency_matrix(
                    metaedge, dense_threshold=1)
                is_directed = metaedge.direction != 'both'
                permuted_matrix, stats = _permute_matrix(
                    original_matrix, directed=is_directed, multiplier=multiplier,
                    excluded_pair_set=excluded_pair_set, seed=seed, log=log)
                path = new_hetmat.get_edges_path(metaedge, file_format='sparse.npz')
                scipy.sparse.save_npz(str(path), permuted_matrix, compressed=True)
                logging_stats.append(stats)
            start_from = permutation_name
        if log:
            return logging_stats

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
        Get the path for the nodes file for the specified metanode. Setting
        file_format=None returns the path without any exension suffix.
        """
        metanode = self.metagraph.get_metanode(metanode)
        path = self.nodes_directory.joinpath(f'{metanode}')
        if file_format is not None:
            path = path.with_name(f'{path.name}.{file_format}')
        return path

    def get_edges_path(self, metaedge, file_format='npy'):
        """
        Get the path for the edges file for the specified metaedge. Setting
        file_format=None returns the path without any exension suffix.
        """
        metaedge_abbrev = self.metagraph.get_metaedge(metaedge).get_abbrev()
        path = self.edges_directory.joinpath(f'{metaedge_abbrev}')
        if file_format is not None:
            path = path.with_name(f'{path.name}.{file_format}')
        return path

    def get_path_counts_path(self, metapath, metric, damping, file_format):
        """
        Setting file_format=None returns the path without any exension suffix.
        Supported metrics are 'dwpc' and 'dwwc'.
        """
        path = self.path_counts_directory.joinpath(f'{metric}-{damping}/{metapath}')
        if file_format is not None:
            path = path.with_name(f'{path.name}.{file_format}')
        return path

    @functools.lru_cache()
    def get_node_identifiers(self, metanode):
        """
        Returns a list of node identifiers for a metapath
        """
        path = self.get_nodes_path(metanode, file_format='tsv')
        node_df = pandas.read_table(path)
        return list(node_df['identifier'])

    def metaedge_to_adjacency_matrix(
            self, metaedge,
            dtype=None, dense_threshold=None,
            file_formats=['sparse.npz', 'npy']):
        """
        file_formats sets the precedence of which file to read in
        """
        metaedge = self.metagraph.get_metaedge(metaedge)
        specs = list()
        configurations = itertools.product(file_formats, (True, False))
        for file_format, invert in configurations:
            path = self.get_edges_path(
                metaedge=metaedge.inverse if invert else metaedge,
                file_format=file_format,
            )
            spec = {'path': path, 'transpose': invert, 'file_format': file_format}
            specs.append(spec)
        matrix = read_first_matrix(specs)
        if dense_threshold is not None:
            matrix = hetio.matrix.sparsify_or_densify(matrix, dense_threshold=dense_threshold)
        if dtype is not None:
            matrix = matrix.astype(dtype)
        row_ids = self.get_node_identifiers(metaedge.source)
        col_ids = self.get_node_identifiers(metaedge.target)
        return row_ids, col_ids, matrix

    def read_path_counts(
            self, metapath, metric, damping,
            file_formats=['sparse.npz', 'npy']):
        """
        Read matrix with values of a path-count-based metric. Attempts to
        locate any files with the matrix (or with trivial transformations).
        """
        category = hetmech.degree_weight.categorize(metapath)
        metrics = [metric]
        if metric == 'dwpc' and category == 'no_repeats':
            metrics.append('dwwc')
        if metric == 'dwwc' and category == 'no_repeats':
            metrics.append('dwpc')
        specs = list()
        configurations = itertools.product(
            file_formats,
            metrics,
            (True, False),
        )
        for file_format, metric, invert in configurations:
            path = self.get_path_counts_path(
                metapath=metapath.inverse if invert else metapath, metric=metric,
                damping=damping,
                file_format=file_format,
            )
            spec = {'path': path, 'transpose': invert, 'file_format': file_format}
            specs.append(spec)
        row_ids = self.get_node_identifiers(metapath.source())
        col_ids = self.get_node_identifiers(metapath.target())
        matrix = read_first_matrix(specs)
        return row_ids, col_ids, matrix

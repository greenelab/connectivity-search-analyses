import os
import pathlib
import re
import shutil
import tempfile
import urllib.request
import zipfile


def create_hetmat_archive(hetmat, destination_path=None):
    """
    Create the minimal archive to store a hetmat (i.e. metagraph, nodes, edges).
    If destination_path is None (default), use the same name as the hetmat
    directory with .zip appended.
    """
    if destination_path is None:
        destination_path = hetmat.directory.joinpath('..', hetmat.directory.absolute().name + '.zip')
    create_archive_by_globs(
        destination_path=destination_path,
        root_directory=hetmat.directory,
        include_globs=['nodes/*', 'edges/*'],
        include_paths=['metagraph.json'],
        zip_mode='w',
    )


def create_archive_by_globs(
        destination_path, root_directory,
        include_globs=[], exclude_globs=[], include_paths=[],
        **kwargs):
    """
    First, paths relative to root_directory are included according to include_globs.
    Second, paths relative to root_directory are excluded according to exclude_globs.
    Finally, paths relative to root_directory are included from include_paths.
    """
    root_directory = pathlib.Path(root_directory)
    source_paths = set()
    for glob in include_globs:
        source_paths |= set(root_directory.glob(glob))
    for glob in exclude_globs:
        source_paths -= set(root_directory.glob(glob))
    source_paths = [path.relative_to(root_directory) for path in source_paths]
    source_paths.extend(map(pathlib.Path, include_paths))
    create_archive(destination_path, root_directory, source_paths, **kwargs)


def create_archive(destination_path, root_directory, source_paths, zip_mode='x', compression=zipfile.ZIP_LZMA):
    """
    Create a zip archive of the source paths at the destination path.
    source_paths as paths relative to the hetmat root directory.
    """
    root_directory = pathlib.Path(root_directory)
    destination_path = pathlib.Path(destination_path)
    assert zip_mode in {'w', 'x', 'a'}
    zip_file = zipfile.ZipFile(destination_path, mode=zip_mode, compression=compression)
    source_paths = sorted(set(map(str, source_paths)))
    for source_path in source_paths:
        source_fs_path = root_directory.joinpath(source_path)
        with source_fs_path.open('rb') as read_file, zip_file.open(source_path, mode='w') as write_file:
            shutil.copyfileobj(read_file, write_file)
    zip_file.close()


def load_archive(archive_path, destination_dir):
    """
    Extract the paths from an archive into the specified hetmat directory.
    """
    is_url = isinstance(archive_path, str) and re.match('^(http|ftp)s?://', archive_path)
    if is_url:
        url = archive_path
        filename = os.path.basename(url)
        temp_dir = pathlib.Path(tempfile.mkdtemp())
        archive_path, _ = urllib.request.urlretrieve(url, temp_dir.joinpath(filename))
    with zipfile.ZipFile(archive_path, mode='r') as zip_file:
        zip_file.extractall(destination_dir)
    if is_url:
        shutil.rmtree(temp_dir)

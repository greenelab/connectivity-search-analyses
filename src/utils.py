from pathlib import Path
from configparser import ConfigParser

repo_dir: Path = Path(__file__).parent.parent
"""root directory of this repository"""

def get_config() -> ConfigParser:
    path = repo_dir.joinpath("config.ini")
    config = ConfigParser()
    config.read(path, encoding="utf-8")
    return config


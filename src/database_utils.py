import psycopg2

from .utils import get_config

def get_db_connection() -> psycopg2.extensions.connection:
    params = get_config()["postgresql"]
    return psycopg2.connect(**params)

"""Shared pytest fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evalguard.db.store import get_connection, init_db


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "eval.db"


@pytest.fixture()
def conn(db_path: Path) -> sqlite3.Connection:
    connection = get_connection(db_path)
    init_db(connection)
    yield connection
    connection.close()

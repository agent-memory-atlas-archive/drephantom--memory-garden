"""共享 fixtures：合成 Vault + 临时派生库 + 已同步检索器。"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_garden.cli import build_retriever
from memory_garden.config import Settings
from memory_garden.db import Database
from memory_garden.importer import VaultSyncService

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_VAULT = REPO_ROOT / "evals" / "cognitive_mvp_vault"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(vault_path=SYNTHETIC_VAULT, database_path=tmp_path / "test.db")


@pytest.fixture()
def database(settings: Settings) -> Database:
    database = Database(settings.database_path)
    database.initialize()
    VaultSyncService(database, settings.vault_path).sync()
    return database


@pytest.fixture()
def retriever(settings: Settings, database: Database):
    hybrid = build_retriever(database, settings)
    hybrid.vector.ensure_vectors()
    return hybrid

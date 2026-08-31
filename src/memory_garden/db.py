"""SQLite schema: read-only source provenance + derived cognition (runs, verdicts, discoveries).

设计边界：Obsidian Vault 永远只读；本库只保存派生数据（索引、检索、轨迹、用户判定）。
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 3

# v1 → v2：discoveries 增加变化分类学/信号类型/呈现追踪列（存量库增量迁移）
_V2_DISCOVERY_COLUMNS = {
    "change_type": "TEXT NOT NULL DEFAULT 'true_change'",
    "signal_type": "TEXT NOT NULL DEFAULT 'stance_pair'",
    "change_confidence": "REAL NOT NULL DEFAULT 0.5",
    "shown_count": "INTEGER NOT NULL DEFAULT 0",
    "last_shown_at": "TEXT",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ── 来源层（provenance）───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    uid TEXT UNIQUE NOT NULL,            -- 稳定身份：内容不变的移动/重命名保持 uid
    rel_path TEXT NOT NULL,
    title TEXT NOT NULL,
    frontmatter_json TEXT NOT NULL DEFAULT '{}',
    tags_json TEXT NOT NULL DEFAULT '[]',
    links_json TEXT NOT NULL DEFAULT '[]',
    recorded_at TEXT,                    -- 记录时刻（frontmatter created / 文件创建时间，低置信）
    event_time TEXT,                     -- 事件时刻（frontmatter explicit date，仅显式声明）
    modified_at TEXT,
    authorship TEXT NOT NULL DEFAULT 'user',  -- user | quoted | ai_generated | derived
    content_hash TEXT NOT NULL,
    is_present INTEGER NOT NULL DEFAULT 1,
    searchable INTEGER NOT NULL DEFAULT 1,
    body TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sources_recorded ON sources(recorded_at);

CREATE TABLE IF NOT EXISTS source_revisions (
    id INTEGER PRIMARY KEY,
    source_uid TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE (source_uid, content_hash)
);

CREATE TABLE IF NOT EXISTS source_atoms (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    uid TEXT UNIQUE NOT NULL,
    seq INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    recorded_at TEXT,
    event_time TEXT,
    authorship TEXT NOT NULL DEFAULT 'user'
);
CREATE INDEX IF NOT EXISTS idx_atoms_source ON source_atoms(source_id, seq);

CREATE VIRTUAL TABLE IF NOT EXISTS source_atoms_fts USING fts5(
    text,
    tokenize='trigram'
);

CREATE TABLE IF NOT EXISTS atom_vectors (
    atom_id INTEGER NOT NULL REFERENCES source_atoms(id) ON DELETE CASCADE,
    embedding_provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding_text_version TEXT NOT NULL,
    embedding_text_hash TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (atom_id, embedding_provider, embedding_model, embedding_text_version)
);

CREATE TABLE IF NOT EXISTS sync_runs (
    id INTEGER PRIMARY KEY,
    vault_hash TEXT NOT NULL,
    files_seen INTEGER NOT NULL,
    files_added INTEGER NOT NULL,
    files_changed INTEGER NOT NULL,
    files_removed INTEGER NOT NULL,
    atoms_total INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

-- ── 对话与审计层 ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    role TEXT NOT NULL,                  -- user | assistant
    content TEXT NOT NULL,
    answer_json TEXT,                    -- CognitiveAnswer（assistant 消息附带）
    evidence_json TEXT,                  -- 引用来源快照
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY,
    message_id INTEGER REFERENCES messages(id),
    backend TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    steps INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT NOT NULL DEFAULT '',
    error TEXT,
    private_vault_sent INTEGER NOT NULL DEFAULT 0,
    trace_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

-- ── 用户认知层（唯一可写入"已确认认知"的地方是用户本人）───────────────
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY,
    message_id INTEGER,
    topic_key TEXT NOT NULL,
    verdict TEXT NOT NULL,               -- accurate|partly_accurate|no_change|not_my_view|insufficient_evidence|defer
    user_revision TEXT,
    confirmed_interpretation TEXT,
    missing_event TEXT,
    accepted_atom_ids_json TEXT NOT NULL DEFAULT '[]',
    rejected_atom_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verdicts_topic ON verdicts(topic_key);

CREATE TABLE IF NOT EXISTS discovery_scans (
    id INTEGER PRIMARY KEY,
    candidates_found INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discoveries (
    id INTEGER PRIMARY KEY,
    scan_id INTEGER REFERENCES discovery_scans(id),
    topic_key TEXT NOT NULL,
    early_atom_id INTEGER NOT NULL REFERENCES source_atoms(id),
    recent_atom_id INTEGER NOT NULL REFERENCES source_atoms(id),
    early_date TEXT,
    recent_date TEXT,
    early_excerpt TEXT NOT NULL,
    recent_excerpt TEXT NOT NULL,
    diff_terms_json TEXT NOT NULL DEFAULT '[]',
    score REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'candidate',   -- candidate | reviewed
    review_verdict TEXT,
    review_revision TEXT,
    change_type TEXT NOT NULL DEFAULT 'true_change',  -- true_change|parallel_stance|wording_drift|deepening|contextual_stance
    signal_type TEXT NOT NULL DEFAULT 'stance_pair',  -- stance_pair|explicit_contrast
    change_confidence REAL NOT NULL DEFAULT 0.5,
    shown_count INTEGER NOT NULL DEFAULT 0,
    last_shown_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_discoveries_topic ON discoveries(topic_key);

-- ── 立场快照层（离线抽取，发现的查询对象）──────────────────────────────
CREATE TABLE IF NOT EXISTS stance_snapshots (
    id INTEGER PRIMARY KEY,
    atom_id INTEGER NOT NULL UNIQUE REFERENCES source_atoms(id) ON DELETE CASCADE,
    snapshot_schema_version INTEGER NOT NULL,
    topic TEXT NOT NULL,
    stance TEXT NOT NULL DEFAULT '',
    has_stance INTEGER NOT NULL DEFAULT 1,
    quote TEXT NOT NULL DEFAULT '',
    tone_strength REAL NOT NULL DEFAULT 0.5,   -- 0=犹疑 1=笃定
    is_own_view INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 0.5,
    extractor TEXT NOT NULL DEFAULT 'deterministic',
    extracted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_topic ON stance_snapshots(topic);

CREATE TABLE IF NOT EXISTS candidate_reactions (
    id INTEGER PRIMARY KEY,
    discovery_id INTEGER NOT NULL REFERENCES discoveries(id) ON DELETE CASCADE,
    reaction TEXT NOT NULL,            -- accurate | wrong | boring
    created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Database:
    """Thread-local connections on a shared SQLite file; WAL for concurrent reads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.conn = connection
        return connection

    def close(self) -> None:
        connection: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if connection is not None:
            connection.close()
            self._local.conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        with connection:
            yield connection

    def initialize(self) -> None:
        connection = self.connect()
        with connection:
            connection.executescript(SCHEMA)
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
            current = int(row["value"]) if row else 0
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"数据库 schema v{current} 高于当前程序支持的 v{SCHEMA_VERSION}，请升级程序"
                )
            if current < 2:
                self._migrate_v1_to_v2(connection)
            if current < 3:
                self._migrate_v2_to_v3(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_atom_vectors_identity "
                "ON atom_vectors(embedding_provider, embedding_model, "
                "embedding_text_version, embedding_dimension)"
            )
            connection.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        existing = {
            info[1] for info in connection.execute("PRAGMA table_info(discoveries)").fetchall()
        }
        for column, definition in _V2_DISCOVERY_COLUMNS.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE discoveries ADD COLUMN {column} {definition}")

    @staticmethod
    def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
        """把单一 atom_id 缓存迁移为带完整 embedding 身份的可并存缓存。

        v2 无法证明旧向量使用的 provider/model/文本版本，因此原样保留向量内容，
        但标为 legacy；当前检索身份永远不会读取这些行，并会按需重建新缓存。
        """
        existing = {
            str(info[1]) for info in connection.execute("PRAGMA table_info(atom_vectors)").fetchall()
        }
        if "embedding_provider" in existing:
            return
        now = utc_now()
        connection.execute("DROP TABLE IF EXISTS atom_vectors_v3")
        connection.execute(
            """
            CREATE TABLE atom_vectors_v3 (
                atom_id INTEGER NOT NULL REFERENCES source_atoms(id) ON DELETE CASCADE,
                embedding_provider TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                embedding_text_version TEXT NOT NULL,
                embedding_text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    atom_id, embedding_provider, embedding_model, embedding_text_version
                )
            )
            """
        )
        if {"atom_id", "dim", "vector_json"} <= existing:
            connection.execute(
                """
                INSERT INTO atom_vectors_v3(
                    atom_id, embedding_provider, embedding_model, embedding_dimension,
                    embedding_text_version, embedding_text_hash, vector_json,
                    created_at, updated_at
                )
                SELECT atom_id, 'legacy', 'unknown-v2', dim, 'legacy', '', vector_json, ?, ?
                FROM atom_vectors
                """,
                (now, now),
            )
        connection.execute("DROP TABLE atom_vectors")
        connection.execute("ALTER TABLE atom_vectors_v3 RENAME TO atom_vectors")

    def fetchone(self, sql: str, params: Sequence[object] = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        return self.connect().execute(sql, params).fetchall()

    def execute(self, sql: str, params: Sequence[object] = ()) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(sql, params)
            return int(cursor.lastrowid or 0)

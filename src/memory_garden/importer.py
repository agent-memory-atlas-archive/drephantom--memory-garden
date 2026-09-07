"""只读 Vault 导入：解析 Markdown → 来源/原子，保留 provenance 与时间分离。

关键不变量：
- Vault 仅以只读方式访问，这里只调用 open/read，从不写回；
- ``event_time`` 只来自显式声明（frontmatter 或消息时间戳），
  文件系统修改时间至多是低置信的记录时刻回退，绝不冒充事件时间；
- 内容哈希不变的身份（uid）在移动/重命名后保持稳定，引用才可复现。
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .db import Database, utc_now

SKIP_DIRS = {".obsidian", ".trash", ".git", ".venv", "node_modules"}
IMPORTER_VERSION = 'v2-complete-chunks'

_DATE_RE = re.compile(r"\b(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})")
_WECHAT_HEADER_RE = re.compile(r"^###\s+(20\d{2}-\d{2}-\d{2})\s+(\d{2}:\d{2})")
_QUOTE_PREFIX_RE = re.compile(r"^(&gt;|>|\[引用\])")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_frontmatter(text: str) -> tuple[dict[str, Any], str, int]:
    """极简 YAML 子集：平铺 key: value 与一维列表。避免引入完整 YAML 依赖。"""
    if not text.startswith("---"):
        return {}, text, 0
    lines = text.splitlines()
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        return {}, text, 0
    meta: dict[str, Any] = {}
    current_key: str | None = None
    for line in lines[1:end]:
        if not line.strip():
            continue
        if line.startswith(("  - ", "- ")) and current_key:
            value = line.strip().lstrip("- ").strip().strip('"').strip("'")
            items = meta.setdefault(current_key, [])
            if isinstance(items, list):
                items.append(value)
            continue
        if ":" in line:
            key, _, raw = line.partition(":")
            key = key.strip()
            raw = raw.strip()
            if raw.startswith("[") and raw.endswith("]"):
                inner = raw[1:-1]
                meta[key] = [v.strip().strip('"') for v in inner.split(",") if v.strip()]
                current_key = None
            elif raw:
                meta[key] = raw.strip('"').strip("'")
                current_key = key
            else:
                meta[key] = []
                current_key = key
    body_offset = sum(len(line) + 1 for line in lines[: end + 1])
    return meta, "\n".join(lines[end + 1 :]), body_offset


def _as_iso_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})", value)
    if match:
        return "-".join(match.groups())
    return None


def infer_event_time(frontmatter: dict[str, Any], body: str) -> str | None:
    """事件时间只接受显式声明；正文里的普通日期数字不自动采信。"""
    for key in ("event", "event_date", "event_time", "date", "happened"):
        iso = _as_iso_date(frontmatter.get(key))
        if iso:
            return iso
    return None


def infer_recorded_at(
    frontmatter: dict[str, Any], file_stat: os.stat_result, body: str
) -> str | None:
    for key in ("created", "recorded", "created_at"):
        iso = _as_iso_date(frontmatter.get(key))
        if iso:
            return iso
    # 低置信回退：文件创建时间只能说明"这段内容最早何时出现在磁盘上"。
    return datetime.fromtimestamp(file_stat.st_ctime, tz=UTC).strftime("%Y-%m-%d")


def infer_authorship(
    frontmatter: dict[str, Any], title: str, body: str, filename: str = ""
) -> str:
    declared = str(frontmatter.get("authorship") or "")
    if declared in {"user", "quoted", "ai_generated", "derived"}:
        return declared
    origin = str(frontmatter.get("content_origin") or "")
    if origin in {"quote", "quoted"} or "generated_by" in frontmatter:
        return "quoted" if origin in {"quote", "quoted"} else "ai_generated"
    if str(frontmatter.get("type") or "").startswith("ai"):
        return "ai_generated"
    name = filename or title
    if name.startswith("AI草稿") or title.startswith("AI草稿"):
        return "ai_generated"
    if name.startswith("引用") or title.startswith("引用") or "[引用]" in body[:200]:
        return "quoted"
    return "user"


def extract_tags(frontmatter: dict[str, Any]) -> list[str]:
    raw = frontmatter.get("tags", [])
    if isinstance(raw, str):
        raw = [raw]
    return [str(tag).strip() for tag in raw if str(tag).strip()][:32]


def extract_links(body: str) -> list[str]:
    wiki = re.findall(r"\[\[([^\]|#]+)", body)
    markdown = re.findall(r"\[[^\]]*\]\(([^)]+\.md)\)", body)
    return list(dict.fromkeys([link.strip() for link in wiki + markdown]))[:64]


@dataclass
class ParsedAtom:
    text: str
    heading: str = ""
    line_start: int = 1
    line_end: int = 1
    event_time: str | None = None
    authorship: str = "user"


@dataclass
class ParsedDocument:
    rel_path: str
    title: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    recorded_at: str | None = None
    event_time: str | None = None
    authorship: str = "user"
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    atoms: list[ParsedAtom] = field(default_factory=list)


def _looks_like_blockquote(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and all(_QUOTE_PREFIX_RE.match(line.strip()) for line in lines)


def _chunk_markdown(body: str, doc_event_time: str | None, authorship: str) -> list[ParsedAtom]:
    """按标题切分为原子；引用行整段归为 quoted。"""
    atoms: list[ParsedAtom] = []
    heading = ""
    buffer: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        # 长章节按行累积切分；长单行继续分段，正文尾部不能被静默截掉。
        chunk: list[str] = []
        first = last = start_line
        size = 0
        def emit() -> None:
            text = '\n'.join(chunk).strip()
            if text:
                atoms.append(ParsedAtom(
                    text=text, heading=heading, line_start=first, line_end=last,
                    event_time=doc_event_time,
                    authorship='quoted' if _looks_like_blockquote(text) else authorship,
                ))
        for line_no, line in enumerate(buffer, start=start_line):
            for offset in range(0, max(1, len(line)), 2000):
                part = line[offset:offset+2000]
                if chunk and size+len(part)+1 > 2000:
                    emit()
                    chunk, size = [], 0
                if not chunk:
                    first = line_no
                chunk.append(part)
                size += len(part)+1
                last = line_no
        emit()

    for index, line in enumerate(body.splitlines(), start=1):
        if line.startswith("#"):
            flush(index - 1)
            heading = line.lstrip("#").strip()
            buffer = [line]
            start_line = index
        else:
            buffer.append(line)
    flush(len(body.splitlines()))
    return atoms


def parse_wechat_document(rel_path: str, text: str) -> ParsedDocument | None:
    """微信导出格式：``### 2025-07-01 15:34`` 逐条消息，天然带事件时间戳。"""
    header = text.splitlines()[0] if text.splitlines() else ""
    if not _WECHAT_HEADER_RE.match(header.strip()) and "wechat" not in rel_path.lower():
        return None
    doc = ParsedDocument(
        rel_path=rel_path,
        title=rel_path.rsplit("/", 1)[-1].removesuffix(".md"),
        authorship="user",
    )
    lines = text.splitlines()
    current_time: str | None = None
    buffer: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        body = "\n".join(buffer).strip()
        if not body:
            return
        quoted = body.startswith("[引用]")
        atoms_authorship = "quoted" if quoted else doc.authorship
        for atom in _chunk_markdown('\n'.join(buffer), current_time, atoms_authorship):
            atom.line_start += start_line - 1
            atom.line_end += start_line - 1
            doc.atoms.append(atom)

    for index, line in enumerate(lines, start=1):
        match = _WECHAT_HEADER_RE.match(line.strip())
        if match:
            flush(index - 1)
            current_time = f"{match.group(1)}T{match.group(2)}:00"
            buffer = []
            start_line = index + 1
        elif line.startswith("# "):
            doc.title = line.lstrip("# ").strip()
        else:
            buffer.append(line)
    flush(len(lines))
    return doc if doc.atoms else None


def classify_markdown(relative_path: str) -> str | None:
    """返回作者类别提示（按文件名约定），不可搜索的文件返回 None。"""
    name = relative_path.rsplit("/", 1)[-1]
    if name.startswith("_") or name.startswith("未命名"):
        return "skip"
    lowered = relative_path.lower()
    if "/templates/" in lowered or lowered.startswith("templates/"):
        return "skip"
    return None


def iter_markdown_paths(vault_path: Path) -> list[tuple[Path, str]]:
    results: list[tuple[Path, str]] = []
    for path in sorted(vault_path.rglob("*.md")):
        parts = path.relative_to(vault_path).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        rel = "/".join(parts)
        if classify_markdown(rel) == "skip":
            continue
        results.append((path, rel))
    return results


def parse_markdown_document(path: Path, relative_path: str, text: str) -> ParsedDocument:
    normalized = text.replace('\r\n', '\n')
    frontmatter, body, body_offset = split_frontmatter(normalized)
    title = str(frontmatter.get("title") or "") or relative_path.rsplit("/", 1)[-1].removesuffix(".md")
    doc = ParsedDocument(
        rel_path=relative_path,
        title=title,
        frontmatter=frontmatter,
        recorded_at=infer_recorded_at(frontmatter, path.stat(), body),
        event_time=infer_event_time(frontmatter, body),
        authorship=infer_authorship(frontmatter, title, body, relative_path.rsplit("/", 1)[-1]),
        tags=extract_tags(frontmatter),
        links=extract_links(body),
    )
    wechat = parse_wechat_document(relative_path, text)
    if wechat is not None and len(wechat.atoms) >= 3:
        doc.atoms = wechat.atoms
        doc.title = wechat.title
    else:
        doc.atoms = _chunk_markdown(body, doc.event_time, doc.authorship)
        if not doc.atoms and body.strip():
            doc.atoms = [
                ParsedAtom(text=body.strip()[:4000], line_start=1, line_end=1,
                           event_time=doc.event_time, authorship=doc.authorship)
            ]
        line_offset = normalized[:body_offset].count('\n')
        for atom in doc.atoms:
            atom.line_start += line_offset
            atom.line_end += line_offset
    return doc


def vault_markdown_hash(vault_path: Path) -> str:
    hasher = hashlib.sha256()
    for path, rel in iter_markdown_paths(vault_path):
        hasher.update(rel.encode("utf-8"))
        hasher.update(sha256_text(path.read_text(encoding="utf-8", errors="replace")).encode())
    return hasher.hexdigest()


class VaultSyncService:
    """幂等同步：内容哈希不变则跳过；变更生成不可变修订；消失文件标记 is_present=0。"""

    def __init__(self, database: Database, vault_path: Path):
        self.database = database
        self.vault_path = vault_path

    def sync(self) -> dict[str, Any]:
        if not self.vault_path.is_dir():
            raise ValueError('Vault 笔记库路径不存在或不是目录，请先检查数据设置。')
        root = str(self.vault_path.resolve())
        bound = self.database.fetchone("SELECT value FROM schema_meta WHERE key='vault_root'")
        if bound and os.path.normcase(str(bound['value'])) != os.path.normcase(root):
            raise ValueError('此数据库已绑定另一个 Vault；请为新笔记库使用独立数据库。')
        self.database.execute(
            "INSERT OR IGNORE INTO schema_meta(key, value) VALUES('vault_root', ?)", (root,)
        )
        vault_hash = vault_markdown_hash(self.vault_path)
        parser = self.database.fetchone("SELECT value FROM schema_meta WHERE key='importer_version'")
        refresh = not parser or parser['value'] != IMPORTER_VERSION
        last = self.database.fetchone(
            "SELECT vault_hash FROM sync_runs ORDER BY id DESC LIMIT 1"
        )
        counts = {
            "files_seen": 0, "files_added": 0, "files_changed": 0,
            "files_removed": 0, "atoms_total": 0,
        }
        if last and last["vault_hash"] == vault_hash and not refresh:
            row = self.database.fetchone("SELECT COUNT(*) AS n FROM source_atoms WHERE is_current=1")
            return {"changed": False, "vault_hash": vault_hash, **counts,
                    "atoms_total": int(row["n"]) if row else 0}

        present_uids: set[str] = set()
        claimed_uids: set[str] = set()
        for path, rel in iter_markdown_paths(self.vault_path):
            counts["files_seen"] += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            doc = parse_markdown_document(path, rel, text)
            doc_hash = sha256_text(text)
            doc_uid = self._resolve_uid(rel, doc_hash, claimed_uids)
            claimed_uids.add(doc_uid)
            present_uids.add(doc_uid)
            is_new = self._is_new(doc_uid)
            with self.database.transaction() as connection:
                self._upsert_document(connection, doc, doc_uid, doc_hash, refresh=refresh)
            counts["files_added" if is_new else "files_changed"] += 1
            counts["atoms_total"] += len(doc.atoms)

        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO schema_meta(key,value) VALUES('importer_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (IMPORTER_VERSION,),
            )
            for row in connection.execute(
                "SELECT uid FROM sources WHERE is_present=1"
            ).fetchall():
                if row["uid"] not in present_uids:
                    connection.execute(
                        "UPDATE sources SET is_present=0 WHERE uid=?", (row["uid"],)
                    )
                    counts["files_removed"] += 1
            connection.execute(
                "INSERT INTO sync_runs(vault_hash, files_seen, files_added, files_changed,"
                " files_removed, atoms_total, created_at) VALUES(?,?,?,?,?,?,?)",
                (vault_hash, counts["files_seen"], counts["files_added"],
                 counts["files_changed"], counts["files_removed"], counts["atoms_total"], utc_now()),
            )
        return {"changed": True, "vault_hash": vault_hash, **counts}

    def _resolve_uid(self, rel_path: str, content_hash: str, claimed: set[str]) -> str:
        row = self.database.fetchone("SELECT uid FROM sources WHERE rel_path=?", (rel_path,))
        if row:
            return str(row["uid"])
        # 内容相同、路径不同、且未被本批次其他文件认领 → 视为移动/重命名，保留身份
        candidates = self.database.fetchall(
            "SELECT uid, rel_path FROM sources WHERE content_hash=?", (content_hash,)
        )
        for candidate in candidates:
            if candidate["rel_path"] != rel_path and str(candidate["uid"]) not in claimed:
                return str(candidate["uid"])
        return "src_" + sha256_text(rel_path)[:20]

    def _is_new(self, uid: str) -> bool:
        return self.database.fetchone(
            "SELECT 1 FROM source_revisions WHERE source_uid=?", (uid,)
        ) is None

    def _upsert_document(
        self, connection: sqlite3.Connection, doc: ParsedDocument, uid: str, content_hash: str,
        *, refresh: bool = False,
    ) -> None:
        existing = connection.execute(
            "SELECT id, content_hash FROM sources WHERE uid=?", (uid,)
        ).fetchone()
        now = utc_now()
        connection.execute(
            "INSERT OR IGNORE INTO source_revisions(source_uid, content_hash, first_seen_at,"
            " last_seen_at) VALUES(?,?,?,?)",
            (uid, content_hash, now, now),
        )
        connection.execute(
            "UPDATE source_revisions SET last_seen_at=? WHERE source_uid=? AND content_hash=?",
            (now, uid, content_hash),
        )
        if existing and existing["content_hash"] == content_hash and not refresh:
            connection.execute(
                "UPDATE sources SET is_present=1, rel_path=? WHERE id=?",
                (doc.rel_path, existing["id"]),
            )
            return
        frontmatter_json = _dumps(doc.frontmatter)
        if existing:
            connection.execute(
                """
                UPDATE sources SET rel_path=?, title=?, frontmatter_json=?, tags_json=?,
                    links_json=?, recorded_at=?, event_time=?, modified_at=?, authorship=?,
                    content_hash=?, is_present=1 WHERE id=?
                """,
                (doc.rel_path, doc.title, frontmatter_json, _dumps(doc.tags), _dumps(doc.links),
                 doc.recorded_at, doc.event_time, now, doc.authorship, content_hash,
                 existing["id"]),
            )
            source_id = int(existing["id"])
            # 历史原子保留供既有发现、判定和引用追溯；仅从当前检索中撤下。
            connection.execute(
                "DELETE FROM source_atoms_fts WHERE rowid IN"
                " (SELECT id FROM source_atoms WHERE source_id=?)",
                (source_id,),
            )
            connection.execute("UPDATE source_atoms SET is_current=0 WHERE source_id=?", (source_id,))
        else:
            cursor = connection.execute(
                """
                INSERT INTO sources(uid, rel_path, title, frontmatter_json, tags_json, links_json,
                    recorded_at, event_time, modified_at, authorship, content_hash, is_present,
                    searchable, body)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,1,1,'')
                """,
                (uid, doc.rel_path, doc.title, frontmatter_json, _dumps(doc.tags),
                 _dumps(doc.links), doc.recorded_at, doc.event_time, now, doc.authorship,
                 content_hash),
            )
            source_id = int(cursor.lastrowid or 0)
        self._insert_atoms(connection, source_id, doc, content_hash)

    def _insert_atoms(
        self, connection: sqlite3.Connection, source_id: int, doc: ParsedDocument, revision_hash: str
    ) -> None:
        tags_text = " ".join(doc.tags)
        for seq, atom in enumerate(doc.atoms):
            atom_uid = "atom_" + sha256_text(f"{source_id}:{revision_hash}:{IMPORTER_VERSION}:{seq}")[:24]
            connection.execute(
                """
                INSERT INTO source_atoms(source_id, uid, seq, heading, text, line_start,
                    line_end, recorded_at, event_time, authorship, revision_hash)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(uid) DO UPDATE SET is_current=1
                """,
                (source_id, atom_uid, seq, atom.heading, atom.text, atom.line_start,
                 atom.line_end, doc.recorded_at, atom.event_time, atom.authorship, revision_hash),
            )
            # 检索文本 = 标题 + 标题层级 + 标签 + 正文；atom.text 本身保持原文
            search_text = f"{doc.title}\n{atom.heading}\n{tags_text}\n{atom.text}"
            connection.execute(
                "INSERT INTO source_atoms_fts(rowid, text) VALUES(?, ?)",
                (int(connection.execute('SELECT id FROM source_atoms WHERE uid=?', (atom_uid,)).fetchone()[0]), search_text),
            )


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)

"""Read-only local and global note maps with source-explicit edges only.

The map reparses current atoms instead of trusting a cached link name. A link is
an authored reference, never evidence of a change of mind or of causality. No
embedding provider, model, Vault write, or hidden-source label is used here.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from .db import Database
from .retrieval import tokenize

MAX_NODES = 40
MAX_EDGES = 200
GLOBAL_MAX_NODES = 500
GLOBAL_MAX_EDGES = 3000
MAX_LINKS_PER_SOURCE = 256
_LINK_RE = re.compile(
    r"(?<!\\)\[\[(?P<wiki>[^\]\n]+)\]\]"
    r'|(?<![\\!])\[(?P<label>[^\]\n]*)\]\((?P<markdown><[^>\n]+>|(?:\\.|[^()\n]|\([^()\n]*\))+?)\)'
)
_INLINE_CODE_RE = re.compile(r"(`+).*?\1")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class _Link:
    raw: str
    target: str
    anchor: str
    alias: str
    kind: str
    offset: int


def _links(text: str, state: dict[str, Any]) -> list[_Link]:
    """Parse inline/wiki note links, excluding code and HTML comments.

    State survives atom boundaries so a code block cannot create false edges.
    Reference-style Markdown links are deliberately left unresolved: the atom
    index does not preserve every definition across chunk/revision boundaries.
    """
    result: list[_Link] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        fence = _FENCE_RE.match(line)
        if fence and not state.get("comment"):
            marker = fence.group(1)
            current = state.get("fence", "")
            if not current:
                state["fence"] = marker
            elif marker[0] == current[0] and len(marker) >= len(current):
                state["fence"] = ""
            offset += len(line)
            continue
        if state.get("fence"):
            offset += len(line)
            continue
        masked = list(line)
        cursor = 0
        while cursor < len(line):
            if state.get("comment"):
                end = line.find("-->", cursor)
                stop = len(line) if end < 0 else end + 3
                masked[cursor:stop] = " " * (stop - cursor)
                cursor = stop
                state["comment"] = end < 0
            else:
                start = line.find("<!--", cursor)
                if start < 0:
                    break
                cursor = start
                state["comment"] = True
        visible = "".join(masked)
        visible = _INLINE_CODE_RE.sub(lambda m: " " * len(m.group()), visible)
        for match in _LINK_RE.finditer(visible):
            if match.group("wiki") is not None:
                value, _, alias = match.group("wiki").partition("|")
                kind = "wiki"
            else:
                value = match.group("markdown").strip()
                if value.startswith("<"):
                    value = value[1:value.index(">")]
                else:
                    value = re.sub(r'\s+[\"\'].*$', "", value)
                alias = match.group("label") or ""
                kind = "markdown"
            value = re.sub(r"\\([ ()\[\]#])", r"\1", value).strip()
            try:
                parsed = urlsplit(value)
            except ValueError:
                # A malformed URL in a note must not prevent opening the map.
                continue
            # External links, absolute filesystem paths and attachments do not
            # belong to the note graph. Percent decoding happens after # split.
            if parsed.scheme or parsed.netloc:
                continue
            target = unquote(parsed.path).replace("\\", "/")
            extension = posixpath.splitext(target)[1].lower()
            if extension and extension != ".md":
                continue
            if kind == "markdown" and target and extension != ".md":
                continue
            result.append(_Link(
                raw=line[match.start():match.end()], target=target,
                anchor=unquote(parsed.fragment), alias=alias, kind=kind,
                offset=offset + match.start(),
            ))
        offset += len(line)
    return result


def _path_key(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/")).casefold()


def _resolve(
    link: _Link, origin: dict[str, Any],
    by_path: dict[str, list[str]], by_name: dict[str, list[str]],
    visible: dict[str, dict[str, Any]],
) -> str | None:
    target = link.target
    if not target:
        return str(origin["uid"])
    if not target.lower().endswith(".md"):
        target += ".md"
    if link.kind == "wiki" and "/" not in target:
        candidates = set(by_name.get(target.casefold(), []))
    else:
        parent = posixpath.dirname(str(origin["rel_path"]))
        if target.startswith("/"):
            paths = [target.lstrip("/")]
        elif link.kind == "markdown" or target.startswith(("./", "../")):
            paths = [posixpath.join(parent, target)]
        else:
            # Obsidian supports Vault-relative and relative note paths. When
            # both interpretations exist, expose no guessed relationship.
            paths = [target, posixpath.join(parent, target)]
        candidates = set()
        for path in paths:
            key = _path_key(path)
            if key == ".." or key.startswith("../"):
                continue
            candidates.update(by_path.get(key, []))
    if len(candidates) != 1:
        return None
    uid = next(iter(candidates))
    return uid if uid in visible else None


def _citation(note: dict[str, Any], atom: dict[str, Any], excerpt: str) -> dict[str, Any]:
    return {
        "atom_id": atom["id"], "source_uid": note["uid"], "title": note["title"],
        "path": note["rel_path"], "line_start": atom["line_start"],
        "line_end": atom["line_end"], "recorded_at": atom["recorded_at"],
        "event_time": atom["event_time"], "authorship": atom["authorship"],
        "excerpt": excerpt, "sender": atom.get("sender") or None,
        "source_kind": note.get("source_kind", "vault"),
    }


def _redact_unresolved(text: str, links: list[tuple[_Link, str | None]]) -> str:
    # An excluded target may be named inside an otherwise visible note. Do not
    # publish that label in graph previews or in another edge's context.
    for link, target in reversed(links):
        if target is None:
            text = text[:link.offset] + "[未解析链接]" + text[link.offset + len(link.raw):]
    return text


def build_map(
    database: Database, topic: str = "", source_uid: str | None = None,
    limit: int | None = None, *, scope: Literal["local", "global"] = "local", query: str = "",
) -> dict[str, Any]:
    """Return a stable, bounded map; both scopes are entirely local and read-only.

    Local scope preserves focus/topic plus one-hop neighbors. Global scope
    includes all eligible sources, including isolated and chat records, or
    only query matches. A focus from a prior local view cannot narrow global
    scope. Counts expose node/edge and per-source scan limits; they never imply
    an unlimited graph. The same link parser and visibility rules serve both.
    """
    if scope not in {"local", "global"}:
        raise ValueError("Unknown map scope")
    node_cap = GLOBAL_MAX_NODES if scope == "global" else MAX_NODES
    edge_cap = GLOBAL_MAX_EDGES if scope == "global" else MAX_EDGES
    limit = max(1, min(node_cap, int(limit) if limit is not None else node_cap))
    topic = topic.strip()[:300]
    query = query.strip()[:300]
    search = query if scope == "global" else topic
    # Global filtering is literal and predictable: splitting a filename such as
    # n44.md into tokens would make the shared '.md' suffix match every note.
    terms = ([search.casefold()] if scope == "global"
             else list(dict.fromkeys([search.casefold(), *tokenize(search)]))) if search else []
    rows = database.fetchall(
        "SELECT id,uid,title,rel_path,recorded_at,event_time,authorship,searchable,tags_json,source_kind "
        "FROM sources WHERE is_present=1 ORDER BY rel_path,uid"
    )
    by_path: dict[str, list[str]] = defaultdict(list)
    by_name: dict[str, list[str]] = defaultdict(list)
    visible: dict[str, dict[str, Any]] = {}
    scores: dict[str, int] = {}
    # Hidden paths participate only in ambiguity detection, never in output.
    for row in rows:
        note = dict(row)
        uid, path = str(note["uid"]), str(note["rel_path"])
        by_path[_path_key(path)].append(uid)
        by_name[posixpath.basename(path).casefold()].append(uid)
        if note["searchable"] and note["authorship"] != "derived":
            visible[uid] = note
            metadata = f"{note['title']} {path} {note['tags_json']}".casefold()
            scores[uid] = sum(4 for term in terms if term and term in metadata)

    source_uid = source_uid if scope == "local" else None
    focus = source_uid if source_uid in visible else None
    invalid_focus = bool(source_uid and not focus)
    states: dict[str, dict[str, Any]] = defaultdict(dict)
    previews: dict[str, list[dict[str, Any]]] = defaultdict(list)
    resolved: list[dict[str, Any]] = []
    unresolved: dict[str, int] = defaultdict(int)
    link_counts: dict[str, int] = defaultdict(int)
    scan_truncated: set[str] = set()
    date_ranges: dict[str, tuple[str, str]] = {}
    # Stream current atoms; retain only bounded link evidence and preview text.
    atoms = database.connect().execute(
        "SELECT a.* ,s.uid AS source_uid FROM source_atoms a JOIN sources s ON s.id=a.source_id "
        "WHERE s.is_present=1 AND s.searchable=1 AND s.authorship!='derived' "
        "AND a.is_current=1 AND a.authorship!='derived' "
        "ORDER BY s.rel_path,s.uid,a.seq,a.id"
    )
    for row in atoms:
        atom = dict(row)
        uid = str(atom["source_uid"])
        note = visible[uid]
        text = str(atom["text"])
        if note["source_kind"] == "chat" and atom["recorded_at"]:
            date = str(atom["recorded_at"])[:10]
            earlier, later = date_ranges.get(uid, (date, date))
            date_ranges[uid] = (min(earlier, date), max(later, date))
        # Chat text is not Markdown note structure; a message containing [[x]]
        # must not silently create an Obsidian-style relationship.
        parsed = _links(text, states[uid]) if note["source_kind"] != "chat" else []
        links = [(link, _resolve(link, note, by_path, by_name, visible)) for link in parsed]
        safe_text = _redact_unresolved(text, links)
        # Excluded link labels must not influence search results either.
        atom_score = sum(1 for term in terms if term and term in safe_text.casefold())
        scores[uid] += atom_score
        preview = _citation(note, atom, safe_text[:800])
        preview["_score"] = atom_score
        previews[uid].append(preview)
        previews[uid] = sorted(previews[uid], key=lambda c: (-c["_score"], c["atom_id"]))[:3]
        for link, target in links:
            if target is None:
                unresolved[uid] += 1
                continue
            if target == uid:
                continue
            link_counts[uid] += 1
            if link_counts[uid] > MAX_LINKS_PER_SOURCE:
                scan_truncated.add(uid)
                continue
            identity = f"{uid}:{target}:{atom['uid']}:{link.offset}"
            evidence = _citation(note, atom, safe_text[:800])
            evidence.update({
                "quote": link.raw, "link_target": link.target,
                "anchor": link.anchor, "alias": link.alias,
            })
            resolved.append({
                "id": "link_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "source": uid, "target": target, "type": "explicit_link",
                "status": "source_explicit", "evidence": evidence,
            })

    def recent_key(uid: str) -> tuple[str, str, str]:
        note = visible[uid]
        return (str(note["recorded_at"] or ""), str(note["rel_path"]), uid)

    if invalid_focus:
        seeds: list[str] = []
    elif focus:
        seeds = [focus]
    elif search:
        seeds = sorted((uid for uid in visible if scores[uid] > 0),
                       key=lambda uid: (-scores[uid], str(visible[uid]["rel_path"]), uid))
    else:
        seeds = sorted(visible, key=recent_key, reverse=True)
    seed_set = set(seeds)
    neighbors: set[str] = set()
    for edge in resolved:
        if edge["source"] in seed_set:
            neighbors.add(edge["target"])
        if edge["target"] in seed_set:
            neighbors.add(edge["source"])
    candidates = list(seeds)
    if scope == "local":
        candidates.extend(sorted(neighbors - seed_set, key=lambda uid: (visible[uid]["rel_path"], uid)))
    candidate_set = set(candidates)
    scoped_edges = [edge for edge in resolved
                    if edge["source"] in candidate_set and edge["target"] in candidate_set]
    selected = candidates[:limit]
    selected_set = set(selected)
    eligible_edges = [edge for edge in scoped_edges
                      if edge["source"] in selected_set and edge["target"] in selected_set]
    eligible_edges.sort(key=lambda edge: (edge["source"], edge["target"], edge["id"]))
    edges = eligible_edges[:edge_cap]
    nodes = []
    for uid in selected:
        note = visible[uid]
        citations = [{key: value for key, value in citation.items() if key != "_score"}
                     for citation in previews[uid]]
        try:
            tags = json.loads(note["tags_json"])
        except (json.JSONDecodeError, TypeError):
            tags = []
        nodes.append({
            "id": uid, "uid": uid, "title": note["title"], "path": note["rel_path"],
            "recorded_at": note["recorded_at"], "event_time": note["event_time"],
            "authorship": note["authorship"], "citations": citations,
            "source_kind": note["source_kind"],
            "date_range": {"start": date_ranges[uid][0], "end": date_ranges[uid][1]} if uid in date_ranges else None,
            "matched_topic": bool(search and scores[uid] > 0),
            "tags": tags if isinstance(tags, list) else [],
        })
    omitted_nodes = max(0, len(candidates) - len(nodes))
    omitted_edges = max(0, len(scoped_edges) - len(edges))
    link_scan_truncated = bool(scan_truncated & candidate_set)
    return {
        "scope": scope, "query": query, "topic": topic,
        "focus_source_uid": focus, "nodes": nodes, "edges": edges,
        "meta": {
            "node_limit": limit, "edge_limit": edge_cap,
            "total_visible": len(visible), "total_matching": len(seeds),
            "total_candidates": len(candidates), "total_edges": len(scoped_edges),
            "returned_nodes": len(nodes), "returned_edges": len(edges),
            "omitted_nodes": omitted_nodes, "omitted_edges": omitted_edges,
            "unresolved_links": sum(unresolved[uid] for uid in selected),
            "link_scan_truncated": link_scan_truncated,
            "links_per_source_limit": MAX_LINKS_PER_SOURCE,
            "truncated": bool(omitted_nodes or omitted_edges or link_scan_truncated),
            "notice": "连线仅表示原文明确写出的笔记链接，不代表观点变化或因果。记录时间与事件时间分开显示；记录时间可能来自文件创建时间。",
        },
    }


def build_local_map(
    database: Database, topic: str = "", source_uid: str | None = None, limit: int = MAX_NODES,
) -> dict[str, Any]:
    """Compatibility entry point for the existing local graph callers."""
    return build_map(database, topic=topic, source_uid=source_uid, limit=limit)

"""Chat exports stay local, preserve provenance and never infer the user's identity."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from memory_garden.chat_import import MAX_BYTES, ChatImportService, parse_chat_export
from memory_garden.cognitive import VerdictService
from memory_garden.config import Settings
from memory_garden.db import Database
from memory_garden.graph import build_local_map
from memory_garden.importer import VaultSyncService, vault_markdown_hash
from memory_garden.memory import MemoryService
from memory_garden.retrieval import BM25Retriever, RetrievalQuery


@pytest.fixture()
def chat_service(tmp_path: Path) -> ChatImportService:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "判断.md").write_text("自主判断的笔记。", encoding="utf-8")
    database = Database(tmp_path / "derived" / "chat.db")
    database.initialize()
    VaultSyncService(database, vault).sync()
    service = ChatImportService(database, Settings(vault_path=vault, database_path=database.path))
    yield service
    database.close()


def test_qq_sender_timestamp_and_multiline_provenance() -> None:
    text = "消息记录导出\n2024-1-2 9:01:02 我(12345)\n我自己作判断。\n也会听意见。\n2024-1-2 9:02:00 对方(98765)\n你必须听我的。\n"
    format, messages, warnings = parse_chat_export(text)
    assert format == "qq"
    assert len(messages) == 2
    assert messages[0].sender == "我(12345)"
    assert messages[0].timestamp == "2024-01-02T09:01:02"
    assert messages[0].text == "我自己作判断。\n也会听意见。"
    assert (messages[0].line_start, messages[0].line_end) == (2, 4)
    assert any("1 行" in warning for warning in warnings)


def test_wechat_inline_name_first_and_unknown_legacy_speaker() -> None:
    text = "[2024-01-01 12:00:00] 小李: 我会先形成自己的判断。\n小王 2024-01-01 12:01:00\n我不这样看。\n"
    _, messages, _ = parse_chat_export(text, "wechat")
    assert [(m.sender, m.text) for m in messages] == [("小李", "我会先形成自己的判断。"), ("小王", "我不这样看。")]
    _, old, warnings = parse_chat_export("# 会话名称\n### 2024-01-01 12:00\n原始记录\n", "wechat")
    assert old[0].sender == "未知发言人"
    assert any("没有明确发言人" in warning for warning in warnings)


def test_json_preserves_explicit_timezone_missing_time_and_original_lines() -> None:
    content = '[\n {"sender":"我","timestamp":"2024-01-01T12:00:00+08:00","text":"引用内容"},\n {"sender":"对方","text":"[图片]"}\n]'
    format, messages, warnings = parse_chat_export(content)
    assert format == "json"
    assert messages[0].timestamp == "2024-01-01T12:00:00+08:00"
    assert messages[1].timestamp is None
    assert (messages[0].line_start, messages[0].line_end) == (2, 2)
    assert any("没有时间" in warning for warning in warnings)
    assert any("占位符" in warning for warning in warnings)


def test_preview_is_ephemeral_and_own_sender_requires_explicit_selection(chat_service: ChatImportService) -> None:
    before = chat_service.database.connect().total_changes
    preview = chat_service.preview(filename="chat.txt", content="2024-01-01 12:00:00 我\n自主判断\n2024-01-01 12:01:00 对方\n你应该听我的。")
    assert preview["counts"] == {"messages": 2, "own": 0, "quoted": 2, "participants": 2}
    assert all(message["authorship"] == "quoted" for message in preview["messages"])
    assert chat_service.database.connect().total_changes == before
    assert not (chat_service.settings.database_path.parent / "imports").exists()
    with pytest.raises(ValueError, match="确认"):
        chat_service.commit(preview["preview_id"], ["我"])
    assert not (chat_service.settings.database_path.parent / "imports").exists()
    with pytest.raises(ValueError, match="明确发言人"):
        chat_service.commit(preview["preview_id"], ["不存在的人"], confirm_agent_access=True)


def test_commit_indexes_full_text_with_own_identity_and_deduplicates(chat_service: ChatImportService) -> None:
    long_text = "自主判断" * 1500
    content = json.dumps([{"sender": "我", "timestamp": "2024-01-01 12:00:00", "text": long_text},
                          {"sender": "别人", "timestamp": "2024-01-01 12:01:00", "text": "你必须依赖我。"}], ensure_ascii=False)
    before = vault_markdown_hash(chat_service.settings.vault_path)
    preview = chat_service.preview(filename="../../chat.json", content=content)
    result = chat_service.commit(preview["preview_id"], ["我"], confirm_agent_access=True)
    assert result["indexed_messages"] == 2 and result["indexed_atoms"] == 4
    atoms = chat_service.database.fetchall("SELECT * FROM source_atoms WHERE source_id=? ORDER BY seq", (result["source_id"],))
    assert "".join(atom["text"] for atom in atoms if atom["sender"] == "我") == long_text
    assert {atom["authorship"] for atom in atoms if atom["sender"] == "我"} == {"user"}
    assert {atom["authorship"] for atom in atoms if atom["sender"] == "别人"} == {"quoted"}
    artifact = Path(result["artifact_path"])
    assert artifact.is_relative_to(chat_service.settings.database_path.parent / "imports")
    assert artifact.read_text(encoding="utf-8") == content
    assert vault_markdown_hash(chat_service.settings.vault_path) == before
    again = chat_service.commit(preview["preview_id"], ["我"], confirm_agent_access=True)
    assert again["duplicate"] is True and again["source_uid"] == result["source_uid"]
    assert chat_service.database.fetchone("SELECT COUNT(*) FROM sources WHERE source_kind='chat'")[0] == 1
    assert chat_service.database.fetchone("SELECT COUNT(*) FROM atom_vectors")[0] == 0
    assert BM25Retriever(chat_service.database).search(RetrievalQuery("自主判断", authorship="user"))


def test_import_survives_vault_resync_and_can_be_deactivated(chat_service: ChatImportService) -> None:
    preview = chat_service.preview(filename="chat.txt", content="2024-01-01 12:00:00 我\n聊天专属检索词。")
    result = chat_service.commit(preview["preview_id"], ["我"], confirm_agent_access=True)
    (chat_service.settings.vault_path / "new.md").write_text("新的笔记触发同步。", encoding="utf-8")
    VaultSyncService(chat_service.database, chat_service.settings.vault_path).sync()
    row = chat_service.database.fetchone("SELECT * FROM sources WHERE uid=?", (result["source_uid"],))
    assert row["is_present"] == 1 and row["source_kind"] == "chat"
    chat_service.deactivate(result["source_uid"])
    assert BM25Retriever(chat_service.database).search(RetrievalQuery("聊天专属检索词")) == []
    restored = chat_service.commit(preview["preview_id"], ["我"], confirm_agent_access=True)
    assert restored["duplicate"] is True
    assert BM25Retriever(chat_service.database).search(RetrievalQuery("聊天专属检索词"))


def test_chat_text_does_not_create_markdown_note_edges(chat_service: ChatImportService) -> None:
    preview = chat_service.preview(filename="chat.txt", content="2024-01-01 12:00:00 我\n自主判断 [[判断]]")
    result = chat_service.commit(preview["preview_id"], ["我"], confirm_agent_access=True)
    graph = build_local_map(chat_service.database, source_uid=result["source_uid"])
    assert graph["edges"] == []
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["source_kind"] == "chat"
    assert graph["nodes"][0]["citations"][0]["sender"] == "我"
    assert graph["nodes"][0]["date_range"] == {"start": "2024-01-01", "end": "2024-01-01"}


@pytest.mark.parametrize("content", ["x\x00y", "x" * (MAX_BYTES + 1), "[" * 10 + "]" * 10,
    '[{"sender":{},"text":"bad"}]', '[{"sender":"我","timestamp":"2024-99-99","text":"bad"}]',
    '[{"sender":"我","text":"ok"},]', "没有任何可识别时间与发言人的段落"],
    ids=["binary", "oversized", "deep_json", "invalid_sender", "invalid_date", "trailing_comma", "unstructured"])
def test_malformed_or_unsupported_content_is_rejected(content: str) -> None:
    with pytest.raises(ValueError):
        parse_chat_export(content)


def test_expired_preview_and_unsupported_files_are_not_imported(chat_service: ChatImportService, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="暂不支持"):
        chat_service.preview(filename="chat.zip", content="fake archive")
    preview = chat_service.preview(filename="chat.txt", content="2024-01-01 12:00:00 我\n一句话")
    original_time = time.monotonic()
    monkeypatch.setattr("memory_garden.chat_import.time.monotonic", lambda: original_time + 1000)
    with pytest.raises(KeyError, match="过期"):
        chat_service.commit(preview["preview_id"], ["我"], confirm_agent_access=True)
    assert chat_service.database.fetchone("SELECT COUNT(*) FROM sources WHERE source_kind='chat'")[0] == 0


def test_record_limit_counts_even_empty_json_messages() -> None:
    with pytest.raises(ValueError, match="2000"):
        parse_chat_export(json.dumps([{"sender": "我", "text": ""}] * 2001), "json")


@pytest.mark.parametrize("escaped", [r'\u0000', r'\ud800'], ids=["decoded_nul", "unpaired_surrogate"])
def test_json_decoded_invalid_text_is_rejected_before_persistence(escaped: str) -> None:
    with pytest.raises(ValueError):
        parse_chat_export('[{"sender":"me","text":"' + escaped + '"}]', "json")


def _qce_fixture() -> dict:
    """Synthetic values against QCE CleanMessage, not copied exporter output."""
    return {"metadata": {"name": "synthetic-export"}, "chatInfo": {"selfUid": "u_self"}, "statistics": {},
            "messages": [
                {"id": "m_one", "timestamp": 1704067200000, "time": "2024-01-01 08:00:00",
                 "sender": {"uid": "u_self", "uin": "11111", "name": "同名"}, "type": "type_1",
                 "content": {"text": "我会先自己判断。", "elements": [], "resources": []}},
                {"id": "m_two", "timestamp": 1704067201000,
                 "sender": {"uid": "u_other", "uin": "22222", "name": "同名"}, "type": "type_1",
                 "content": {"text": "你应该听我的。", "html": "<img src='https://invalid.example/never-fetch'>", "elements": [], "resources": []}},
                {"id": "m_three", "timestamp": 1704067202000,
                 "sender": {"uid": "u_self", "name": "后来改的名字"}, "type": "type_1",
                 "content": {"text": "别人原先说的话。", "elements": [{"type": "forward", "data": {}}], "resources": []}},
            ]}


def test_qce_single_json_distinguishes_same_names_and_uses_millisecond_time(chat_service: ChatImportService) -> None:
    content = json.dumps(_qce_fixture(), ensure_ascii=False, indent=2)
    preview = chat_service.preview(filename="qce.json", content=content, format="qq")
    assert preview["format"] == "qq_json"
    assert preview["counts"]["own"] == 0  # selfUid never authorizes identity
    assert preview["participants"] == ["同名（u_other）", "同名（u_self）"]
    assert preview["messages"][0]["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert preview["messages"][2]["sender"] == "同名（u_self）"  # stable ID despite rename
    assert preview["messages"][2]["force_quoted"] is True
    result = chat_service.commit(preview["preview_id"], ["同名（u_self）"], confirm_agent_access=True)
    assert result["own_messages"] == 1 and result["quoted_messages"] == 2
    atoms = chat_service.database.fetchall("SELECT * FROM source_atoms WHERE source_id=? ORDER BY seq", (result["source_id"],))
    assert [atom["authorship"] for atom in atoms] == ["user", "quoted", "quoted"]
    assert "never-fetch" not in "".join(atom["text"] for atom in atoms)
    lines = content.splitlines()
    original_range = "\n".join(lines[atoms[0]["line_start"] - 1:atoms[0]["line_end"]])
    assert '"m_one"' in original_range and '"m_two"' not in original_range


def test_chatlab_standard_uses_seconds_and_preserves_reply_ids(chat_service: ChatImportService) -> None:
    fixture = {"chatlab": {"version": "0.0.2", "exportedAt": 1704067200},
               "meta": {"name": "合成对话", "platform": "qq", "type": "private", "ownerId": "p1"},
               "members": [{"platformId": "p1", "accountName": "本人"}, {"platformId": "p2", "accountName": "朋友"}],
               "messages": [
                   {"sender": "p1", "accountName": "本人", "timestamp": 1704067200, "type": 0,
                    "content": "我还在考虑。", "platformMessageId": "one"},
                   {"sender": "p2", "accountName": "朋友", "timestamp": 1704067201, "type": 25,
                    "content": "慢慢来。", "platformMessageId": "two", "replyToMessageId": "one"},
                   {"sender": "p1", "accountName": "本人", "timestamp": 1704067202, "type": 80,
                    "content": "群资料已更新。"},
                   {"sender": "p2", "accountName": "朋友", "timestamp": 1704067203, "type": 1, "content": None},
               ]}
    preview = chat_service.preview(filename="chatlab.json", content=json.dumps(fixture, ensure_ascii=False))
    assert preview["format"] == "chatlab"
    assert preview["counts"]["own"] == 0  # ownerId is a hint, not a confirmed identity
    assert preview["messages"][0]["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert preview["messages"][3]["text"] == "[图片]"
    result = chat_service.commit(preview["preview_id"], ["本人（p1）"], confirm_agent_access=True)
    assert result["own_messages"] == 1 and result["quoted_messages"] == 3
    source = chat_service.database.fetchone("SELECT frontmatter_json FROM sources WHERE id=?", (result["source_id"],))
    provenance = json.loads(source["frontmatter_json"])["message_provenance"]
    assert provenance[1]["platform_message_id"] == "two"
    assert provenance[1]["reply_to_message_id"] == "one"


@pytest.mark.parametrize("fixture", [
    {"chatlab": {"version": "future"}, "messages": []},
    {"metadata": {}, "chatInfo": {}, "chunked": {"chunks": ["unsafe.jsonl"]}},
    {"messages": [{"sender": "me", "text": "unknown envelope"}]},
], ids=["unknown_chatlab_version", "qce_chunked_manifest", "unknown_envelope"])
def test_unknown_exporter_versions_and_chunk_manifests_are_explicitly_rejected(fixture: dict) -> None:
    with pytest.raises(ValueError):
        parse_chat_export(json.dumps(fixture))


def test_duplicate_json_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        parse_chat_export('[{"sender":"me","sender":"other","text":"ambiguous"}]', "json")


def test_corrected_own_identity_replaces_retrieval_and_blocks_old_memory(chat_service: ChatImportService) -> None:
    content = json.dumps([{"sender": "甲", "timestamp": "2024-01-01 12:00:00", "text": "自主判断依赖他人。"},
                          {"sender": "乙", "timestamp": "2024-01-01 12:01:00", "text": "自主判断由我决定。"}], ensure_ascii=False)
    preview = chat_service.preview(filename="identity.json", content=content)
    first = chat_service.commit(preview["preview_id"], ["甲"], confirm_agent_access=True)
    database = chat_service.database
    old_atom = database.fetchone("SELECT id FROM source_atoms WHERE source_id=? AND authorship='user'", (first["source_id"],))
    thread_id = database.execute("INSERT INTO threads(title,created_at) VALUES('身份修正测试','2024-01-01')")
    database.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,'user','自主判断','2024-01-01')", (thread_id,))
    answer = {"topic": "自主判断", "answer_type": "no_clear_change", "summary": "这条解释需要核对。", "citations": [{"atom_id": old_atom["id"]}]}
    message_id = database.execute(
        "INSERT INTO messages(thread_id,role,content,answer_json,created_at) VALUES(?,'assistant','回看',?,'2024-01-01')",
        (thread_id, json.dumps(answer, ensure_ascii=False)),
    )
    VerdictService(database).save_verdict(message_id=message_id, verdict="accurate")
    assert MemoryService(database).context_for_topic("自主判断")

    corrected = chat_service.commit(preview["preview_id"], ["乙"], confirm_agent_access=True)
    assert corrected["superseded_imports"] == 1
    assert corrected["source_uid"] != first["source_uid"]
    old = database.fetchone("SELECT is_present,searchable FROM sources WHERE id=?", (first["source_id"],))
    assert tuple(old) == (0, 0)
    assert Path(first["artifact_path"]).exists()
    assert not MemoryService(database).context_for_topic("自主判断")
    assert MemoryService(database).list_items()[0]["available_for_recall"] is False
    found = BM25Retriever(database).search(RetrievalQuery("自主判断", authorship="user"))
    assert any(hit.source_id == corrected["source_id"] for hit in found)
    assert all(hit.source_id != first["source_id"] for hit in found)
    assert chat_service.commit(preview["preview_id"], ["乙"], confirm_agent_access=True)["superseded_imports"] == 0

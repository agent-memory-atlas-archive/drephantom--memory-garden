"""Synthetic Chatlog-format exports: attribution stays explicit and local."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_garden.chat_import import ChatImportService, parse_chat_export
from memory_garden.config import Settings
from memory_garden.db import Database


def _message(**changes: object) -> dict:
    message = {"seq": 1704067200001, "time": "2024-01-01T08:00:00+08:00",
               "talker": "wxid_friend", "talkerName": "合成会话", "isChatRoom": False,
               "sender": "wxid_self", "senderName": "同名", "isSelf": True,
               "type": 1, "subType": 0, "content": "我现在会先形成自己的判断。"}
    message.update(changes)
    return message


def test_chatlog_and_session_json_keep_identity_time_and_original_lines() -> None:
    rows = [_message(), _message(seq=1704067200002, sender="wxid_friend", isSelf=False,
                                content="你应该完全听我的。", id=1704067200002, createTime=1704067200, isSend=0)]
    content = json.dumps(rows, ensure_ascii=False, indent=2)
    actual, messages, warnings = parse_chat_export(content)
    assert actual == "wechat_json"
    assert [message.sender for message in messages] == ["同名（wxid_self）", "同名（wxid_friend）"]
    assert messages[0].sender_id == "wxid_self"
    assert messages[0].timestamp == "2024-01-01T08:00:00+08:00"
    assert messages[0].platform_message_id == "wxid_friend:1704067200001"
    assert messages[0].text == rows[0]["content"]
    lines = content.splitlines()
    source_slice = "\n".join(lines[messages[1].line_start - 1:messages[1].line_end])
    assert json.loads(source_slice.rstrip(",")) == rows[1]
    assert any("本人标记仍需你确认" in warning for warning in warnings)


def test_nontext_content_and_nested_quotes_never_become_personal_claims() -> None:
    rows = [_message(type=49, subType=57, content="包含引用的回复", contents={"refer": _message(content="他人秘密")}),
            _message(type=49, subType=19, content="转发标题", contents={"recordInfo": {"body": "转发原文"}}),
            _message(type=3, content="https://example.invalid/private", contents={"path": "C:/private/image"}),
            _message(type=10000, content="某人加入群聊"), _message(content=" ")]
    _, messages, _ = parse_chat_export(json.dumps(rows), "wechat_json")
    assert [message.text for message in messages] == ["[引用消息]", "[转发消息]", "[图片]", "[系统消息]", "[空文本消息]"]
    assert all(message.force_quoted for message in messages)


def test_is_self_never_authorizes_import_and_only_plaintext_is_own(tmp_path: Path) -> None:
    database = Database(tmp_path / "derived" / "chat.db")
    database.initialize()
    service = ChatImportService(database, Settings(vault_path=tmp_path / "vault", database_path=database.path))
    rows = [_message(), _message(seq=1704067200002, type=49, subType=57, content="别人说的内容")]
    content = json.dumps(rows, ensure_ascii=False, indent=2)
    try:
        preview = service.preview(filename="微信.json", content=content)
        assert preview["format"] == "wechat_json"
        assert preview["counts"]["own"] == 0
        assert all(message["authorship"] == "quoted" for message in preview["messages"])
        assert database.fetchone("SELECT COUNT(*) FROM sources")[0] == 0
        committed = service.commit(preview["preview_id"], ["同名（wxid_self）"], confirm_agent_access=True)
        assert committed["own_messages"] == 1 and committed["quoted_messages"] == 1
        assert Path(committed["artifact_path"]).read_text(encoding="utf-8") == content
        source = database.fetchone("SELECT frontmatter_json FROM sources WHERE id=?", (committed["source_id"],))
        provenance = json.loads(source[0])["message_provenance"]
        assert provenance[0]["sender_id"] == "wxid_self"
        assert provenance[0]["platform_message_id"] == "wxid_friend:1704067200001"
        atoms = database.fetchall("SELECT authorship,text FROM source_atoms ORDER BY seq")
        assert [(row["authorship"], row["text"]) for row in atoms] == [
            ("user", rows[0]["content"]), ("quoted", "[引用消息]")]
    finally:
        database.close()


@pytest.mark.parametrize("changes", [
    {"time": ""}, {"time": "2024-01-01"}, {"time": "2024-99-99T00:00:00Z"},
    {"time": 1704067200}, {"isSelf": "true"}, {"isChatRoom": 0}, {"sender": ""},
    {"sender": {}}, {"talker": ""}, {"seq": True}, {"seq": -1}, {"type": "1"},
    {"subType": None}, {"content": {"text": "unknown"}},
])
def test_malformed_chatlog_records_are_rejected_without_guessing(changes: dict) -> None:
    with pytest.raises(ValueError):
        parse_chat_export(json.dumps([_message(**changes)]), "wechat_json")


def test_missing_signature_and_mixed_formats_are_rejected() -> None:
    missing_time = _message()
    del missing_time["time"]
    generic = {"sender": "我", "timestamp": "2024-01-01", "text": "普通消息"}
    for rows in ([missing_time], [_message(), generic], [generic, _message()]):
        with pytest.raises(ValueError):
            parse_chat_export(json.dumps(rows), "wechat_json")


def test_generic_json_and_existing_formats_are_not_reclassified() -> None:
    generic = {"sender": "我", "timestamp": "2024-01-01", "text": "普通消息", "content": "ignored"}
    actual, messages, _ = parse_chat_export(json.dumps([generic]))
    assert actual == "json" and messages[0].text == "普通消息"
    assert messages[0].sender_id == ""
    with pytest.raises(ValueError, match="不一致"):
        parse_chat_export(json.dumps([generic]), "wechat_json")
    with pytest.raises(ValueError, match="不一致"):
        parse_chat_export(json.dumps([_message()]), "chatlab")

"""A local connector is a bounded HTTP adapter; all responses here are synthetic."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from memory_garden.chat_import import MAX_BYTES, ChatImportService, parse_chat_export
from memory_garden.config import Settings
from memory_garden.db import Database
from memory_garden.local_chat import LocalQCEConnector


def _message(message_id: str, uid: str, text: str, *, stamp: int = 1_700_000_000) -> dict[str, Any]:
    return {"msgId": message_id, "senderUid": uid, "sendNickName": "小林", "msgTime": str(stamp),
            "elements": [{"elementType": 1, "textElement": {"content": text}}]}


def test_sessions_request_only_selected_list_and_keep_token_out_of_url() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"success": True, "data": {
            "friends": [{"uid": "u_one", "uin": "10001", "nick": "小林", "remark": "林同学"}],
            "totalCount": 120, "hasNext": True}})

    connector = LocalQCEConnector("http://localhost:40653", "temporary-secret", transport=httpx.MockTransport(handler))
    result = connector.sessions(page=2)
    assert result["items"] == [{"id": "u_one", "name": "林同学", "kind": "friend", "identity": "10001"}]
    assert result["has_more"] and result["page"] == 2
    assert len(requests) == 1 and requests[0].method == "GET"
    assert str(requests[0].url) == "http://127.0.0.1:40653/api/friends?page=2&limit=100"
    assert requests[0].headers["Authorization"] == "Bearer temporary-secret"


def test_export_preserves_own_text_provenance_and_excludes_reply_and_forward_payloads(
    database: Database, settings: Settings,
) -> None:
    first = _message("10", "u_me", "我自己决定。")
    reply = _message("11", "u_other", "我会提供意见。", stamp=1_700_000_001)
    reply["elements"].insert(0, {"elementType": 7, "replyElement": {"replyMsgId": "10", "sourceMsgText": "不要复制的引用"}})
    media = {"msgId": "12", "elements": [{"elementType": 16, "multiForwardMsgElement": {"records": [{"text": "转发内部立场"}]}}]}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"success": True, "data": {"messages": [media, reply, first], "hasNext": True}})

    connector = LocalQCEConnector("http://127.0.0.1:40653", transport=httpx.MockTransport(handler))
    result = connector.export(kind="friend", peer_id="u_peer", count=20)
    assert len(requests) == 1 and requests[0].url.path == "/api/messages/fetch"
    assert json.loads(requests[0].content) == {"peer": {"chatType": 1, "peerUid": "u_peer"}, "page": 1, "limit": 20, "batchSize": 20}
    assert "不要复制的引用" not in result["content"] and "转发内部立场" not in result["content"]
    _, messages, _ = parse_chat_export(result["content"])
    assert [message.platform_message_id for message in messages] == ["10", "11"]
    assert messages[1].reply_to_message_id == "10"
    assert messages[0].sender != messages[1].sender  # Same nickname, distinct account IDs.
    assert result["scope"]["has_more"] and result["scope"]["text_messages"] == 2
    service = ChatImportService(database, settings)
    before = database.connect().total_changes
    preview = service.preview(filename=result["filename"], content=result["content"])
    assert preview["counts"]["own"] == 0 and database.connect().total_changes == before
    committed = service.commit(preview["preview_id"], [messages[0].sender], confirm_agent_access=True)
    atoms = database.fetchall("SELECT sender, authorship FROM source_atoms WHERE source_id=? ORDER BY seq", (committed["source_id"],))
    assert [atom["authorship"] for atom in atoms] == ["user", "quoted"]


@pytest.mark.parametrize("url", ["https://example.com:443", "http://192.168.1.2:40653", "http://localhost.evil:40653",
                               "http://user:secret@127.0.0.1:40653", "http://127.0.0.1:40653/?token=x",
                               "http://127.0.0.1:40653/qce", "file:///chat.db", "http://127.0.0.1:99999"])
def test_rejects_remote_hosts_and_login_urls_before_network(url: str) -> None:
    with pytest.raises(ValueError, match="本机 QCE"):
        LocalQCEConnector(url)


@pytest.mark.parametrize(("status", "message"), [(302, "未能完成"), (401, "令牌"), (403, "令牌"), (503, "尚未连接")])
def test_no_redirect_follow_and_clear_connection_errors(status: int, message: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, headers={"location": "https://example.com/steal"})

    connector = LocalQCEConnector("http://127.0.0.1:40653", "temporary-secret", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match=message):
        connector.sessions()
    assert len(requests) == 1


def test_response_budget_and_unknown_formats_fail_without_partial_export() -> None:
    oversized = httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * (MAX_BYTES + 1)))
    with pytest.raises(ValueError, match="2 MiB"):
        LocalQCEConnector("http://127.0.0.1:40653", transport=oversized).sessions()
    wrong = httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []}))
    with pytest.raises(ValueError, match="有效数据"):
        LocalQCEConnector("http://127.0.0.1:40653", transport=wrong).sessions()


def test_unidentified_speakers_and_system_text_remain_quoted() -> None:
    missing_sender = _message("10", "", "不明来源")
    system = _message("11", "u_me", "系统提示")
    system["msgType"] = 5
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"success": True, "data": {"messages": [missing_sender, system]}}))
    result = LocalQCEConnector("http://127.0.0.1:40653", transport=transport).export(kind="group", peer_id="123")
    _, messages, _ = parse_chat_export(result["content"])
    assert all(message.force_quoted for message in messages)


def test_connection_failure_does_not_reveal_token_or_private_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("temporary-secret upstream-private-data", request=request)

    connector = LocalQCEConnector("http://127.0.0.1:40653", "temporary-secret", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="服务已启动") as error:
        connector.sessions()
    assert "temporary-secret" not in str(error.value)


def test_no_network_for_invalid_count_or_peer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid user input reached the local service")

    connector = LocalQCEConnector("http://127.0.0.1:40653", transport=httpx.MockTransport(handler))
    for kwargs in [{"count": 501, "peer_id": "u_peer"}, {"count": 1, "peer_id": "../another-api"}]:
        with pytest.raises(ValueError):
            connector.export(kind="friend", **kwargs)

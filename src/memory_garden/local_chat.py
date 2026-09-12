"""Read selected QQ conversations from a user-run local QCE service.

This is an HTTP format adapter, not a QQ client or a database decryptor. Only
the three documented read endpoints below are used. No credential is persisted.
"""
from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from .chat_import import MAX_BYTES, UNKNOWN_SENDER

SessionKind = Literal["friend", "group"]
MAX_LOCAL_MESSAGES = 500


def _local_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        valid = (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                 and parsed.port is not None and 1 <= parsed.port <= 65535
                 and not parsed.username and not parsed.password and not parsed.query
                 and not parsed.fragment and parsed.path in {"", "/"})
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("请输入本机 QCE 服务地址，例如 http://127.0.0.1:40653；不要粘贴登录链接或令牌。")
    # Resolve the special localhost spelling ourselves: this connector never
    # needs DNS or an environment proxy to reach the user's own service.
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{parsed.port}"


def _identity(value: Any) -> str:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        result = str(value).strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", result):
            return result
    raise ValueError("QCE 返回的会话或消息编号格式不支持，请改用导出文件导入。")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _timestamp(value: Any) -> str:
    try:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError
        stamp = int(value)
        if stamp <= 0:
            raise ValueError
        seconds = stamp / 1000 if stamp >= 100_000_000_000 else stamp
        return datetime.fromtimestamp(seconds, UTC).isoformat()
    except (ValueError, OverflowError, OSError) as exc:
        raise ValueError("QCE 消息缺少有效时间，无法作为可回溯证据导入。") from exc


class LocalQCEConnector:
    def __init__(self, base_url: str, token: str = "", *, transport: httpx.BaseTransport | None = None):
        self.base_url = _local_url(base_url)
        if len(token) > 4096 or any(ord(char) < 32 or ord(char) > 126 for char in token):
            raise ValueError("QCE 访问令牌格式无效，请只复制令牌本身。")
        self._token = token.strip()
        self._transport = transport

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                 body: dict[str, Any] | None = None) -> dict[str, Any]:
        if (method, path) not in {("GET", "/api/friends"), ("GET", "/api/groups"),
                                  ("POST", "/api/messages/fetch")}:
            raise ValueError("不支持的本地聊天读取操作。")
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        deadline = time.monotonic() + 35
        try:
            with (
                httpx.Client(base_url=self.base_url, headers=headers, timeout=30, follow_redirects=False,
                             trust_env=False, transport=self._transport) as client,
                client.stream(method, path, params=params, json=body) as response,
            ):
                if response.status_code in {401, 403}:
                    raise ValueError("QCE 令牌无效、已过期或未允许本机访问，请核对本地服务的令牌。")
                if response.status_code == 503:
                    raise ValueError("QCE 尚未连接已登录的 QQ；独立文件查看模式不能读取会话。")
                if response.status_code != 200:
                    raise ValueError("本地 QCE 未能完成读取，请检查服务状态，或改用导出文件。")
                payload = bytearray()
                for chunk in response.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > MAX_BYTES:
                        raise ValueError("本次聊天响应超过 2 MiB，请减少读取条数。")
                    if time.monotonic() > deadline:
                        raise ValueError("本地 QCE 读取超时，请减少条数后重试。")
        except httpx.HTTPError as exc:
            raise ValueError("无法读取本地 QCE，请确认服务已启动、账号已登录且地址正确。") from exc
        try:
            result = json.loads(payload)
        except (ValueError, RecursionError, UnicodeError) as exc:
            raise ValueError("本地服务返回了无法识别的响应，请核对 QCE 版本。") from exc
        if not isinstance(result, dict) or result.get("success") is not True or not isinstance(result.get("data"), dict):
            raise ValueError("本地 QCE 未返回有效数据，请核对服务状态和版本。")
        data: dict[str, Any] = result["data"]
        return data

    def sessions(self, *, kind: SessionKind = "friend", page: int = 1, limit: int = 100) -> dict[str, Any]:
        if kind not in {"friend", "group"} or not 1 <= page <= 1000 or not 1 <= limit <= 100:
            raise ValueError("请选择好友或群聊，并使用有效的页码。")
        field = "friends" if kind == "friend" else "groups"
        data = self._request("GET", f"/api/{field}", params={"page": page, "limit": limit})
        rows = data.get(field)
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError("QCE 会话列表格式不支持，请改用导出文件导入。")
        items = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("QCE 会话列表格式不支持。")
            identity = _identity(row.get("uid") if kind == "friend" else row.get("groupCode"))
            name = (_text(row.get("remark")) or _text(row.get("nick")) if kind == "friend"
                    else _text(row.get("groupName")))
            items.append({"id": identity, "name": (name or identity)[:200], "kind": kind,
                          "identity": str(row.get("uin") or identity) if kind == "friend" else identity})
        total = data.get("totalCount")
        return {"items": items, "page": page, "has_more": data.get("hasNext") is True,
                "total": total if isinstance(total, int) and total >= 0 else len(items)}

    def export(self, *, kind: SessionKind, peer_id: str, count: int = 200) -> dict[str, Any]:
        if kind not in {"friend", "group"} or type(count) is not int or not 1 <= count <= MAX_LOCAL_MESSAGES:
            raise ValueError("请选择好友或群聊；一次可读取最近 1 至 500 条消息。")
        peer_id = _identity(peer_id)
        data = self._request("POST", "/api/messages/fetch", body={
            "peer": {"chatType": 1 if kind == "friend" else 2, "peerUid": peer_id},
            "page": 1, "limit": count, "batchSize": count,
        })
        rows = data.get("messages")
        if not isinstance(rows, list) or len(rows) > count:
            raise ValueError("QCE 消息格式不支持，请改用 JSON 导出文件导入。")
        messages, omitted = _normalize_messages(rows)
        if not messages:
            raise ValueError("所选会话本次没有可导入的文字消息；图片、语音、卡片和转发内容暂不展开。")
        content = json.dumps(messages, ensure_ascii=False, indent=2)
        if len(content.encode("utf-8")) > MAX_BYTES:
            raise ValueError("生成的聊天文件超过 2 MiB，请减少读取条数。")
        warnings = ["仅包含本次返回的近期文字消息，不代表完整聊天历史。"]
        if omitted:
            warnings.append(f"{omitted} 条消息含有未展开的附件、引用预览、卡片、转发或系统内容；仅保留独立文字正文。")
        return {"filename": f"qq-{kind}-{peer_id}.json", "content": content, "warnings": warnings,
                "scope": {"provider": "qce", "kind": kind, "peer_id": peer_id, "requested_count": count,
                          "received_count": len(rows), "text_messages": len(messages),
                          "has_more": data.get("hasNext") is True}}


def _normalize_messages(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    """Extract native text elements only; do not flatten others' quoted text."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    omitted = 0
    labels: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("elements"), list):
            raise ValueError("QCE 原始消息结构不支持，请改用导出文件导入。")
        texts, reply_id, skipped = [], "", False
        for element in row["elements"]:
            if not isinstance(element, dict):
                raise ValueError("QCE 消息片段格式不支持。")
            text_element = element.get("textElement")
            if element.get("elementType") == 1 and isinstance(text_element, dict) and isinstance(text_element.get("content"), str):
                texts.append(text_element["content"])
            else:
                skipped = True
                reply = element.get("replyElement")
                if isinstance(reply, dict):
                    # Only a direct ID; do not invent a match from time or name.
                    reply_id = _text(reply.get("replyMsgId")) or _text(reply.get("replayMsgId"))
        omitted += int(skipped)
        text = "".join(texts)
        if not text.strip():
            continue
        message_id = _identity(row.get("msgId"))
        if message_id in seen:
            continue
        seen.add(message_id)
        uid = _text(row.get("senderUid")) or _text(row.get("senderUin"))
        name = _text(row.get("sendMemberName")) or _text(row.get("sendNickName"))
        if uid and uid != "0":
            uid = _identity(uid)
            labels.setdefault(uid, name[:80] or uid)
            label = f"{labels[uid]} ({uid})"
        else:
            uid, label = "", UNKNOWN_SENDER
        result.append({"sender": label, "timestamp": _timestamp(row.get("msgTime")), "text": text,
                       "sender_id": uid, "sender_name": name[:80], "platform_message_id": message_id,
                       "reply_to_message_id": reply_id[:200], "message_type": "text",
                       "force_quoted": not uid or row.get("msgType") in {5, 19}})
    result.sort(key=lambda message: (message["timestamp"], message["platform_message_id"]))
    return result, omitted

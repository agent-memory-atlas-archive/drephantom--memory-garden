"""Preview and explicitly index local chat exports without sending them to a model.

Export formats vary. Only recognisable timestamp/sender records are accepted;
unknown speakers are always quoted. Previews expire in memory, and the original
export is copied into the derived-data directory only after confirmation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database, utc_now

MAX_BYTES = 2 * 1024 * 1024
MAX_MESSAGES = 2000
MAX_PREVIEWS = 5
PREVIEW_TTL = 900
UNKNOWN_SENDER = "未知发言人"
_STAMP = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?"
_DATE_FIRST = re.compile(rf"^\s*(?:###\s*)?\[?(?P<stamp>{_STAMP})\]?\s*(?P<rest>.*)$")
_NAME_FIRST = re.compile(rf"^\s*(?P<name>.+?)\s+\[?(?P<stamp>{_STAMP})\]?\s*$")
_MEDIA = re.compile(r"\[(?:图片|视频|语音|文件|动画表情|image|video|audio|file)\]", re.I)


@dataclass(frozen=True)
class ChatMessage:
    sender: str
    timestamp: str | None
    text: str
    line_start: int
    line_end: int
    index: int
    sender_id: str = ""
    sender_name: str = ""
    platform_message_id: str = ""
    reply_to_message_id: str = ""
    force_quoted: bool = False
    message_type: str = "text"


@dataclass
class _Preview:
    filename: str
    content: str
    format: str
    messages: list[ChatMessage]
    warnings: list[str]
    expires_at: float


def _timestamp(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    value = value.strip().replace("/", "-")
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?", value):
        date, _, clock = value.replace("T", " ").partition(" ")
        year, month, day = (int(part) for part in date.split("-"))
        value = f"{year:04d}-{month:02d}-{day:02d}" + ("T" + clock.zfill(8 if clock.count(":") == 2 else 5) if clock else "")
    try:
        if "T" not in value and " " not in value and len(value) == 10:
            return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError as exc:
        raise ValueError("消息时间格式无效，请使用 YYYY-MM-DD HH:MM:SS 或 ISO 时间。") from exc


def _safe_filename(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if Path(name).suffix.lower() not in {".txt", ".md", ".json"}:
        raise ValueError("请选择 UTF-8 的 TXT、Markdown 或 JSON 聊天导出；暂不支持压缩包、媒体和数据库文件。")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if Path(name).stem.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{n}" for n in range(1, 10)], *[f"LPT{n}" for n in range(1, 10)]}:
        name = "chat-" + name
    return name[:100 - len(Path(name).suffix)] + Path(name).suffix if len(name) > 100 else name


def _check_content(content: str) -> None:
    if not isinstance(content, str):
        raise ValueError("导出内容必须是文本。")
    try:
        size = len(content.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("文本编码无效，请重新导出为 UTF-8。") from exc
    if size > MAX_BYTES:
        raise ValueError("单次导入最多 2 MiB，请按时间分段导出。")
    if not content.strip():
        raise ValueError("文件没有可预览的文本。")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", content):
        raise ValueError("文件包含二进制或控制字符，请选择纯文本聊天导出。")


def _json_depth_guard(content: str) -> None:
    depth, quoted, escaped = 0, False, False
    for char in content:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 32:
                raise ValueError("JSON 嵌套过深；请使用包含 sender、timestamp、text 的消息数组。")
        elif char in "]}":
            depth -= 1


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 包含重复字段，请重新导出。")
        result[key] = value
    return result


def _message_array_start(content: str, decoder: json.JSONDecoder) -> int:
    """Locate the top-level messages array without regex-matching quoted text."""
    cursor = content.index("{") + 1
    while cursor < len(content):
        while content[cursor].isspace() or content[cursor] == ",":
            cursor += 1
        key, cursor = decoder.raw_decode(content, cursor)
        while content[cursor].isspace():
            cursor += 1
        cursor += 1  # colon, already validated by decoding the envelope
        while content[cursor].isspace():
            cursor += 1
        if key == "messages":
            return cursor
        _, cursor = decoder.raw_decode(content, cursor)
    raise ValueError("JSON 缺少 messages 消息数组。")


def _unix_timestamp(value: Any, *, milliseconds: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("导出消息的 Unix 时间戳格式无效。")
    try:
        return datetime.fromtimestamp(value / 1000 if milliseconds else value, UTC).isoformat()
    except (ValueError, OverflowError, OSError) as exc:
        raise ValueError("导出消息的 Unix 时间戳超出有效范围。") from exc


def _identity_label(name: Any, identity: Any) -> str:
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 100:
        raise ValueError("导出消息缺少有效的发言人 ID，无法区分同名用户。")
    if name is not None and not isinstance(name, str):
        raise ValueError("导出的发言人名称必须是文本。")
    return f"{str(name or '未命名').strip()[:80]}（{identity.strip()}）"


_WECHAT_JSON_FIELDS = {"seq", "time", "talker", "talkerName", "isChatRoom", "sender", "senderName",
                       "isSelf", "type", "subType", "content"}


def _adapt_wechat_message(item: dict[str, Any], labels: dict[str, str]) -> dict[str, Any]:
    """Read an existing Chatlog JSON export, without accessing a WeChat client.

    Wire model: myysophia/wechat-log@f10e56c671519c81bf4ba5e24623419851561007,
    internal/model/message.go. Chatlog Session's JSON export preserves these fields
    at xlight/chatlog-session@bd248cf0c39e80ab72d3aea8272115d480573637.
    This independent adapter is format compatibility, not an exporter endorsement.
    """
    if not item.keys() >= _WECHAT_JSON_FIELDS:
        raise ValueError("微信 Chatlog JSON 缺少完整消息字段，请使用原始消息数组。")
    if any(not isinstance(item[key], bool) for key in ("isSelf", "isChatRoom")):
        raise ValueError("微信 Chatlog JSON 的身份与会话标记必须是布尔值。")
    if any(isinstance(item[key], bool) or not isinstance(item[key], int) or item[key] < 0
           for key in ("seq", "type", "subType")):
        raise ValueError("微信 Chatlog JSON 的 seq、type 和 subType 必须是非负整数。")
    if any(not isinstance(item[key], str) for key in ("time", "talker", "talkerName", "sender", "senderName", "content")):
        raise ValueError("微信 Chatlog JSON 的时间、发言人和内容必须是文本。")
    talker = item["talker"]
    if (not talker.strip() or len(talker) > 100 or len(str(item["seq"])) > 20
            or re.search(r"[\x00-\x1f]", talker)):
        raise ValueError("微信 Chatlog JSON 的会话或消息编号无效。")
    stamp = _timestamp(item["time"])
    if stamp is None or "T" not in stamp:
        raise ValueError("微信 Chatlog JSON 需要原始消息时间，不能用导入日期代替。")
    identity, name = item["sender"], item["senderName"]
    label = labels.setdefault(identity, _identity_label(name, identity))
    kind, subtype = item["type"], item["subType"]
    placeholders = {3: "[图片]", 34: "[语音]", 42: "[名片]", 43: "[视频]", 47: "[动画表情]",
                    48: "[位置]", 49: "[分享消息]", 50: "[通话记录]", 10000: "[系统消息]",
                    10002: "[撤回消息]", 99998: "[未读取的时间范围]", 99999: "[消息缺口]"}
    text = item["content"] if kind == 1 else placeholders.get(kind, "[非文本消息]")
    if kind == 49:
        text = {6: "[文件]", 19: "[转发消息]", 57: "[引用消息]"}.get(subtype, text)
    return {"sender": label, "sender_id": identity, "sender_name": name, "timestamp": stamp,
            "text": text if text.strip() else "[空文本消息]", "message_type": f"{kind}:{subtype}",
            "platform_message_id": f"{talker}:{item['seq']}", "force_quoted": kind != 1 or not text.strip()}


def _adapt_wire_message(item: dict[str, Any], wire: str, labels: dict[str, str]) -> dict[str, Any]:
    """Independent field adapters, based on official wire-format declarations.

    QCE: qq-chat-export-core/src/types.rs at 7fcca88880c2eb8c12c51c2b6cc49ee805a53d0c.
    ChatLab: docs/cn/standard/chatlab-format.md, v0.0.2 at
    89974f34e3f3c5c85d21b3933d087ad1f1c00711. No exporter implementation is copied.
    """
    if wire == "wechat_json":
        return _adapt_wechat_message(item, labels)
    if wire == "chatlab":
        identity = item.get("sender")
        name = item.get("accountName")
        label = labels.setdefault(str(identity), _identity_label(name, identity))
        kind = item.get("type")
        if isinstance(kind, bool) or not isinstance(kind, int):
            raise ValueError("ChatLab 消息 type 必须是标准中的整数类型。")
        body = item.get("content")
        if body is not None and not isinstance(body, str):
            raise ValueError("ChatLab content 必须是文本或 null。")
        placeholder = {1: "[图片]", 2: "[语音]", 3: "[视频]", 4: "[文件]", 5: "[动画表情]",
                       26: "[转发消息]", 80: "[系统消息]", 81: "[撤回消息]"}.get(kind, "[非文本消息]")
        return {"sender": label, "timestamp": _unix_timestamp(item.get("timestamp")),
                "text": body if body and body.strip() else placeholder,
                "sender_id": identity, "sender_name": name or "", "message_type": str(kind),
                "platform_message_id": item.get("platformMessageId") or "",
                "reply_to_message_id": item.get("replyToMessageId") or "",
                "force_quoted": kind not in {0, 25} or not body or not body.strip()}
    if wire == "qq_json":
        sender, body = item.get("sender"), item.get("content")
        if not isinstance(sender, dict) or not isinstance(body, dict):
            raise ValueError("QCE JSON 消息需要 sender 和 content 对象。")
        identity = sender.get("uid") or sender.get("uin")
        name = sender.get("name") or sender.get("nickname") or sender.get("groupCard")
        label = labels.setdefault(str(identity), _identity_label(name, identity))
        text = body.get("text", "")
        if not isinstance(text, str):
            raise ValueError("QCE content.text 必须是文本。")
        elements = body.get("elements", [])
        if not isinstance(elements, list) or any(not isinstance(element, dict) for element in elements):
            raise ValueError("QCE content.elements 必须是消息元素数组。")
        kinds = {str(element.get("type")) for element in elements}
        if not text.strip():
            media_labels = {"image": "[图片]", "audio": "[语音]", "video": "[视频]", "file": "[文件]",
                            "face": "[动画表情]", "market_face": "[动画表情]", "forward": "[转发消息]"}
            text = " ".join(media_labels[kind] for kind in sorted(kinds) if kind in media_labels) or "[非文本消息]"
        stamp = _unix_timestamp(item["timestamp"], milliseconds=True) if item.get("timestamp") else _timestamp(item.get("time"))
        return {"sender": label, "timestamp": stamp, "text": text,
                "sender_id": identity, "sender_name": name or "", "platform_message_id": item.get("id") or "",
                "message_type": str(item.get("type") or ""),
                "force_quoted": bool(item.get("system") or item.get("recalled") or kinds & {"reply", "forward", "system"} or not body.get("text", "").strip())}
    return item


def _parse_json(content: str) -> tuple[str, list[ChatMessage]]:
    _json_depth_guard(content)
    decoder = json.JSONDecoder(object_pairs_hook=_unique_json_keys)
    cursor = len(content) - len(content.lstrip())
    wire = "json"
    envelope = content[cursor:cursor + 1] == "{"
    labels: dict[str, str] = {}
    if envelope:
        try:
            root = decoder.decode(content)
        except (ValueError, RecursionError) as exc:
            raise ValueError("JSON 格式无法解析，请检查导出文件。") from exc
        if not isinstance(root.get("messages"), list):
            raise ValueError("暂不支持分块 manifest；请选择包含 messages 数组的单文件 JSON。")
        if len(root["messages"]) > MAX_MESSAGES:
            raise ValueError("单次最多导入 2000 条消息，请按时间分段导出。")
        if isinstance(root.get("chatlab"), dict):
            if root["chatlab"].get("version") not in {"0.0.1", "0.0.2"}:
                raise ValueError("当前只支持 ChatLab 0.0.1 / 0.0.2 标准 JSON，请核对格式版本。")
            wire = "chatlab"
            members = root.get("members", [])
            if not isinstance(members, list) or len(members) > MAX_MESSAGES:
                raise ValueError("ChatLab members 成员列表无效或过大。")
            for member in members:
                if not isinstance(member, dict):
                    raise ValueError("ChatLab 成员必须是对象。")
                identity = member.get("platformId")
                label = _identity_label(member.get("accountName"), identity)
                if identity in labels:
                    raise ValueError("ChatLab 成员 ID 重复，无法确认发言人。")
                labels[str(identity)] = label
        elif isinstance(root.get("metadata"), dict) and isinstance(root.get("chatInfo"), dict):
            wire = "qq_json"
        else:
            raise ValueError("未识别的 JSON 对象；支持 QCE 单文件 JSON、ChatLab 标准 JSON 或 sender/timestamp/text 数组。")
        cursor = _message_array_start(content, decoder)
    if content[cursor:cursor + 1] != "[":
        raise ValueError("JSON 格式应为消息数组，每条包含 sender、timestamp、text。")
    cursor += 1
    messages: list[ChatMessage] = []
    record_count = 0
    while True:
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if content[cursor:cursor + 1] == "]":
            if not envelope and content[cursor + 1:].strip():
                raise ValueError("JSON 数组之后含有额外内容。")
            return wire, messages
        start = cursor
        try:
            item, cursor = decoder.raw_decode(content, cursor)
        except (ValueError, RecursionError) as exc:
            raise ValueError("JSON 格式无法解析，请检查消息数组。") from exc
        record_count += 1
        if record_count > MAX_MESSAGES:
            raise ValueError("单次最多导入 2000 条消息，请按时间分段导出。")
        if not isinstance(item, dict):
            raise ValueError("每条 JSON 消息必须是对象。")
        if record_count == 1 and not envelope and "text" not in item and item.keys() >= _WECHAT_JSON_FIELDS:
            wire = "wechat_json"
        item = _adapt_wire_message(item, wire, labels)
        if not isinstance(item.get("text"), str):
            raise ValueError("每条 JSON 消息需要 text 文本字段。")
        sender = item.get("sender")
        if sender is not None and not isinstance(sender, str):
            raise ValueError("sender 必须是发言人名称文本。")
        stamp = item.get("timestamp")
        if stamp is not None and not isinstance(stamp, str):
            raise ValueError("timestamp 必须是明确的日期时间文本。")
        sender = str(sender or UNKNOWN_SENDER).strip() or UNKNOWN_SENDER
        if len(sender) > 200 or "\n" in sender or "\r" in sender:
            raise ValueError("发言人名称超过 200 个字符，请检查导出字段。")
        text = item["text"]
        # JSON escapes can decode into NUL or unpaired surrogate characters even
        # though the upload itself is valid ASCII. Validate decoded fields too.
        try:
            (sender + text + (stamp or "")).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("JSON 消息包含无效 Unicode 字符，请重新导出为 UTF-8。") from exc
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", sender + text + (stamp or "")):
            raise ValueError("JSON 消息包含二进制或控制字符。")
        if text.strip():
            provenance = {key: item.get(key, "") for key in ("sender_id", "sender_name", "platform_message_id", "reply_to_message_id", "message_type")}
            if any(not isinstance(value, str) or len(value) > 200 for value in provenance.values()):
                raise ValueError("消息发言人或平台来源编号字段无效。")
            try:
                "".join(provenance.values()).encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("消息来源字段包含无效 Unicode 字符。") from exc
            messages.append(ChatMessage(sender, _timestamp(stamp), text,
                content.count("\n", 0, start) + 1, content.count("\n", 0, cursor) + 1, record_count - 1,
                **provenance, force_quoted=bool(item.get("force_quoted", False))))
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        char = content[cursor:cursor + 1]
        if char == ",":
            cursor += 1
            if content[cursor:].lstrip().startswith("]"):
                raise ValueError("JSON 消息数组不能以逗号结尾。")
        elif char != "]":
            raise ValueError("JSON 消息之间需要逗号。")


def _parse_text(content: str) -> tuple[list[ChatMessage], int]:
    messages: list[ChatMessage] = []
    current: tuple[str, str | None, int, int] | None = None
    lines: list[str] = []
    ignored = 0
    record_count = 0

    def flush(end: int) -> None:
        if current is None:
            return
        text = "\n".join(lines).strip("\n")
        if text.strip():
            messages.append(ChatMessage(current[0], current[1], text, current[2], max(end, current[2]), current[3]))

    for number, line in enumerate(content.splitlines(), 1):
        match = _DATE_FIRST.match(line)
        sender, stamp, inline = "", None, ""
        if match:
            stamp = _timestamp(match.group("stamp"))
            rest = match.group("rest").strip()
            sender = rest
            split = re.match(r"^(.*?)[：:]\s*(.*)$", rest)
            if split:
                sender, inline = split.groups()
            sender = sender.strip() or UNKNOWN_SENDER
        else:
            match = _NAME_FIRST.match(line)
            if match:
                sender, stamp = match.group("name").strip(), _timestamp(match.group("stamp"))
        if match:
            record_count += 1
            if record_count > MAX_MESSAGES:
                raise ValueError("单次最多导入 2000 条消息，请按时间分段导出。")
            if len(sender) > 200:
                raise ValueError("发言人名称超过 200 个字符，请检查文本导出格式。")
            flush(number - 1)
            current = (sender, stamp, number, record_count - 1)
            lines = [inline] if inline else []
        elif current is not None:
            lines.append(line)
        elif line.strip():
            ignored += 1
    flush(len(content.splitlines()))
    return messages, ignored


def parse_chat_export(content: str, format: str = "auto") -> tuple[str, list[ChatMessage], list[str]]:
    _check_content(content)
    if format not in {"auto", "wechat", "qq", "json", "qq_json", "chatlab", "wechat_json"}:
        raise ValueError("不支持这种聊天导出格式。")
    content = content.lstrip("\ufeff")
    if format in {"json", "qq_json", "chatlab", "wechat_json"} or re.match(r"\s*(?:\{|\[\s*(?:\{|\]))", content):
        actual, messages = _parse_json(content)
        ignored = 0
        if format in {"qq_json", "chatlab", "wechat_json"} and actual != format:
            raise ValueError("文件与选择的 JSON 导出格式不一致，请使用自动识别并核对预览。")
    else:
        actual = format if format != "auto" else "qq" if re.search(rf"{_STAMP}\s+.+\(\d{{5,}}\)", content) else "wechat"
        messages, ignored = _parse_text(content)
    if not messages:
        raise ValueError("没有识别出消息。TXT 需包含完整日期时间和发言人；JSON 需使用 sender、timestamp、text 消息数组。")
    warnings = []
    if ignored:
        warnings.append(f"忽略了第一条消息前的 {ignored} 行导出说明；请核对预览中的消息数量。")
    if any(message.sender == UNKNOWN_SENDER for message in messages):
        warnings.append("部分记录没有明确发言人，会作为他人或未确认内容保存，不能归为你的观点。")
    if any(message.timestamp is None for message in messages):
        warnings.append("部分消息没有时间，时间线会明确标为未知，不使用导入时间代替。")
    if any(_MEDIA.search(message.text) for message in messages):
        warnings.append("图片、语音、文件等只保留导出中的文字占位符，不读取附件内容。")
    if actual in {"qq_json", "chatlab"}:
        warnings.append("已按导出格式读取稳定发言人 ID；导出者标记不会自动认定为你，请自行勾选本人。Unix 时间戳按 UTC 保留。")
    if actual == "wechat_json":
        warnings.append("已识别微信 Chatlog JSON：保留发言人编号与原始时间；导出中的本人标记仍需你确认。非文本消息仅保留占位，不展开引用、转发或附件。")
    if any(message.force_quoted for message in messages):
        warnings.append("系统、撤回、转发或无法分离原文的引用消息只作引用材料，不作为你的观点。")
    return actual, messages, warnings


def _own_names(names: list[str], participants: set[str]) -> list[str]:
    if not isinstance(names, list) or len(names) > 32 or any(not isinstance(name, str) for name in names):
        raise ValueError("请选择最多 32 个属于你本人的发言名称。")
    selected = sorted({name.strip() for name in names if name.strip()})
    if UNKNOWN_SENDER in selected or any(name not in participants for name in selected):
        raise ValueError("本人名称必须来自预览中的明确发言人，不能选择未知发言人。")
    return selected


def _is_own(message: ChatMessage, own_names: list[str]) -> bool:
    return message.sender in own_names and not message.force_quoted


class ChatImportService:
    def __init__(self, database: Database, settings: Settings):
        self.database, self.settings = database, settings
        self._previews: dict[str, _Preview] = {}
        self._lock = threading.RLock()

    def preview(self, *, filename: str, content: str, format: str = "auto", own_names: list[str] | None = None) -> dict[str, Any]:
        filename = _safe_filename(filename)
        if format == "auto" and Path(filename).suffix.lower() == ".json":
            format = "json"
        actual, messages, warnings = parse_chat_export(content, format)
        participants = sorted({message.sender for message in messages})
        own = _own_names(own_names or [], set(participants))
        now = time.monotonic()
        with self._lock:
            self._previews = {key: value for key, value in self._previews.items() if value.expires_at > now}
            while len(self._previews) >= MAX_PREVIEWS:
                self._previews.pop(next(iter(self._previews)))
            preview_id = uuid.uuid4().hex
            self._previews[preview_id] = _Preview(filename, content, actual, messages, warnings, now + PREVIEW_TTL)
        own_count = sum(_is_own(message, own) for message in messages)
        return {
            "preview_id": preview_id, "filename": filename, "format": actual,
            "participants": participants,
            "messages": [{"sender": message.sender, "timestamp": message.timestamp, "text": message.text[:1000],
                          "authorship": "user" if _is_own(message, own) else "quoted", "line_start": message.line_start,
                          "line_end": message.line_end, "index": message.index,
                          "text_truncated": len(message.text) > 1000, "sender_id": message.sender_id,
                          "message_type": message.message_type, "force_quoted": message.force_quoted} for message in messages[:30]],
            "counts": {"messages": len(messages), "own": own_count, "quoted": len(messages) - own_count, "participants": len(participants)},
            "warnings": warnings, "expires_in_seconds": PREVIEW_TTL, "preview_truncated": len(messages) > 30,
        }

    def export_preview(self, preview_id: str) -> tuple[str, str]:
        """Download the selected local preview without adding it to agent memory."""
        with self._lock:
            preview = self._previews.get(preview_id)
            if preview is None or preview.expires_at <= time.monotonic():
                self._previews.pop(preview_id, None)
                raise KeyError('预览已过期，请重新读取所选会话。')
            return preview.filename, preview.content

    def commit(self, preview_id: str, own_names: list[str] | None = None, *, confirm_agent_access: bool = False) -> dict[str, Any]:
        if confirm_agent_access is not True:
            raise ValueError("请先确认允许 Agent 检索这份聊天记录；若使用云端模型，相关片段可能在后续提问时发送到该服务。")
        with self._lock:
            preview = self._previews.get(preview_id)
            if preview is None or preview.expires_at <= time.monotonic():
                self._previews.pop(preview_id, None)
                raise KeyError("预览已过期或服务已重启，请重新选择文件并核对。")
            own = _own_names(own_names or [], {message.sender for message in preview.messages})
            raw_hash = hashlib.sha256(preview.content.encode("utf-8")).hexdigest()
            identity = hashlib.sha256((raw_hash + ":" + json.dumps(own, ensure_ascii=False)).encode()).hexdigest()
            import_id, uid = identity[:24], "chat_" + identity[:32]
            imports_root = (self.settings.database_path.parent / "imports").resolve()
            folder = imports_root / import_id
            artifact = folder / preview.filename
            if not artifact.resolve().is_relative_to(imports_root):
                raise ValueError("导入文件路径无效。")
            existing = self.database.fetchone("SELECT id,rel_path,is_present,searchable FROM sources WHERE uid=? AND source_kind='chat'", (uid,))
            if existing:
                artifact = self.settings.database_path.parent.resolve() / existing["rel_path"]
                if not artifact.resolve().is_relative_to(imports_root):
                    raise ValueError("已有导入文件路径无效。")
            duplicate = bool(existing)
            now = utc_now()
            own_count = sum(_is_own(message, own) for message in preview.messages)
            folder.mkdir(parents=True, exist_ok=True)
            if not artifact.exists():
                # Exclusive create avoids clobbering any existing local artifact.
                with artifact.open("x", encoding="utf-8", newline="") as handle:
                    handle.write(preview.content)
            elif hashlib.sha256(artifact.read_bytes()).hexdigest() != raw_hash:
                raise ValueError("已有导入副本内容不同，请保留该文件并重新检查导入目录。")
            with self.database.transaction() as connection:
                superseded_imports = int(connection.execute(
                    "SELECT COUNT(*) FROM sources WHERE source_kind='chat' AND content_hash=? AND uid!=? "
                    "AND (is_present=1 OR searchable=1)", (raw_hash, uid),
                ).fetchone()[0])
                # Re-confirming the same transcript with a corrected identity
                # replaces its searchable attribution. Keep the older sources
                # and copied files so historic citations remain inspectable.
                connection.execute(
                    "UPDATE sources SET is_present=0,searchable=0 WHERE source_kind='chat' "
                    "AND content_hash=? AND uid!=?", (raw_hash, uid),
                )
                if existing:
                    source_id = int(existing["id"])
                    connection.execute("UPDATE sources SET is_present=1,searchable=1 WHERE id=?", (source_id,))
                    atom_count = int(connection.execute("SELECT COUNT(*) FROM source_atoms WHERE source_id=? AND is_current=1", (source_id,)).fetchone()[0])
                else:
                    metadata = {"source_kind": "chat", "format": preview.format, "original_filename": preview.filename,
                                "own_names": own, "participants": sorted({message.sender for message in preview.messages}),
                                "imported_at": now, "raw_content_hash": raw_hash, "message_count": len(preview.messages),
                                "timestamp_semantics": "exported_message_timestamp",
                                "message_provenance": [{"index": message.index, "sender_id": message.sender_id,
                                    "sender_name": message.sender_name, "platform_message_id": message.platform_message_id,
                                    "reply_to_message_id": message.reply_to_message_id, "message_type": message.message_type,
                                    "force_quoted": message.force_quoted} for message in preview.messages]}
                    title = "聊天记录 · " + Path(preview.filename).stem
                    cursor = connection.execute(
                        "INSERT INTO sources(uid,rel_path,title,frontmatter_json,tags_json,links_json,recorded_at,event_time,modified_at,authorship,content_hash,is_present,searchable,body,source_kind) "
                        "VALUES(?,?,?,?,'[\"聊天记录\"]','[]',NULL,NULL,?,?,?,1,1,'','chat')",
                        (uid, artifact.relative_to(self.settings.database_path.parent.resolve()).as_posix(), title,
                         json.dumps(metadata, ensure_ascii=False), now, "user" if own_count else "quoted", raw_hash),
                    )
                    source_id = int(cursor.lastrowid or 0)
                    connection.execute("INSERT INTO source_revisions(source_uid,content_hash,first_seen_at,last_seen_at) VALUES(?,?,?,?)", (uid, raw_hash, now, now))
                    atom_count = 0
                    for message in preview.messages:
                        parts = [message.text[index:index + 2000] for index in range(0, len(message.text), 2000)]
                        for part_index, text in enumerate(parts):
                            heading = f"{message.sender} · {message.timestamp or '时间未知'} · 消息 {message.index + 1}"
                            if len(parts) > 1:
                                heading += f" · 片段 {part_index + 1}/{len(parts)}"
                            cursor = connection.execute(
                                "INSERT INTO source_atoms(source_id,uid,seq,heading,text,line_start,line_end,recorded_at,event_time,authorship,is_current,revision_hash,sender) "
                                "VALUES(?,?,?,?,?,?,?,?,?,?,1,?,?)",
                                (source_id, f"chatatom_{import_id}_{message.index}_{part_index}", atom_count, heading, text,
                                 message.line_start, message.line_end, message.timestamp, message.timestamp,
                                 "user" if _is_own(message, own) else "quoted", raw_hash, message.sender),
                            )
                            connection.execute("INSERT INTO source_atoms_fts(rowid,text) VALUES(?,?)", (cursor.lastrowid, f"{title}\n{heading}\n{text}"))
                            atom_count += 1
            return {"import_id": import_id, "source_uid": uid, "source_id": source_id,
                    "indexed_messages": len(preview.messages), "indexed_atoms": atom_count,
                    "own_messages": own_count, "quoted_messages": len(preview.messages) - own_count,
                    "duplicate": duplicate, "artifact_path": str(artifact), "superseded_imports": superseded_imports}

    def deactivate(self, source_uid: str) -> dict[str, Any]:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT id FROM sources WHERE uid=? AND source_kind='chat'", (source_uid,)).fetchone()
            if row is None:
                raise KeyError("这份聊天导入不存在。")
            connection.execute("UPDATE sources SET is_present=0,searchable=0 WHERE id=?", (row["id"],))
        return {"source_uid": source_uid, "is_present": False, "searchable": False}

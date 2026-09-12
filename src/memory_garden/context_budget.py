"""Bound serialized model input without rewriting user words or saved evidence.

This is a character budget, not a provider-specific token estimator. It covers the
whole request, including tool schemas. Only disposable tool observations shrink;
the original messages, database and audit trace are not modified.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any


def serialized(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def _atom_ids(value: Any) -> list[int]:
    found: list[int] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == 'atom_id' and isinstance(child, int) and not isinstance(child, bool) and child > 0:
                    if child not in found:
                        found.append(child)
                elif key == 'atom_ids' and isinstance(child, list):
                    for atom in child:
                        if isinstance(atom, int) and not isinstance(atom, bool) and atom > 0 and atom not in found:
                            found.append(atom)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return found


def render_observation(tool: str, data: dict[str, Any], boundary: str, max_chars: int = 6400) -> str:
    """Trim complete result entries, retaining endpoints and explicit omissions.

    Short results keep their original data. Timelines retain both ends; ranked
    result lists retain their first entries. Individual strings are never cut
    into a misleading partial quote: an oversized field is explicitly omitted.
    """
    def render(value: dict[str, Any]) -> str:
        return f'[{tool}]\n{serialized(value)}\n边界: {boundary}'

    if len(render(data)) <= max_chars:
        return render(data)
    value = copy.deepcopy(data)
    atoms = _atom_ids(value)
    previous = value.get('_compaction', {})
    metadata: dict[str, Any] = {
        'omitted_items': previous.get('omitted_items', 0),
        'omitted_fields': list(previous.get('omitted_fields', [])), 'atom_ids': atoms[:64],
        'more_atom_ids': max(0, len(atoms) - 64),
        'notice': ('结果已按完整条目缩短；省略不代表不存在。需要原文时可 read_source(atom_id) 重新读取。'
                   if atoms else '结果已缩短；省略不代表不存在。需要完整结果时请缩小范围后重新调用当前工具。'),
    }
    value['_compaction'] = metadata
    while len(render(value)) > max_chars:
        lists = [(key, child) for key, child in value.items()
                 if key != '_compaction' and isinstance(child, list) and child]
        if lists:
            key, longest = max(lists, key=lambda pair: len(serialized(pair[1])))
            longest.pop(len(longest) // 2 if key == 'timeline' and len(longest) > 2 else -1)
            metadata['omitted_items'] += 1
            continue
        fields = [(key, child) for key, child in value.items()
                  if key not in {'_compaction', 'atom_id'} and child is not None]
        if not fields:
            break
        key, _ = max(fields, key=lambda pair: len(serialized(pair[1])))
        value[key] = None
        metadata['omitted_fields'].append(key)
    return render(value)


def _decode_observation(content: str) -> tuple[str, dict[str, Any], str] | None:
    if not content.startswith('[') or '\n' not in content:
        return None
    header, payload = content.split('\n', 1)
    if not header.endswith(']'):
        return None
    try:
        data, end = json.JSONDecoder().raw_decode(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return header[1:-1], data, payload[end:].strip().removeprefix('边界: ')


def _observation_pointer(content: str) -> str | None:
    decoded = _decode_observation(content)
    if decoded is None:
        return None
    tool, data, boundary = decoded
    atoms = _atom_ids(data)
    return render_observation(tool, {
        'context_compacted': True, 'atom_ids': atoms[:64],
        'more_atom_ids': max(0, len(atoms) - 64),
        'notice': '较早工具结果已从本次输入移出。这里只是索引，不是原文或证据结论；'
                  + ('需要引用时请 read_source(atom_id) 重新读取。' if atoms
                     else f'需要结果时请按保留的参数重新调用 {tool}。'),
    }, boundary)


@dataclass
class PromptBudget:
    body: dict[str, Any]
    before_chars: int
    after_chars: int
    max_chars: int
    compacted_tool_messages: int

    @property
    def audit(self) -> dict[str, Any]:
        return {'method': 'tool_observations_v1', 'unit': 'serialized_characters',
                'before_chars': self.before_chars, 'after_chars': self.after_chars,
                'max_chars': self.max_chars, 'compacted_tool_messages': self.compacted_tool_messages,
                'within_budget': self.after_chars <= self.max_chars}


def budget_prompt(body: dict[str, Any], max_chars: int) -> PromptBudget:
    before = len(serialized(body))
    result = PromptBudget(body, before, before, max_chars, 0)
    if before <= max_chars:
        return result
    result.body = copy.deepcopy(body)
    messages = result.body.get('messages', [])
    # A parallel assistant tool-call batch and all of its replies stay paired.
    newest_batch = max((index for index, message in enumerate(messages)
                        if message.get('role') == 'assistant' and message.get('tool_calls')), default=-1)
    compacted: set[int] = set()
    for index, message in enumerate(messages):
        if index >= newest_batch or message.get('role') != 'tool':
            continue
        pointer = _observation_pointer(str(message.get('content') or ''))
        if pointer is None or len(pointer) >= len(message.get('content') or ''):
            continue
        message['content'] = pointer
        compacted.add(index)
        result.after_chars = len(serialized(result.body))
        if result.after_chars <= max_chars:
            break
    # A very wide recent batch can itself exceed the remaining space. Preserve
    # whole records where possible, always keeping the boundary and source ids.
    if result.after_chars > max_chars:
        candidates = sorted((index for index, message in enumerate(messages)
                             if message.get('role') == 'tool' and index not in compacted),
                            key=lambda index: len(str(messages[index].get('content') or '')), reverse=True)
        for index in candidates:
            message = messages[index]
            content = str(message.get('content') or '')
            decoded = _decode_observation(content)
            if decoded is None:
                continue
            target = max(800, len(content) - (result.after_chars - max_chars) - 64)
            smaller = render_observation(*decoded, max_chars=target)
            if len(smaller) >= len(content):
                continue
            message['content'] = smaller
            compacted.add(index)
            result.after_chars = len(serialized(result.body))
            if result.after_chars <= max_chars:
                break
    result.compacted_tool_messages = len(compacted)
    return result

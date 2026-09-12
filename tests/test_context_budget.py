"""Synthetic request growth checks; no tokenizer/model quality claims or network."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from memory_garden.context_budget import budget_prompt, render_observation, serialized
from memory_garden.llm import LLMError, OpenAICompatibleClient
from memory_garden.tools import ToolObservation


def payload(content):
    return json.JSONDecoder().raw_decode(content.split('\n', 1)[1])[0]


def batch(index, *, count=1):
    calls = [{'id': f'call-{index}-{i}', 'type': 'function',
              'function': {'name': 'search_sources', 'arguments': '{}'}} for i in range(count)]
    return [{'role': 'assistant', 'content': '', 'tool_calls': calls}, *[
        {'role': 'tool', 'tool_call_id': call['id'], 'content': render_observation('search_sources', {
            'hits': [{'atom_id': index * 100 + j + 1, 'excerpt': '完整来源文字。' * 60} for j in range(5)]
        }, '候选需核对；时间相邻不等于因果。')} for call in calls]]


def test_tool_observation_keeps_valid_json_endpoints_and_omission_locators():
    data = {'timeline': [{'atom_id': i, 'date': f'2020-01-{i:02}', 'excerpt': '原话' * 180}
                         for i in range(1, 21)]}
    original = copy.deepcopy(data)
    rendered = ToolObservation(tool='get_topic_timeline', data=data, boundary='时间相邻不等于因果。').render()
    compact = payload(rendered)
    assert len(rendered) <= 6400
    assert compact['timeline'][0]['atom_id'] == 1
    assert compact['timeline'][-1]['atom_id'] == 20
    assert compact['_compaction']['omitted_items'] == 20 - len(compact['timeline'])
    assert compact['_compaction']['atom_ids'] == list(range(1, 21))
    assert data == original
    assert rendered.endswith('边界: 时间相邻不等于因果。')


def test_second_budget_pass_keeps_omitted_source_ids_and_never_creates_partial_quotes():
    data = {'atom_id': 7, 'text': '一段必须完整保留或明确省略的原话。' * 2000}
    first = render_observation('read_source', data, '只能引用实际原话。')
    first_data = payload(first)
    assert first_data['text'] is None
    assert first_data['_compaction']['omitted_fields'] == ['text']
    second = render_observation('read_source', first_data, '只能引用实际原话。', max_chars=1000)
    assert payload(second)['_compaction']['atom_ids'] == [7]
    assert 'read_source(atom_id)' in second


def test_total_budget_compacts_old_tools_without_breaking_pairs_or_recent_batch():
    messages = [{'role': 'system', 'content': '保留系统边界。'}, {'role': 'user', 'content': '我的新问题原话。'}]
    for i in range(8):
        messages.extend(batch(i))
    messages.extend(batch(9, count=2))
    body = {'model': 'synthetic', 'messages': messages, 'tools': [{'schema': '参数说明' * 500}]}
    original = copy.deepcopy(body)
    result = budget_prompt(body, 15000)
    assert result.before_chars > 15000 >= result.after_chars
    assert result.after_chars == len(serialized(result.body))
    assert result.compacted_tool_messages > 0
    assert result.body['messages'][-3:] == original['messages'][-3:]
    assert result.body['messages'][:2] == original['messages'][:2]
    assert result.body['tools'] == original['tools']
    assert body == original
    first_result = payload(result.body['messages'][3]['content'])
    assert first_result['context_compacted'] is True
    assert first_result['atom_ids'] == [1, 2, 3, 4, 5]
    assert 'read_source(atom_id)' in first_result['notice']
    assert [m.get('tool_call_id') for m in result.body['messages']] == [m.get('tool_call_id') for m in messages]


def test_oversized_recent_parallel_batch_is_trimmed_by_whole_records():
    body = {'messages': [{'role': 'user', 'content': '原话'}, *batch(1, count=6)]}
    result = budget_prompt(body, 9000)
    assert result.before_chars > 9000 >= result.after_chars
    changed = [payload(message['content']) for message in result.body['messages'] if message['role'] == 'tool']
    assert any('_compaction' in data for data in changed)
    for data in changed:
        assert all(hit['excerpt'] == '完整来源文字。' * 60 for hit in data['hits'])
    assert len(result.body['messages']) == len(body['messages'])


def test_irreducible_user_input_and_tool_schema_are_counted_not_silently_cut():
    body = {'messages': [{'role': 'user', 'content': '完整原话' * 400}],
            'tools': [{'description': '参数定义' * 400}]}
    result = budget_prompt(body, 2000)
    assert result.before_chars > 2000
    assert result.after_chars == result.before_chars
    assert result.body == body
    assert result.audit['within_budget'] is False


def test_client_rejects_irreducible_input_before_network_and_records_only_counts(settings, monkeypatch):
    configured = replace(settings, llm_max_input_chars=1000)
    client = OpenAICompatibleClient(configured)
    monkeypatch.setattr('memory_garden.llm.httpx.Client', lambda **kwargs: pytest.fail('must not open network'))
    with pytest.raises(LLMError, match='上下文字符预算'):
        client.chat({'messages': [{'role': 'user', 'content': '不得进入日志的私密原话' * 300}]})
    assert client.total_requests == 0
    assert client.context_budget_events[0]['within_budget'] is False
    assert '私密原话' not in str(client.context_budget_events)


def test_client_transmits_compacted_copy_and_counts_schema(settings, monkeypatch):
    sent = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'choices': [{'message': {'content': '合成回答'}}]}

    class HttpClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            sent.append(kwargs['json'])
            return Response()

    monkeypatch.setattr('memory_garden.llm.httpx.Client', HttpClient)
    configured = replace(settings, llm_max_input_chars=12000, llm_api_key='synthetic',
                         llm_base_url='https://example.invalid')
    client = OpenAICompatibleClient(configured)
    messages = [{'role': 'user', 'content': '本轮原话'}]
    for i in range(10):
        messages.extend(batch(i))
    original = copy.deepcopy(messages)
    client.chat_with_tools(messages, [{'name': 'read_source', 'description': 'schema' * 100}])
    assert len(serialized(sent[0])) <= 12000
    assert messages == original
    assert client.context_budget_events[-1]['compacted_tool_messages'] > 0
    assert client.context_budget_events[-1]['after_chars'] == len(serialized(sent[0]))

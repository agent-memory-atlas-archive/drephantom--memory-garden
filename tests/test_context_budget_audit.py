"""An over-budget model request leaves inspectable counters even when no network call occurs."""
import json
from dataclasses import replace

from memory_garden.agent import AgentHarness, OpenAIProvider
from memory_garden.llm import OpenAICompatibleClient
from memory_garden.retrieval import build_retriever


def test_budget_failure_persists_request_scope_counters(database, settings, monkeypatch):
    configured = replace(settings, llm_max_input_chars=1000,
                         llm_base_url='https://example.invalid', llm_api_key='synthetic', llm_chat_model='synthetic')
    client = OpenAICompatibleClient(configured)
    def no_network():
        raise AssertionError('irreducible planner schema should be rejected before opening network')
    monkeypatch.setattr(client, '_network_client', no_network)
    harness = AgentHarness(database, build_retriever(database, configured), configured, OpenAIProvider(client))
    result = harness.run('合成问题：继续聊自主判断。')
    events = [item for item in result.trace if item['tool'] == 'model_input_budget']
    assert len(events) == 1
    assert events[0]['args']['within_budget'] is False
    assert events[0]['args']['before_chars'] > events[0]['args']['max_chars']
    assert client.total_requests == 0
    saved = database.fetchone('SELECT trace_json FROM agent_runs WHERE message_id=?', (result.message_id,))
    assert events[0] in json.loads(saved['trace_json'])

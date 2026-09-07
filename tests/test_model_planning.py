"""Model decisions govern product turns; offline rules remain an explicitly separate mode."""
from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from memory_garden.agent import AgentHarness, OpenAIProvider
from memory_garden.config import Settings
from memory_garden.evaluation import ScriptedProvider, standard_trace_script
from memory_garden.llm import LLMError, OpenAICompatibleClient
from memory_garden.planning import TurnPlan


class PlannedScript(ScriptedProvider):
    def __init__(self, plan, script=()):
        super().__init__(list(script))
        self.plan = plan
        self.planning_messages = []

    def plan_turn(self, messages):
        self.planning_messages = messages
        return self.plan


@pytest.mark.parametrize('question', [
    '也可能如此，为什么人的想法会随经历变化呢？',
    '以前与现在不一样，是不是很常见？',
    '我现在只是想聊一聊这件事。',
])
def test_model_discussion_is_not_overridden_by_keyword_routes(settings, database, retriever, question):
    first = AgentHarness(database, retriever, settings).run('自主判断')
    reply = '新的信息和经历可能改变人看待问题的角度，但具体到一个人，需要听他的解释。'
    provider = PlannedScript(TurnPlan(intent='conversation', reply=reply))
    result = AgentHarness(database, retriever, settings, provider).run(question, thread_id=first.thread_id)
    assert result.reply == reply and result.answer.answer_type == 'conversation'
    assert result.tool_calls == 0 and provider.cursor == 0
    assert any(first.reply in m['content'] for m in provider.planning_messages)
    assert result.trace[0]['args']['intent'] == 'conversation'


def test_model_source_lookup_does_not_require_full_cognitive_protocol(settings, database, retriever):
    provider = PlannedScript(TurnPlan(intent='source_lookup', query='写作'), [
        {'tool': 'search_sources', 'args': {'query': '写作'}},
        {'final': '这条记录谈到了写作。[A{CHALLENGE}]'},
    ])
    result = AgentHarness(database, retriever, settings, provider).run('把那条写作原话找给我')
    assert result.answer.answer_type == 'source_answer'
    assert result.answer.citations and result.tool_calls == 1
    assert not any(e['tool'] == 'evidence_plan_guard' for e in result.trace)


def test_model_cognitive_trace_keeps_evidence_guards(settings, database, retriever):
    provider = PlannedScript(TurnPlan(intent='cognitive_trace', query='自主判断'),
                             standard_trace_script('自主判断', '区间内经历的可能关联'))
    result = AgentHarness(database, retriever, settings, provider).run('看看我的自主判断是否发生过变化')
    assert result.answer.answer_type == 'traced_change' and result.answer.citations
    assert result.steps == 8 and result.tool_calls == 6


def test_failed_model_does_not_silently_return_an_offline_answer(settings, database, retriever):
    class Unavailable(PlannedScript):
        def plan_turn(self, messages):
            raise LLMError('synthetic failure')
    result = AgentHarness(database, retriever, settings, Unavailable(None)).run('为什么人的想法会变化')
    assert result.backend == 'model_unavailable'
    assert result.answer.question_to_user is None and result.tool_calls == 0
    assert '模型没有完成回应' in result.reply
    assert '补充一句' not in result.reply and result.message_id


def test_planning_uses_one_step_of_the_model_budget(settings, database, retriever):
    provider = PlannedScript(TurnPlan(intent='cognitive_trace', query='自主判断'), standard_trace_script('自主判断', '假设'))
    result = AgentHarness(database, retriever, replace(settings, agent_max_steps=1), provider).run('自主判断')
    assert result.steps == 1 and result.tool_calls == 0 and provider.cursor == 0
    assert result.backend == 'model_unavailable'


def test_causal_trace_cannot_skip_interval_and_both_sides(settings, database, retriever):
    script = standard_trace_script('自主判断', '个人原因')
    provider = PlannedScript(TurnPlan(intent='cognitive_trace', query='自主判断', requires_causal_evidence=True),
                             script[:3]+script[-1:])
    result = AgentHarness(database, retriever, settings, provider).run('我的自主判断为什么发生了变化？')
    assert result.backend == 'model_unavailable'
    assert any(e['tool'] == 'evidence_plan_guard' for e in result.trace)


def test_missing_model_connection_does_not_masquerade_as_an_agent(settings, database, retriever):
    result = AgentHarness(database, retriever, replace(settings, backend='deepseek')).run('为什么会变化')
    assert result.backend == 'model_unavailable' and result.tool_calls == 0
    assert result.stop_reason == 'model_not_configured'


def test_conversation_cannot_invent_a_source_id(settings, database, retriever):
    provider = PlannedScript(TurnPlan(intent='conversation', reply='你的私人记录说明了原因。[A999]'))
    result = AgentHarness(database, retriever, settings, provider).run('为什么会这样')
    assert result.backend == 'model_unavailable' and '[A999]' not in result.reply


def test_planning_request_requires_the_typed_decision(settings, monkeypatch):
    captured = []
    def chat(self, body, timeout_seconds=None):
        captured.append(body)
        return {'choices': [{'message': {'tool_calls': [{'function': {'name': 'plan_turn',
            'arguments': '{"intent":"conversation","query":"","reply":"可以先聊一般情况。","updates_current_view":false}'}}]}}]}
    monkeypatch.setattr(OpenAICompatibleClient, 'chat', chat)
    plan = OpenAIProvider(OpenAICompatibleClient(settings)).plan_turn([{'role':'user','content':'聊聊一般情况'}])
    assert plan.intent == 'conversation'
    assert captured[0]['tool_choice'] == {'type':'function', 'function':{'name':'plan_turn'}}


def test_agent_demo_uses_model_but_never_private_vault_or_old_demo_db(tmp_path, monkeypatch):
    (tmp_path/'.env').write_text('MG_LLM_BASE_URL=https://example.invalid\nMG_LLM_API_KEY=synthetic-only\n'
        'MG_LLM_CHAT_MODEL=fake\nMG_VAULT_PATH=private-vault\nMG_ALLOW_CLOUD_EMBEDDING=true\n', encoding='utf-8')
    monkeypatch.setenv('MG_PUBLIC_DEMO_MODE', 'true')
    monkeypatch.setenv('MG_DEMO_USE_MODEL', 'true')
    agent = Settings.load(tmp_path)
    assert agent.llm_ready and agent.backend != 'local'
    assert agent.vault_path == (tmp_path/'evals/cognitive_mvp_vault').resolve()
    assert agent.database_path == tmp_path/'.local/agent-demo.db'
    assert agent.runtime_settings_path == tmp_path/'.local/agent-demo-settings.json'
    assert not agent.allow_cloud_embedding and not agent.allow_cloud_rerank
    assert not agent.database_path.exists()
    monkeypatch.setenv('MG_DEMO_USE_MODEL', 'false')
    offline = Settings.load(tmp_path)
    assert not offline.llm_ready and offline.backend == 'local'
    assert offline.database_path != agent.database_path


def test_final_response_keeps_only_the_user_reply(settings, monkeypatch):
    def chat(self, messages, tools, **kwargs):
        assert tools[-1]['name'] == 'finish_response' and kwargs['tool_choice'] == 'required'
        return {'content': 'synthetic internal working text', 'reasoning_content': 'synthetic reasoning',
                'tool_calls': [{'function': {'name': 'finish_response', 'arguments': '{"reply":"这条原话可供核对。[A3]"}'}}]}
    monkeypatch.setattr(OpenAICompatibleClient, 'chat_with_tools', chat)
    result = OpenAIProvider(OpenAICompatibleClient(settings)).complete([], [])
    assert result == {'content': '这条原话可供核对。[A3]', 'tool_calls': []}


@pytest.mark.parametrize('calls', [[], [
    {'function': {'name': 'finish_response', 'arguments': '{"reply":"尚未完成检索。"}'}},
    {'function': {'name': 'search_sources', 'arguments': '{"query":"写作"}'}},
]])
def test_unstructured_or_mixed_final_response_is_rejected(settings, monkeypatch, calls):
    monkeypatch.setattr(OpenAICompatibleClient, 'chat_with_tools', lambda *a, **kw: {
        'content': 'working text must not be a final answer', 'tool_calls': calls})
    with pytest.raises(LLMError):
        OpenAIProvider(OpenAICompatibleClient(settings)).complete([], [])


@pytest.mark.parametrize('host,expected', [('api.deepseek.com', {'thinking': {'type': 'disabled'}}),
                                        ('example.invalid', {'reasoning_effort': 'none'})])
def test_provider_specific_non_thinking_request(settings, monkeypatch, host, expected):
    captured = []
    def post(self, url, **kwargs):
        captured.append(kwargs['json'])
        return httpx.Response(200, request=httpx.Request('POST', url), json={'choices':[{'message': {'content':'ok'}}]})
    monkeypatch.setattr(httpx.Client, 'post', post)
    client = OpenAICompatibleClient(replace(settings, llm_base_url='https://'+host,
        llm_api_key='synthetic-key', llm_reasoning_effort='none'))
    client.chat({'model': 'synthetic', 'messages': []})
    for key, value in expected.items():
        assert captured[0][key] == value
    assert 'synthetic-key' not in json.dumps(captured)


def test_tool_budget_finalization_cannot_exceed_model_steps(settings, database, retriever):
    provider = PlannedScript(TurnPlan(intent='source_lookup', query='写作'), [
        {'tool': 'search_sources', 'args': {'query': '写作'}},
        {'final': '已找到原话。[A{CHALLENGE}]'},
    ])
    result = AgentHarness(database, retriever, replace(settings, agent_max_steps=2, agent_max_tool_calls=1),
                          provider).run('查写作原话')
    assert result.steps == 2 and result.tool_calls == 1 and provider.cursor == 1
    assert result.backend == 'model_unavailable'


def test_citation_repair_cannot_exceed_model_steps(settings, database, retriever):
    provider = PlannedScript(TurnPlan(intent='source_lookup', query='写作'), [
        {'tool': 'search_sources', 'args': {'query': '写作'}},
        {'final': '这是非法引用。[A999999]'},
        {'final': '这是合法引用。[A{CHALLENGE}]'},
    ])
    result = AgentHarness(database, retriever, replace(settings, agent_max_steps=3), provider).run('查写作原话')
    assert result.steps == 3 and provider.cursor == 2 and result.backend == 'model_unavailable'


def test_deepseek_forced_structure_explicitly_disables_thinking(settings, monkeypatch):
    captured = []
    def chat(self, body, **kwargs):
        captured.append(body)
        return {'choices':[{'message': {'tool_calls':[{'function': {'name': 'finish_response',
            'arguments': '{"reply":"可以先放下。"}'}}]}}]}
    monkeypatch.setattr(OpenAICompatibleClient, 'chat', chat)
    client = OpenAICompatibleClient(replace(settings, llm_base_url='https://api.deepseek.com', llm_reasoning_effort='high'))
    assert OpenAIProvider(client).complete([], [])['content'] == '可以先放下。'
    assert captured[0]['thinking'] == {'type':'disabled'}


@pytest.mark.parametrize('step_budget', [1, 2])
def test_real_provider_dialogue_does_not_recycle_assistant_claims(settings, database, retriever, step_budget):
    previous = AgentHarness(database, retriever, settings).run('自主判断')
    with database.transaction() as conn:
        conn.execute("UPDATE messages SET content=? WHERE id=?", ('助手未经证实的个人因果故事', previous.message_id))
    class Client:
        calls = 0
        def chat_with_tools(self, messages, tools, **kwargs):
            self.calls += 1
            if tools[0]['name'] == 'plan_turn':
                return {'tool_calls':[{'function': {'name':'plan_turn', 'arguments':
                    '{"intent":"conversation","query":"人的想法为什么变化","updates_current_view":false,"requires_causal_evidence":false}'}}]}
            content = json.dumps(messages, ensure_ascii=False)
            assert '助手未经证实的个人因果故事' not in content
            assert '为什么人的想法会变化' in content
            assert [t['name'] for t in tools] == ['finish_response']
            return {'tool_calls':[{'function': {'name':'finish_response', 'arguments':
                '{"reply":"新信息、处境变化都可能影响人的判断，每个人的情况不同。"}'}}]}
    client = Client()
    result = AgentHarness(database, retriever, replace(settings, agent_max_steps=step_budget),
                          OpenAIProvider(client)).run('为什么人的想法会变化', thread_id=previous.thread_id)
    assert result.steps == step_budget and client.calls == step_budget and result.tool_calls == 0
    assert result.backend == ('model_unavailable' if step_budget == 1 else 'openai_compatible')

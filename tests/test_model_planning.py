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
from memory_garden.models import CognitiveAnswer, DialogueState
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


@pytest.mark.parametrize('requires_causal_evidence', [False, True])
def test_model_source_lookup_does_not_require_full_cognitive_protocol(settings, database, retriever, requires_causal_evidence):
    provider = PlannedScript(TurnPlan(intent='source_lookup', query='写作', requires_causal_evidence=requires_causal_evidence), [
        {'tool': 'search_sources', 'args': {'query': '写作'}},
        {'final': '这条记录谈到了写作。[A{CHALLENGE}]'},
    ])
    result = AgentHarness(database, retriever, settings, provider).run('把那条写作原话找给我')
    assert result.answer.answer_type == 'source_answer'
    assert result.answer.citations and result.tool_calls == 1
    assert not any(e['tool'] == 'evidence_plan_guard' for e in result.trace)


class SyntheticPlanningClient:
    """Exercise the real provider's plan/final protocol without network access."""

    def __init__(self, plan, final_response=None):
        self.plan, self.final_response = plan, final_response
        self.requests = []

    def chat_with_tools(self, messages, tools, **kwargs):
        names = [tool['name'] for tool in tools]
        self.requests.append({'messages': messages, 'tools': names})
        if names == ['plan_turn']:
            return {'tool_calls': [{'function': {'name': 'plan_turn', 'arguments': json.dumps(self.plan)}}]}
        assert names == ['finish_response'] and self.final_response is not None
        return self.final_response


def test_general_ai_concern_after_writing_trace_uses_no_note_tools(settings, database, retriever, monkeypatch):
    tid = database.execute("INSERT INTO threads(title,created_at) VALUES('合成写作线程','2025-01-01')")
    old_statement = '现在我把写作当成理解自己的方式。'
    uid = database.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,'user',?,'2025-01-01')",
                           (tid, old_statement))
    previous = CognitiveAnswer(answer_type='insufficient_evidence', summary='合成的旧写作解释，不是当前问题。', topic='写作',
        dialogue=DialogueState(topic_query='写作', current_statement=old_statement, current_statement_message_id=uid))
    database.execute("INSERT INTO messages(thread_id,role,content,answer_json,created_at,reply_to_user_message_id) "
                     "VALUES(?,'assistant',?,?,'2025-01-01',?)", (tid, previous.summary, previous.model_dump_json(), uid))
    question = '如果AI能替代绝大部分人了，那世界会怎样，我总有一些不太好的预感'
    reply = '技术能力如何发展，与人们如何分配机会和保障，是两个需要分别讨论的问题。'
    client = SyntheticPlanningClient({
        'intent': 'conversation', 'query': '讨论AI替代劳动的社会变化与不安',
        'topic_action': 'none', 'topic': '', 'updates_current_view': False, 'requires_causal_evidence': False,
    }, {'tool_calls': [{'function': {'name': 'finish_response', 'arguments': json.dumps({'reply': reply})}}]})

    def no_search(*args, **kwargs):
        raise AssertionError('A general discussion must not search private notes')

    monkeypatch.setattr(retriever, 'search', no_search)
    result = AgentHarness(database, retriever, settings, OpenAIProvider(client)).run(question, thread_id=tid)
    assert result.reply == reply and result.answer.answer_type == 'conversation'
    assert result.tool_calls == 0 and result.steps == 2 and not result.answer.citations
    assert result.answer.dialogue.topic_query == '' and not result.answer.dialogue.current_statement
    assert [request['tools'] for request in client.requests] == [['plan_turn'], ['finish_response']]
    generation = json.dumps(client.requests[1]['messages'], ensure_ascii=False)
    assert question in generation and old_statement not in generation and previous.summary not in generation


def test_conversation_plan_cannot_bypass_its_own_private_causal_evidence_requirement(settings, database, retriever):
    client = SyntheticPlanningClient({
        'intent': 'conversation', 'query': '核对写作经历与我的担忧之间的原因',
        'topic_action': 'continue', 'topic': '写作', 'updates_current_view': False, 'requires_causal_evidence': True,
    })
    result = AgentHarness(database, retriever, settings, OpenAIProvider(client)).run('我的担忧和之前的写作经历有关吗？')
    assert result.backend == 'model_unavailable' and result.tool_calls == 0 and result.steps == 1
    assert len(client.requests) == 1 and result.message_id is not None
    assert result.stop_reason == 'model_planning_failed'


@pytest.mark.parametrize('final_response', [
    {'tool_calls': [{'function': {'name': 'finish_response', 'arguments': json.dumps({
        'conclusion': {'answer_type': 'traced_change', 'summary': 'synthetic unsupported personal change'}})}}]},
    {'tool_calls': [{'function': {'name': 'search_sources', 'arguments': '{"query":"写作"}'}}]},
])
def test_conversation_final_cannot_smuggle_trace_conclusions_or_note_tools(settings, database, retriever, final_response):
    client = SyntheticPlanningClient({
        'intent': 'conversation', 'query': '讨论社会未来', 'topic_action': 'none', 'topic': '',
        'updates_current_view': False, 'requires_causal_evidence': False,
    }, final_response)
    result = AgentHarness(database, retriever, settings, OpenAIProvider(client)).run('AI会如何影响社会？')
    assert result.backend == 'model_unavailable' and result.tool_calls == 0
    assert result.steps == 2 and len(client.requests) == 2
    assert 'synthetic unsupported' not in result.reply


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


def test_unplanned_adapter_cannot_disguise_discussion_as_offline_answer(settings, database, retriever):
    provider = ScriptedProvider([])
    result = AgentHarness(database, retriever, settings, provider).run('聊聊未来的社会')
    assert result.backend == 'model_unavailable' and result.tool_calls == 0
    assert result.stop_reason == 'model_planning_unavailable' and provider.cursor == 0


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

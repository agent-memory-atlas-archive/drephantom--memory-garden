"""Synthetic regressions for scoped working memory and one accepted model conclusion."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from memory_garden.agent import AgentHarness, OpenAIProvider
from memory_garden.db import utc_now
from memory_garden.llm import LLMError
from memory_garden.models import DialogueState
from memory_garden.planning import GroundedConclusion, QuotedAnchor, TurnPlan
from memory_garden.tools import CognitiveTools


class ConversationProvider:
    name = 'synthetic_planned_conversation'

    def __init__(self, plan):
        self.plan = plan
        self.messages = []

    def plan_turn(self, messages):
        self.messages = messages
        return self.plan

    def complete(self, messages, tools):
        raise AssertionError('synthetic conversation already supplied its reply')


def test_topic_switch_clears_prior_current_view_and_provenance(settings, database, retriever):
    first_provider = ConversationProvider(TurnPlan(intent='conversation', reply='先按你的原话理解。',
        topic_action='switch', topic='职业', query='职业选择', updates_current_view=True))
    first = AgentHarness(database, retriever, settings, first_provider).run('我现在更重视工作的稳定性。')
    assert first.answer.dialogue.current_statement_message_id == first.user_message_id
    switch_provider = ConversationProvider(TurnPlan(intent='conversation', reply='可以聊写作。',
        topic_action='switch', topic='写作', query='写作'))
    second = AgentHarness(database, retriever, settings, switch_provider).run('换个话题，聊写作。', first.thread_id)
    assert second.answer.topic == '写作'
    assert second.answer.dialogue.current_statement is None
    assert second.answer.dialogue.current_statement_message_id is None


def test_follow_up_query_changes_without_changing_topic():
    old = DialogueState(topic_query='自主判断', current_statement='我现在会先形成自己的判断。',
                        current_statement_message_id=17)
    plan = TurnPlan(intent='cognitive_trace', query='自主判断后来有哪些记录', topic_action='continue', topic='自主判断')
    current = AgentHarness._resolve_dialogue(plan, old, '那后来呢？')
    assert current.topic_query == old.topic_query
    assert current.current_statement_message_id == 17
    assert current.current_statement == old.current_statement
    assert current is not old


def test_disagreeing_continuation_cannot_carry_a_different_topics_view():
    old = DialogueState(topic_query='职业', current_statement='我现在重视稳定。')
    plan = TurnPlan(intent='cognitive_trace', query='写作', topic_action='continue', topic='写作')
    assert AgentHarness._resolve_dialogue(plan, old, '写作呢？').current_statement is None


def test_precreated_message_is_not_duplicated_in_context_or_storage(settings, database, retriever):
    now = utc_now()
    thread = database.execute('INSERT INTO threads(title,created_at) VALUES(?,?)', ('合成任务', now))
    question = '我现在愿意先形成自己的判断。'
    user_id = database.execute('INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,?,?,?)',
                              (thread, 'user', question, now))
    provider = ConversationProvider(TurnPlan(intent='conversation', reply='先保留这句表达。',
        topic_action='switch', topic='自主判断', query='自主判断', updates_current_view=True))
    result = AgentHarness(database, retriever, settings, provider).run(question, thread, user_message_id=user_id)
    assert sum(message['content'] == question for message in provider.messages) == 1
    assert database.fetchone("SELECT COUNT(*) AS n FROM messages WHERE role='user'")['n'] == 1
    assert result.thread_id == thread and result.user_message_id == user_id
    assert result.answer.dialogue.current_statement_message_id == user_id
    row = database.fetchone('SELECT answer_json FROM messages WHERE id=?', (result.message_id,))
    assert json.loads(row['answer_json'])['dialogue']['current_statement_message_id'] == user_id


def test_precreated_message_cannot_be_attached_to_another_thread(settings, database, retriever):
    now = utc_now()
    thread = database.execute('INSERT INTO threads(title,created_at) VALUES(?,?)', ('合成任务', now))
    user_id = database.execute('INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,?,?,?)',
                              (thread, 'user', '原话', now))
    with pytest.raises(ValueError, match='不属于'):
        AgentHarness(database, retriever, settings).run('原话', thread + 1, user_message_id=user_id)
    with pytest.raises(ValueError, match='内容不匹配'):
        AgentHarness(database, retriever, settings).run('改写过的原话', thread, user_message_id=user_id)


@pytest.fixture()
def grounded(settings, database, retriever):
    harness = AgentHarness(database, retriever, settings)
    tools = CognitiveTools(database, retriever)
    candidates = tools.find_change_candidates({'topic': '自主判断', 'limit': 1}).data['candidates']
    pair = candidates[0]
    anchors = {side: {'atom_id': pair[side]['atom_id'], 'quote': database.fetchone(
        'SELECT text FROM source_atoms WHERE id=?', (pair[side]['atom_id'],))['text'][:70]}
        for side in ('early', 'recent')}
    summary = f"这些文字还不足以确认变化。[A{anchors['early']['atom_id']}] [A{anchors['recent']['atom_id']}]"
    final = GroundedConclusion(answer_type='no_clear_change', summary=summary, **anchors)
    trace = [{'tool': 'find_change_candidates', 'data': {'candidates': candidates}}]
    plan = TurnPlan(intent='cognitive_trace', query='自主判断', topic='自主判断', topic_action='continue')
    return harness, tools, final, trace, plan


def accept(grounded, final=None, text=None):
    harness, tools, base, trace, plan = grounded
    final = final or base
    return harness._accept_conclusion(final, text or final.summary, plan,
        DialogueState(topic_query='自主判断'), trace, tools)


def test_model_conclusion_is_not_reclassified_using_lexical_overlap(grounded):
    grounded[3][0]['data']['candidates'][0]['lexical_overlap'] = 0.0
    answer = accept(grounded)
    assert answer.answer_type == 'no_clear_change'
    assert answer.summary == grounded[2].summary
    assert answer.recent_position.status == 'latest_memory_candidate'


def test_different_visible_and_saved_model_answers_are_rejected(grounded):
    with pytest.raises(ValueError, match='不一致'):
        accept(grounded, text='你的立场已经完全改变。')


def test_structured_anchors_need_not_all_be_repeated_inline(grounded):
    final = grounded[2].model_copy(deep=True)
    final.summary = f'先看这条原话，另一条可在来源中对照。[A{final.early.atom_id}]'
    answer = accept(grounded, final)
    assert answer.recent_position.citations[0].atom_id == final.recent.atom_id
    assert len(answer.citations) == 2


@pytest.mark.parametrize('answer_type', ['source_answer', 'no_clear_change'])
def test_source_conclusion_cannot_use_structurally_valid_but_empty_evidence(grounded, answer_type):
    harness, tools, _, trace, plan = grounded
    if answer_type == 'source_answer':
        plan = plan.model_copy(update={'intent': 'source_lookup'})
    final = GroundedConclusion(answer_type=answer_type, summary='你的原话是我更重视稳定。')
    with pytest.raises(ValueError, match='证据锚点和正文引用'):
        harness._accept_conclusion(final, final.summary, plan,
            DialogueState(topic_query='自主判断'), trace, tools)


@pytest.mark.parametrize('answer_type', ['source_answer', 'traced_change', 'no_clear_change'])
def test_source_conclusion_cannot_hide_all_citations_in_structured_panel(grounded, answer_type):
    harness, tools, base, trace, plan = grounded
    if answer_type == 'source_answer':
        plan = plan.model_copy(update={'intent': 'source_lookup'})
    final = base.model_copy(update={'answer_type': answer_type, 'summary': '这些是核对后的原文。'})
    with pytest.raises(ValueError, match='证据锚点和正文引用'):
        harness._accept_conclusion(final, final.summary, plan,
            DialogueState(topic_query='自主判断'), trace, tools)


def test_insufficient_evidence_can_abstain_without_fabricating_a_source(grounded):
    final = GroundedConclusion(answer_type='insufficient_evidence', summary='现有记录不足以回答这个问题。')
    answer = accept(grounded, final)
    assert answer.abstained and answer.citations == []


def test_paraphrased_duplicate_questions_are_rejected(grounded):
    final = grounded[2].model_copy(deep=True)
    final.summary += '你现在怎么看这件事？'
    final.question_to_user = '这条记录仍然代表你的看法吗？'
    with pytest.raises(ValueError, match='最多一个追问'):
        accept(grounded, final)


def test_internally_retrieved_but_unreturned_source_cannot_be_quoted(grounded):
    harness, tools, base, _, _ = grounded
    returned = {base.early.atom_id, base.recent.atom_id}
    hidden_id = next(iter(tools._seen_atom_ids - returned))
    text = harness.database.fetchone('SELECT text FROM source_atoms WHERE id=?', (hidden_id,))['text'][:60]
    final = GroundedConclusion(answer_type='insufficient_evidence', summary=f'原话。[A{hidden_id}]',
                              sources=[{'atom_id': hidden_id, 'quote': text}])
    with pytest.raises(ValueError, match='没有观察'):
        accept(grounded, final)


@pytest.mark.parametrize('mutation, expected', [
    ('invented_quote', '引文与来源'), ('unseen', '没有观察'),
    ('one_endpoint', '两个有来源'), ('same_endpoint', '独立记录'),
    ('non_user', '用户本人'), ('missing_anchor', '锚点不一致'),
])
def test_invalid_final_evidence_is_rejected(grounded, mutation, expected):
    harness, tools, base, _, _ = grounded
    final = base.model_copy(deep=True)
    if mutation == 'invented_quote':
        final.early.quote = 'synthetic invented cause absent from the note'
    elif mutation == 'unseen':
        tools._seen_atom_ids.discard(final.early.atom_id)
    elif mutation == 'one_endpoint':
        final.answer_type, final.recent = 'traced_change', None
    elif mutation == 'same_endpoint':
        final.recent = final.early.model_copy()
    elif mutation == 'non_user':
        harness.database.execute("UPDATE source_atoms SET authorship='quoted' WHERE id=?", (final.early.atom_id,))
    elif mutation == 'missing_anchor':
        final.summary += ' [A999999]'
    with pytest.raises(ValueError, match=expected):
        accept(grounded, final)


def test_final_tool_rejects_two_competing_replies():
    class Client:
        def chat_with_tools(self, *args, **kwargs):
            return {'tool_calls': [{'function': {'name': 'finish_response', 'arguments': json.dumps({
                'reply': '发生变化', 'conclusion': {'answer_type': 'insufficient_evidence', 'summary': '证据不足'}})}}]}
    with pytest.raises(LLMError):
        OpenAIProvider(Client()).complete([], [])


def test_mocked_production_final_saves_and_displays_one_grounded_conclusion(settings, database, retriever):
    class Client:
        def chat_with_tools(self, messages, tools, **kwargs):
            if tools[0]['name'] == 'plan_turn':
                args = {'intent': 'cognitive_trace', 'query': '自主判断', 'topic': '自主判断',
                        'topic_action': 'switch', 'updates_current_view': False}
                return {'tool_calls': [{'function': {'name': 'plan_turn', 'arguments': json.dumps(args)}}]}
            observations = [m for m in messages if m['role'] == 'tool']
            if not observations:
                return {'tool_calls': [{'id': str(i), 'function': {'name': name, 'arguments': json.dumps(args)}}
                    for i, (name, args) in enumerate([
                        ('search_sources', {'query': '自主判断'}),
                        ('get_topic_timeline', {'topic': '自主判断'}),
                        ('find_change_candidates', {'topic': '自主判断', 'limit': 1}),
                    ])]}
            candidate = json.loads(observations[-1]['content'].splitlines()[1])['candidates'][0]
            anchors = {side: {'atom_id': candidate[side]['atom_id'], 'quote': database.fetchone(
                'SELECT text FROM source_atoms WHERE id=?', (candidate[side]['atom_id'],))['text'][:50]}
                for side in ('early', 'recent')}
            summary = f"可以看到表达差异，但这些文字还不足以确认立场改变。[A{anchors['early']['atom_id']}] [A{anchors['recent']['atom_id']}]"
            final = {'answer_type': 'no_clear_change', 'summary': summary, **anchors}
            return {'content': '不要显示的草稿', 'tool_calls': [{'function': {
                'name': 'finish_response', 'arguments': json.dumps({'conclusion': final})}}]}

    result = AgentHarness(database, retriever, settings, OpenAIProvider(Client())).run('自主判断有什么变化？')
    assert result.backend == 'openai_compatible'
    assert result.answer.answer_type == 'no_clear_change'
    assert result.answer.source_contract == 'model_grounded_v1'
    assert result.reply == result.answer.summary
    assert '不要显示的草稿' not in result.reply
    row = database.fetchone('SELECT content,answer_json,reply_to_user_message_id FROM messages WHERE id=?', (result.message_id,))
    saved = json.loads(row['answer_json'])
    assert saved['summary'] == row['content'] == result.reply
    assert saved['answer_type'] == 'no_clear_change'
    assert row['reply_to_user_message_id'] == result.user_message_id


def test_production_lookup_cannot_finish_with_only_unanchored_prose(settings, database, retriever):
    class Client:
        def chat_with_tools(self, messages, tools, **kwargs):
            if tools[0]['name'] == 'plan_turn':
                return {'tool_calls': [{'function': {'name': 'plan_turn', 'arguments': json.dumps({
                    'intent': 'source_lookup', 'query': '写作', 'topic': '写作', 'topic_action': 'switch'})}}]}
            if not any(m['role'] == 'tool' for m in messages):
                return {'tool_calls': [{'id': 'search', 'function': {'name': 'search_sources',
                    'arguments': '{"query":"写作"}'}}]}
            return {'tool_calls': [{'function': {'name': 'finish_response',
                                                'arguments': '{"reply":"我认为你的观点改变了。"}'}}]}
    result = AgentHarness(database, retriever, settings, OpenAIProvider(Client())).run('找写作原话')
    assert result.backend == 'model_unavailable'
    assert result.stop_reason == 'final_structure_rejected'
    assert result.answer.answer_type != 'traced_change'


class RoleRepairClient:
    """The same sequence as the synthetic live failure, without any network request."""
    def __init__(self, grounded, mode='context'):
        self.harness, _, self.final, _, _ = grounded
        self.mode = mode
        self.calls = 0
        self.repair_tool_names = []

    @staticmethod
    def tool(name, args):
        return {'id': name, 'function': {'name': name, 'arguments': json.dumps(args)}}

    def chat_with_tools(self, messages, tools, **kwargs):
        self.calls += 1
        if tools[0]['name'] == 'plan_turn':
            return {'tool_calls': [self.tool('plan_turn', {'intent': 'cognitive_trace', 'query': '自主判断',
                'topic': '自主判断', 'topic_action': 'switch'})]}
        observations = [m for m in messages if m['role'] == 'tool']
        if not observations:
            return {'tool_calls': [self.tool(name, args) for name, args in [
                ('search_sources', {'query': '自主判断'}),
                ('get_topic_timeline', {'topic': '自主判断'}),
                ('find_change_candidates', {'topic': '自主判断', 'limit': 1}),
            ]]}
        correction = any(m['role'] == 'user' and '核验问题：' in m['content'] for m in messages)
        final = self.final.model_copy(deep=True)
        if not correction or self.mode == 'repeat':
            # Reading this source does not establish its role as a tested challenge.
            final.challenge = [final.early.model_copy()]
        elif self.mode == 'tool':
            tested = [m for m in observations if 'search_hypothesis_evidence' in m['content']]
            if not tested:
                return {'tool_calls': [self.tool('search_hypothesis_evidence', {
                    'query': '自主判断', 'hypothesis': '表达差异是否构成变化', 'stance': 'challenge'})]}
            hit = json.loads(tested[-1]['content'].splitlines()[1])['hits'][0]
            atom_id = hit['atom_id']
            quote = self.harness.database.fetchone('SELECT text FROM source_atoms WHERE id=?', (atom_id,))['text'][:50]
            final.challenge = [QuotedAnchor(atom_id=atom_id, quote=quote)]
            if f'[A{atom_id}]' not in final.summary:
                final.summary += f'这条记录也可核对。[A{atom_id}]'
        else:
            self.repair_tool_names = [tool['name'] for tool in tools]
            final.sources = [final.early.model_copy()]
        return {'tool_calls': [self.tool('finish_response', {'conclusion': final.model_dump()})]}


@pytest.mark.parametrize('mode', ['context', 'tool'])
def test_invalid_evidence_role_can_be_repaired_within_the_same_run(grounded, mode):
    harness = grounded[0]
    client = RoleRepairClient(grounded, mode)
    result = AgentHarness(harness.database, harness.retriever, harness.settings, OpenAIProvider(client)).run('自主判断有什么变化？')
    assert result.backend == 'openai_compatible'
    assert result.answer.source_contract == 'model_grounded_v1'
    assert sum(event['tool'] == 'final_structure_guard' for event in result.trace) == 1
    assert result.steps == client.calls <= harness.settings.agent_max_steps
    if mode == 'tool':
        assert result.answer.counter_evidence
        assert any(event['tool'] == 'search_hypothesis_evidence' for event in result.trace)
    else:
        assert not result.answer.counter_evidence


def test_invalid_final_is_given_only_one_repair_attempt(grounded):
    harness = grounded[0]
    client = RoleRepairClient(grounded, 'repeat')
    result = AgentHarness(harness.database, harness.retriever, harness.settings, OpenAIProvider(client)).run('自主判断有什么变化？')
    assert result.backend == 'model_unavailable'
    assert result.stop_reason == 'final_structure_rejected'
    assert result.steps == client.calls == 4
    assert sum(event['tool'] == 'final_structure_guard' for event in result.trace) == 1


@pytest.mark.parametrize('step_budget', [3, 4])
def test_final_repair_cannot_extend_model_step_budget(grounded, step_budget):
    harness = grounded[0]
    client = RoleRepairClient(grounded, 'tool')
    result = AgentHarness(harness.database, harness.retriever, replace(harness.settings, agent_max_steps=step_budget),
                          OpenAIProvider(client)).run('自主判断有什么变化？')
    assert result.backend == 'model_unavailable'
    assert result.steps == client.calls == step_budget
    assert result.answer.source_contract is None


def test_exhausted_tool_budget_allows_final_correction_but_no_more_tools(grounded):
    harness = grounded[0]
    client = RoleRepairClient(grounded)
    result = AgentHarness(harness.database, harness.retriever,
        replace(harness.settings, agent_max_tool_calls=3), OpenAIProvider(client)).run('自主判断有什么变化？')
    assert result.backend == 'openai_compatible'
    assert result.tool_calls == 3 and result.steps == client.calls == 4
    assert client.repair_tool_names == ['finish_response']


def test_structured_follow_up_preserves_already_supplied_current_view(grounded):
    harness, tools, base, trace, plan = grounded
    state = DialogueState(topic_query='自主判断', current_statement='我现在会先形成自己的判断。',
                          current_statement_message_id=17)
    final = base.model_copy(update={'question_to_user': '你想先核对哪条原文？'})
    answer = harness._accept_conclusion(final, final.summary, plan, state, trace, tools)
    assert answer.dialogue.phase == 'reflecting'
    assert answer.dialogue.current_statement == state.current_statement
    assert answer.dialogue.current_statement_message_id == 17
    repeated = base.model_copy(update={'question_to_user': '这条最近记录现在仍然代表你的看法吗？'})
    with pytest.raises(ValueError, match='再次索要'):
        harness._accept_conclusion(repeated, repeated.summary, plan, state, trace, tools)

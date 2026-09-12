"""Offline discussion does not turn a general question into private retrospection."""
from __future__ import annotations

import pytest

from memory_garden.agent import AgentHarness
from memory_garden.dialogue import conversation_answer, route_turn
from memory_garden.models import DialogueState
from memory_garden.tools import fallback_topic_key, topic_terms

GENERAL_QUESTIONS = [
    '如果AI能替代绝大部分人了，那世界会怎样，我总有一些不太好的预感',
    '也许是这样的，为什么人的想法会如此变化呢',
    '为什么人的想法会变化',
    '为什么我们会随着经历改变想法',
    '为什么我觉得AI变化很快',
    '聊聊人工智能和未来社会',
    '换个话题，AI会如何影响社会',
    '如果我的工作将来被替代了，生活会怎样',
    '为什么我对未来会有不好的预感',
    '我现在该如何理解人工智能的发展',
    '人工智能可能在很多行业改变人们的工作方式。',
]


@pytest.mark.parametrize('question', GENERAL_QUESTIONS)
@pytest.mark.parametrize('previous', [None, DialogueState(topic_query='写作', phase='awaiting_current_view',
                                                         current_statement='我现在愿意在周末写作。')])
def test_general_question_never_becomes_lookup_or_current_position(question, previous):
    route = route_turn(question, previous)
    assert route.kind == 'conversation' and route.query == ''
    answer = conversation_answer(question, route, previous)
    assert answer.answer_type == 'conversation' and answer.topic is None
    assert answer.recent_position is None and answer.citations == []
    assert answer.dialogue.current_statement is None and answer.dialogue.topic_query == ''
    assert '未连接生成模型' in answer.summary
    assert '写作' not in answer.summary and '观点变化' not in answer.summary
    assert answer.question_to_user is None


@pytest.mark.parametrize('question', [
    '回看写作', '帮我查一下关于自主判断的记录', '找那条写作原话',
    '把那条写作原话找给我', '关于写作，我的想法变过吗？',
    '自主判断这个主题以前到现在有没有变化？', '我以前如何看待写作？',
    '换个话题，独处这个主题以前到现在有没有变化？',
    '我对写作的想法变过吗', '我的想法变过吗', '自主判断这个主题有没有变化',
    '帮我看看有没有我自己没注意到的变化',
])
def test_explicit_record_requests_keep_lookup(question):
    route = route_turn(question, DialogueState(topic_query='写作'))
    assert route.kind == 'lookup'


@pytest.mark.parametrize('topic', ['写作', '自主判断', '谨慎决策', '一个人恢复精力', '书摘中的独立', '人工智能'])
def test_short_topic_only_input_retains_documented_local_search_shortcut(topic):
    assert route_turn(topic, None).kind == 'lookup'


@pytest.mark.parametrize('topic', ['AI 生成的独立草稿', '延迟写下的转折经历', '自主判断'])
def test_delimited_topic_instruction_preserves_retrieval_query_and_memory_key(topic):
    route = route_turn(f'回看「{topic}」', DialogueState(topic_query='写作'))
    assert route.kind == 'lookup' and route.query == topic
    assert topic_terms(route.query) == topic_terms(topic)
    assert fallback_topic_key(route.query) == fallback_topic_key(topic)


def test_reference_lookup_retains_current_view_context():
    previous = DialogueState(topic_query='写作', phase='awaiting_current_view',
                             current_statement='我现在愿意在周末写作。')
    for question in ['那以前呢？', '看看以前的记录', '继续回看']:
        route = route_turn(question, previous)
        assert route.kind == 'reference_lookup' and route.query == '写作'
    statement = '现在的话，我愿意每周写一点。'
    route = route_turn(statement, previous)
    assert route.kind == 'statement'
    assert conversation_answer(statement, route, previous).recent_position.statement == statement


@pytest.mark.parametrize('topic', ['写作', '自主判断'])
def test_general_question_after_grounded_turn_has_no_tool_or_position_write(settings, database, retriever, monkeypatch, topic):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run(topic)
    assert first.tool_calls > 0
    assert first.answer.question_to_user

    def forbidden_search(*args, **kwargs):
        raise AssertionError('General discussion must not search private notes.')

    monkeypatch.setattr(retriever, 'search', forbidden_search)
    result = harness.run(GENERAL_QUESTIONS[0], thread_id=first.thread_id)
    assert result.tool_calls == 0 and not result.private_vault_sent
    assert result.answer.answer_type == 'conversation' and result.answer.recent_position is None
    assert result.answer.citations == [] and result.answer.dialogue.current_statement is None
    assert '未连接生成模型' in result.reply
    assert '来源' not in result.reply and '愿意补充' not in result.reply
    assert database.fetchone('SELECT COUNT(*) n FROM verdicts')['n'] == 0

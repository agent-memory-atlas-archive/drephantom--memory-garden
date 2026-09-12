"""Synthetic multi-turn interactions: a reply to a question is not a new failed search."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from memory_garden.agent import AgentHarness, detect_current_stated
from memory_garden.web import create_app


@pytest.mark.parametrize('statement', [
    '现在的话，我更想先把课程学完，写作可以留到周末。',
    '眼下精力有限，能在周末写一点就好。',
    '我不太认同之前那种写法了，慢慢写也可以。',
    '对我来说，偶尔写几句已经足够。',
])
def test_reply_to_current_view_question_is_received(settings, database, retriever, statement):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run('关于写作，我的想法变过吗？')
    assert first.answer.question_to_user
    assert '对照的早期记录' not in first.reply
    second = harness.run(statement, thread_id=first.thread_id)
    assert second.answer.answer_type == 'conversation'
    assert second.answer.recent_position.statement == statement
    assert second.answer.recent_position.status == 'user_stated_now'
    assert '写作' in second.answer.topic
    assert statement in second.reply
    assert '写作 变过' not in second.reply
    assert second.answer.question_to_user is None and not second.answer.citations
    assert second.tool_calls == 0
    assert '证据不足' not in second.reply and '愿意补充一句' not in second.reply
    assert database.fetchone('SELECT COUNT(*) AS n FROM verdicts')['n'] == 0


def test_context_survives_restart_and_reference_query(settings, database, retriever):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run('写作')
    stated = '现在的话，我愿意在周末写一些练习。'
    harness.run(stated, thread_id=first.thread_id)
    resumed = AgentHarness(database, retriever, settings).run('那以前呢？', thread_id=first.thread_id)
    assert resumed.answer.topic == '写作'
    assert stated in resumed.reply
    assert resumed.answer.question_to_user is None
    assert resumed.answer.dialogue.current_statement == stated
    refs = {c.atom_id for c in resumed.answer.citations}
    trace_refs = {ref for event in resumed.trace for ref in event.get('refs', [])}
    assert refs and refs <= trace_refs


def test_topic_switch_and_new_thread_do_not_inherit_current_statement(settings, database, retriever):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run('写作')
    harness.run('现在的话，我愿意在周末写一些练习。', thread_id=first.thread_id)
    other = harness.run('换个话题，独处这个主题以前到现在有没有变化？', thread_id=first.thread_id)
    assert '写作' not in (other.answer.topic or '')
    assert other.answer.dialogue.current_statement is None
    fresh = harness.run('独处这个主题以前到现在有没有变化？')
    assert fresh.thread_id != first.thread_id and fresh.answer.dialogue.current_statement is None


@pytest.mark.parametrize('pause', ['先不聊了', '我暂时不想说', '先放着', '到这里就好'])
def test_pause_does_not_prompt_or_search_again(settings, database, retriever, pause):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run('写作')
    stopped = harness.run(pause, thread_id=first.thread_id)
    assert stopped.answer.dialogue.phase == 'paused'
    assert stopped.answer.question_to_user is None and stopped.tool_calls == 0
    assert '？' not in stopped.reply
    ack = harness.run('谢谢', thread_id=first.thread_id)
    assert ack.answer.dialogue.phase == 'paused' and ack.tool_calls == 0


@pytest.mark.parametrize('uncertain', ['不知道', '我也说不清'])
def test_uncertainty_is_not_recorded_as_a_position(settings, database, retriever, uncertain):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run('写作')
    result = harness.run(uncertain, thread_id=first.thread_id)
    assert result.answer.recent_position is None
    assert result.answer.dialogue.current_statement is None
    assert result.answer.question_to_user is None and result.tool_calls == 0


@pytest.mark.parametrize('question', ['我现在的想法是什么', '现在我该怎么理解这条记录？'])
def test_question_is_not_mislabelled_as_a_current_position(question):
    assert not detect_current_stated(question)


def test_old_thread_recovers_topic_without_overwriting_history(settings, database, retriever):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run('写作')
    legacy = first.answer.model_dump(exclude={'dialogue', 'topic'})
    database.execute('UPDATE messages SET answer_json=? WHERE id=?',
                     (json.dumps(legacy, ensure_ascii=False), first.message_id))
    result = harness.run('现在的话，我愿意把写作留到周末。', thread_id=first.thread_id)
    assert result.answer.topic == '写作'
    assert database.fetchone('SELECT content FROM messages WHERE id=?', (first.message_id,))['content'] == first.reply


def test_user_citation_text_cannot_become_a_vault_reference(settings, database, retriever):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run('写作')
    result = harness.run('我把 [A999] 当作练习编号，周末再写。', thread_id=first.thread_id)
    assert '[A999]' not in result.reply and '［A999］' in result.reply
    assert not result.answer.citations


def test_unavailable_provider_still_receives_the_followup_locally(settings, database, retriever):
    from memory_garden.llm import LLMError
    class UnavailableProvider:
        name = 'unavailable'
        def complete(self, messages, tools):
            raise LLMError('Synthetic connection failure')
    first = AgentHarness(database, retriever, settings).run('写作')
    result = AgentHarness(database, retriever, settings, provider=UnavailableProvider()).run(
        '现在的话，我愿意把写作留到周末。', thread_id=first.thread_id,
    )
    assert result.backend == 'local_fallback' and not result.private_vault_sent
    assert result.answer.answer_type == 'conversation' and result.answer.question_to_user is None


def test_connected_model_can_respond_naturally_with_the_actual_question_in_history(settings, database, retriever):
    first = AgentHarness(database, retriever, settings).run('写作')
    class ConversationalProvider:
        name = 'conversation-test'
        def complete(self, messages, tools):
            assert not tools
            assert any(first.answer.question_to_user in m['content'] for m in messages)
            assert not any('（用户已回应）' in m['content'] for m in messages)
            return {'content': '你想把写作留给周末。可以先按这个节奏来，不急着给它下结论。'}
    result = AgentHarness(database, retriever, settings, provider=ConversationalProvider()).run(
        '现在的话，我愿意把写作留到周末。', thread_id=first.thread_id,
    )
    assert result.backend == 'conversation-test'
    assert result.reply.startswith('你想把写作留给周末')
    assert result.tool_calls == 0 and not result.answer.citations


@pytest.mark.parametrize('invalid_reply', ['你的笔记证明了这件事 [A999]。', '你愿意补充一句最能代表现在看法的原话吗？'])
def test_dialogue_generation_cannot_reintroduce_missing_citations_or_repeat_prompt(settings, database, retriever, invalid_reply):
    first = AgentHarness(database, retriever, settings).run('写作')
    class InvalidProvider:
        name = 'invalid'
        def complete(self, messages, tools):
            return {'content': invalid_reply}
    result = AgentHarness(database, retriever, settings, provider=InvalidProvider()).run(
        '现在的话，我愿意把写作留到周末。', thread_id=first.thread_id,
    )
    assert result.backend == 'local_fallback'
    assert result.reply != invalid_reply and result.answer.question_to_user is None


def test_web_followup_persists_state_and_has_no_verdict_form(settings):
    with TestClient(create_app(settings)) as client:
        first = client.post('/api/ask', json={'question': '写作'}).json()
        result = client.post('/api/ask', json={'question': '现在的话，我愿意把写作留到周末。',
                            'thread_id': first['thread_id']}).json()
        assert result['answer_type'] == 'conversation' and not result['verdict_worthy']
        history = client.get(f"/api/threads/{first['thread_id']}").json()
        assert history[-1]['answer']['dialogue']['current_statement'].startswith('现在的话')


def test_one_record_with_internal_contrast_is_not_labelled_an_early_position(settings, database, retriever):
    result = AgentHarness(database, retriever, settings).run('写作')
    assert '这段文字把「以前」和「现在」放在一起作了对照' in result.reply
    assert '写下它的时候' in result.reply
    assert '找到的相关记录' in result.reply
    assert result.answer.answer_type != 'traced_change'


def test_new_comparison_question_without_punctuation_is_not_a_reply(settings, database, retriever):
    harness = AgentHarness(database, retriever, settings)
    first = harness.run('写作')
    result = harness.run('自主判断这个主题有没有变化', thread_id=first.thread_id)
    assert result.answer.answer_type != 'conversation'
    assert '自主' in (result.answer.topic or '')

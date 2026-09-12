"""Working context compaction is extractive, scoped, bounded and repeatable."""
from __future__ import annotations

from memory_garden.cognitive import VerdictService
from memory_garden.context import build_thread_context
from memory_garden.db import utc_now
from memory_garden.models import CognitiveAnswer, DialogueState, SourceCitation


def thread(database):
    return database.execute('INSERT INTO threads(title,created_at) VALUES(?,?)', ('合成上下文', utc_now()))


def turn(database, thread_id, topic, text, *, previous=None, update=False, question=None):
    now = utc_now()
    user_id = database.execute('INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,?,?,?)',
                               (thread_id, 'user', text, now))
    state = previous.model_copy(deep=True) if previous else DialogueState(topic_query=topic)
    if update:
        state.current_statement = text
        state.current_statement_message_id = user_id
    state.phase = 'awaiting_current_view' if question else 'reflecting'
    answer = CognitiveAnswer(answer_type='conversation', topic=topic,
        summary='助手未经验证的个人因果猜测', dialogue=state, question_to_user=question,
        citations=[SourceCitation(atom_id=13, recorded_at='2024-01-02', event_time=None)])
    message_id = database.execute(
        'INSERT INTO messages(thread_id,role,content,answer_json,created_at,reply_to_user_message_id) VALUES(?,?,?,?,?,?)',
        (thread_id, 'assistant', answer.summary, answer.model_dump_json(), now, user_id))
    return user_id, message_id, state


def test_long_conversation_keeps_older_current_view_and_first_question(database):
    tid = thread(database)
    first, _, _ = turn(database, tid, '自主判断', '我最初为什么依赖别人的认可？')
    original = '我现在倾向先形成自己的判断，再认真听取不同意见。'
    uid, _, state = turn(database, tid, '自主判断', original, update=True)
    for i in range(20):
        turn(database, tid, '自主判断', f'继续聊第 {i} 个细节。', previous=state)
    context = build_thread_context(database, tid, topic='自主判断')
    assert context.data['current_view']['quote'] == original
    assert context.data['current_view']['message_id'] == uid
    assert first in context.selected_user_ids
    assert any(anchor['atom_id'] == 13 and anchor['recorded_at'] == '2024-01-02'
               and anchor['status'] == 'historical_reference_requires_reread'
               for anchor in context.data['source_anchors'])
    assert '助手未经验证的个人因果猜测' not in context.rendered
    assert context.char_count <= context.max_chars


def test_topics_threads_and_future_turns_are_isolated(database):
    tid = thread(database)
    _, _, writing = turn(database, tid, '写作', '写作对我来说是表达。', update=True)
    turn(database, tid, '职业', '职业上我更重视稳定。', update=True)
    _, _, _ = turn(database, thread(database), '写作', '另一个线程的私密细节。', update=True)
    before, _, _ = turn(database, tid, '写作', '未来才出现的写作看法。', previous=writing, update=True)
    context = build_thread_context(database, tid, topic='写作', before_message_id=before)
    all_text = context.rendered + str(context.recent_messages)
    assert context.data['current_view']['quote'] == '写作对我来说是表达。'
    for excluded in ('职业上我更重视稳定', '另一个线程的私密细节', '未来才出现的写作看法'):
        assert excluded not in all_text


def test_corrections_and_current_words_have_priority_under_budget(database):
    tid = thread(database)
    original = '目前我愿意先自己判断。' * 20
    _, mid, state = turn(database, tid, '自主判断', original, update=True)
    revision = '更准确地说，我仍然重视求证，并非排斥他人建议。' * 8
    VerdictService(database).save_verdict(message_id=mid, verdict='partly_accurate', user_revision=revision)
    for i in range(25):
        turn(database, tid, '自主判断', f'普通对话 {i}：' + '一段不必全部进入上下文的文字。' * 30, previous=state)
    context = build_thread_context(database, tid, topic='自主判断', max_chars=1500)
    assert context.char_count <= 1500
    assert context.data['user_corrections'][0]['quote'] == revision
    assert context.data['current_view']['quote'] == original
    assert context.data['omitted_user_messages'] > 0


def test_repeated_compaction_uses_original_messages_not_earlier_summary(database):
    tid = thread(database)
    original = '现在我的看法是：没有得到认可，也不意味着选择一定错误。'
    uid, _, state = turn(database, tid, '自主判断', original, update=True)
    for i in range(15):
        turn(database, tid, '自主判断', f'后续话题 {i}', previous=state)
    first = build_thread_context(database, tid, topic='自主判断', max_messages=10, max_chars=1200)
    second = build_thread_context(database, tid, topic='自主判断', max_messages=10, max_chars=1200)
    assert first.rendered == second.rendered
    assert first.data['current_view']['quote'] == original
    assert first.data['current_view']['message_id'] == uid
    assert first.scanned_count == 10
    # Even a corrupt in-memory cache is not a persisted/recompressed source.
    first.data['current_view']['quote'] = '摘要变形的说法'
    again = build_thread_context(database, tid, topic='自主判断', max_messages=10, max_chars=1200)
    assert again.data['current_view']['quote'] == original


def test_pending_question_is_separate_from_user_facts(database):
    tid = thread(database)
    turn(database, tid, '写作', '回看写作。', question='这条最近记录现在仍代表你的看法吗？')
    context = build_thread_context(database, tid, topic='写作')
    assert context.data['pending_question']['kind'] == 'previous_agent_question_not_a_fact'
    assert context.data['current_view'] is None
    assert '助手未经验证的个人因果猜测' not in context.rendered


def test_deferred_feedback_stops_repeat_confirmation_without_creating_a_view(database):
    tid = thread(database)
    _, mid, _ = turn(database, tid, '写作', '回看写作。', question='这条最近记录现在仍代表你的看法吗？')
    VerdictService(database).save_verdict(message_id=mid, verdict='defer')
    context = build_thread_context(database, tid, topic='写作')
    assert context.data['pending_question'] is None
    assert context.data['latest_user_feedback']['choice'] == 'defer'
    assert context.data['current_view'] is None
    assert context.data['user_corrections'] == []


def test_older_correction_survives_scan_window_and_other_topics(database):
    tid = thread(database)
    _, mid, _ = turn(database, tid, '写作', '写作是否还重要？')
    revision = '写作仍然重要；只是当下需要先照顾生活。'
    VerdictService(database).save_verdict(message_id=mid, verdict='partly_accurate', user_revision=revision)
    for i in range(65):
        _, other, _ = turn(database, tid, '职业', f'职业方面的新讨论 {i}。')
        if i > 55:
            VerdictService(database).save_verdict(message_id=other, verdict='partly_accurate',
                                                  user_revision=f'职业修正 {i}')
    context = build_thread_context(database, tid, topic='写作')
    assert context.scanned_count == 100
    assert context.data['older_history_not_scanned'] is True
    assert context.data['user_corrections'][0]['quote'] == revision
    assert context.data['user_corrections'][0]['source_answer_id'] == mid
    assert '职业修正' not in context.rendered
    assert context.audit['scanned_message_chars'] > 0


def test_correction_after_pending_cutoff_is_not_recalled(database):
    tid = thread(database)
    before, _, _ = turn(database, tid, '写作', '当时的问题。')
    _, later, _ = turn(database, tid, '写作', '后来才说的话。')
    VerdictService(database).save_verdict(message_id=later, verdict='partly_accurate', user_revision='未来修正')
    context = build_thread_context(database, tid, topic='写作', before_message_id=before)
    assert context.data['user_corrections'] == []

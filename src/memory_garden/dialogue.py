"""Small, explicit dialogue transitions for offline use and provider fallbacks.

Acknowledging a reply to our own question does not require searching the Vault again.
These rules keep user words verbatim; they do not infer feelings, causes or durable beliefs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from .db import Database
from .models import CognitiveAnswer, DialogueState, PositionEvidence, SourceCitation


@dataclass
class TurnRoute:
    kind: str
    query: str


def is_question(text: str) -> bool:
    return bool(re.search(r'[？?]|(?:吗|么|呢)[。！!\s]*$', text) or re.search(
        r'^(?:那|所以|请问|你觉得|你认为)?\s*(?:为什么|怎么|如何|是否|能不能|可不可以)', text
    ) or re.search(r'有没有|是不是|是否|有何|有什么', text)
        or re.search(r'(?:我|你|这|那|它|现在).*?(?:怎么|如何|什么|多少|能不能|该不该)', text))


def current_stated(text: str) -> bool:
    if is_question(text):
        return False
    return bool(re.search(r'(?:我现在|现在我|如今我|目前我|我认为现在|现在的话|如今|目前|眼下)', text))


def load_dialogue(database: Database, thread_id: int | None) -> DialogueState | None:
    if thread_id is None:
        return None
    rows = database.fetchall(
        "SELECT a.answer_json, (SELECT u.content FROM messages u WHERE u.thread_id=a.thread_id "
        "AND u.role='user' AND u.id<a.id ORDER BY u.id DESC LIMIT 1) AS question "
        "FROM messages a WHERE a.thread_id=? AND a.role='assistant' ORDER BY a.id DESC LIMIT 6",
        (thread_id,),
    )
    # Older threads have no dialogue field. Recover their topic from the last grounded turn,
    # rather than the user's follow-up which the old code mistakenly treated as a fresh topic.
    for row in rows:
        if not row['answer_json']:
            continue
        try:
            answer = CognitiveAnswer.model_validate_json(row['answer_json'])
        except ValidationError:
            continue
        if answer.dialogue is not None:
            return answer.dialogue
        if answer.topic or answer.citations or answer.early_position:
            from .tools import topic_terms
            topic = answer.topic or ' '.join(topic_terms(str(row['question'] or '')))
            return DialogueState(
                topic_query=topic,
                phase='awaiting_current_view' if answer.question_to_user else 'reflecting',
            )
    return None


def route_turn(text: str, state: DialogueState | None) -> TurnRoute:
    stripped = text.strip()
    compact = re.sub(r'[\s，,。.!！~～]', '', stripped)
    if re.fullmatch(r'(?:嗯|好|那)?(?:我)?(?:暂时不想说|先不聊(?:了|这个)?|不想聊(?:了|这个)?|'
                    r'先放着|先到这里|到这里就好|以后再说|暂时放着|先不说了)', compact):
        return TurnRoute('pause', state.topic_query if state else '')
    switch = re.match(r'^(?:换个话题|聊聊|说说)[，,：:\s]*(.+)', stripped)
    if switch:
        return TurnRoute('lookup', switch.group(1))
    if state is not None:
        if compact in {'不知道', '我不知道', '说不清', '我也说不清', '还没想好', '暂时没有答案'}:
            return TurnRoute('uncertain', state.topic_query)
        if compact in {'嗯', '嗯嗯', '好的', '好', '谢谢', '谢谢你', '是的', '对', '对的'}:
            return TurnRoute('acknowledge', state.topic_query)
        # Explicit navigation back to the records keeps the topic, even if no topic noun is repeated.
        if re.fullmatch(r'(?:那|再)?(?:以前呢|后来呢|还有呢|继续回看|看看以前的记录|回看旧记录|'
                        r'继续看(?:看)?(?:以前的)?记录)[？?。！!]*', stripped):
            return TurnRoute('reference_lookup', state.topic_query or stripped)
        requests_records = bool(re.search(r'回看|检索|对照|查找|找.*(?:记录|原话)|看看.*(?:记录|以前)', stripped))
        personal_statement = bool(re.search(r'我|对我来说|对我而言', stripped))
        if not is_question(stripped) and not requests_records and (
            current_stated(stripped) or personal_statement
            or (state.phase == 'awaiting_current_view' and len(compact) > 6)
        ):
            return TurnRoute('statement', state.topic_query)
    # A present-tense disclosure without a comparison request is a statement, not a failed search.
    if current_stated(stripped) and not re.search(r'回看|以前|过去|变化|改变|检索|对照', stripped):
        return TurnRoute('statement', '')
    return TurnRoute('lookup', stripped)


def _quote(text: str) -> str:
    # User-typed [A123] is literal text, never a database reference for the answer renderer.
    return re.sub(r'\[A(\d+)\]', r'［A\1］', text)


def conversation_answer(text: str, route: TurnRoute, previous: DialogueState | None) -> CognitiveAnswer:
    state = DialogueState(topic_query=route.query, current_statement=previous.current_statement if previous else None)
    recent = None
    if route.kind == 'pause':
        state.phase = 'paused'
        reply = '好，先到这里。等你想继续时再聊。'
    elif route.kind == 'uncertain':
        reply = '还没想好也可以，不用现在给它一个答案。先留在这里。'
    elif route.kind == 'acknowledge':
        state.phase = previous.phase if previous and previous.phase == 'paused' else 'reflecting'
        reply = '好，先留在这里。'
    else:
        state.current_statement = text
        recent = PositionEvidence(statement=text, status='user_stated_now', citations=[
            SourceCitation(title='这段对话中的用户补充', authorship='current_turn')
        ])
        reply = f'你补充的是：「{_quote(text)}」\n\n先按你此刻的表达来理解这件事，不急着把它归成观点变化，也不用马上解释原因。'
    return CognitiveAnswer(
        answer_type='conversation', summary=reply, topic=state.topic_query or None,
        recent_position=recent, dialogue=state,
    )

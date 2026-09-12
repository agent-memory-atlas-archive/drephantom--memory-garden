"""Small, conservative dialogue transitions for explicitly offline use.

Acknowledging a reply to our own question does not require searching the Vault again.
These rules keep user words verbatim; they do not infer feelings, causes or durable beliefs.
Full questions are not searches by default. Short topic-only inputs retain the
existing local-search shortcut; general discussion needs a connected model.
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
        or re.search(r'(?:我|你|这|那|它|现在).*?(?:怎么|怎样|如何|什么|多少|能不能|该不该)', text))


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


def _requests_lookup(text: str) -> bool:
    """Recognize requests to inspect records or compare one's own stated views.

    This is deliberately not a semantic classifier. Unknown full sentences go
    to conversation; a model, when connected, makes its own turn plan instead.
    """
    request = re.search(
        r'(?:^|[，,：:。])\s*(?:请(?:你)?\s*)?(?:(?:帮我|替我|给我|我想要?|能不能|可不可以)\s*)?'
        r'(?P<verb>回看|回顾|对照|检索|查找|搜索|查|找|翻|看看|读读)(?P<target>[^。]*)', text)
    if request and (request['verb'] in {'回看', '回顾', '对照', '检索'} or re.search(
            r'笔记|记录|原话|日记|写过|说过|以前|过去', request['target'])):
        return True
    if re.match(r'^把.+(?:笔记|记录|原话|日记).*(?:找|查|给我|读)', text):
        return True
    if re.match(r'^(?:如果|假如|假设)', text):
        return False
    change = r'(?:变过|变化|改变|变了|变得|不一样|不同|以前.*现在|过去.*现在)'
    own_view = re.search(
        r'(?:我的|我对[^，。！？?]{1,24}的)(?:想法|看法|观点|立场|态度)'
        r'[^，。！？?]*' + change, text)
    topic_comparison = re.search(r'(?:这个主题|这方面|这件事)[^，。！？?]*' + change, text)
    self_comparison = re.search(
        r'我(?:自己|本人)?(?:以前|过去|现在|后来)?(?:到现在)?'
        r'(?:为什么|怎么|是否|有没有|已经|真的)?(?:发生了)?' + change, text)
    discovery = re.search(r'(?:帮我)?看看有没有我自己没注意到的变化', text)
    if is_question(text) and (own_view or topic_comparison or self_comparison or discovery):
        return True
    return bool(is_question(text) and re.search(
        r'(?:我.{0,12}(?:以前|过去|曾经|当时)|(?:以前|过去|曾经|当时).{0,6}我)'
        r'.{0,20}(?:想法|看法|观点|看待|认为|怎么想|说过|写过)', text))


def _short_topic(text: str) -> bool:
    """Keep the documented 2–8-character topic shortcut, not sentence fallback."""
    return bool(re.fullmatch(r'[A-Za-z\u4e00-\u9fff]{2,8}', text) and not is_question(text)
                and not re.search(
                    r'^(?:我|你|他|她|它|请|帮|想|要|是|有|会|能)|'
                    r'如果|假如|假设|怎么|怎样|为什么|如何|能不能|会不会|聊|说', text))


def route_turn(text: str, state: DialogueState | None) -> TurnRoute:
    stripped = text.strip()
    compact = re.sub(r'[\s，,。.!！~～]', '', stripped)
    if re.fullmatch(r'(?:嗯|好|那)?(?:我)?(?:暂时不想说|先不聊(?:了|这个)?|不想聊(?:了|这个)?|'
                    r'先放着|先到这里|到这里就好|以后再说|暂时放着|先不说了)', compact):
        return TurnRoute('pause', state.topic_query if state else '')
    if compact in {'嗯', '嗯嗯', '好的', '好', '谢谢', '谢谢你', '是的', '对', '对的', '你好', '您好'}:
        return TurnRoute('acknowledge', state.topic_query if state else '')
    # An explicitly delimited topic is a lookup instruction, not part of the
    # retrieval query. Preserve its exact text for source search and memory keys.
    named_topic = re.fullmatch(r'回看「([^「」\r\n]+)」', stripped)
    if named_topic and named_topic.group(1).strip():
        return TurnRoute('lookup', named_topic.group(1))
    if state is not None and re.fullmatch(
            r'(?:那|再)?(?:以前呢|后来呢|还有呢|继续回看|看看以前的记录|回看旧记录|'
            r'继续看(?:看)?(?:以前的)?记录)[？?。！!]*', stripped):
        return TurnRoute('reference_lookup', state.topic_query or stripped)
    switch = re.match(r'^(?:换个话题|聊聊|说说|讨论一下)[，,：:\s]*(.+)', stripped)
    request = switch.group(1).strip() if switch else stripped
    if _requests_lookup(request):
        return TurnRoute('lookup', request)
    # A discussion invitation/conditional scenario remains conversation even
    # when the previous answer was waiting for a personal current-view reply.
    if switch or re.match(r'^(?:如果|假如|假设|我想(?:和你)?(?:聊|讨论|谈谈|问))', stripped):
        return TurnRoute('conversation', '')
    if state is not None:
        if compact in {'不知道', '我不知道', '说不清', '我也说不清', '还没想好', '暂时没有答案'}:
            return TurnRoute('uncertain', state.topic_query)
        personal_statement = bool(re.search(r'我|对我来说|对我而言', stripped))
        if not is_question(stripped) and (current_stated(stripped) or personal_statement):
            return TurnRoute('statement', state.topic_query)
    # A present-tense disclosure without a comparison request is a statement, not a failed search.
    if current_stated(stripped) and not re.search(r'回看|以前|过去|变化|改变|检索|对照', stripped):
        return TurnRoute('statement', '')
    if _short_topic(stripped):
        return TurnRoute('lookup', stripped)
    return TurnRoute('conversation', '')


def _quote(text: str) -> str:
    # User-typed [A123] is literal text, never a database reference for the answer renderer.
    return re.sub(r'\[A(\d+)\]', r'［A\1］', text)


def conversation_answer(text: str, route: TurnRoute, previous: DialogueState | None) -> CognitiveAnswer:
    continuing = route.kind != 'conversation'
    state = DialogueState(topic_query=route.query,
                          current_statement=previous.current_statement if previous and continuing else None)
    recent = None
    if route.kind == 'pause':
        state.phase = 'paused'
        reply = '好，先到这里。等你想继续时再聊。'
    elif route.kind == 'uncertain':
        reply = '还没想好也可以，不用现在给它一个答案。先留在这里。'
    elif route.kind == 'acknowledge':
        state.phase = previous.phase if previous and previous.phase == 'paused' else 'reflecting'
        reply = '好，先留在这里。'
    elif route.kind == 'statement':
        state.current_statement = text
        recent = PositionEvidence(statement=text, status='user_stated_now', citations=[
            SourceCitation(title='这段对话中的用户补充', authorship='current_turn')
        ])
        reply = f'你补充的是：「{_quote(text)}」\n\n先按你此刻的表达来理解这件事，不急着把它归成观点变化，也不用马上解释原因。'
    else:
        reply = '当前未连接生成模型，暂时无法展开这类讨论。连接模型后可以继续；你的消息会保留在这段对话里。'
    return CognitiveAnswer(
        answer_type='conversation', summary=reply, topic=state.topic_query or None,
        recent_position=recent, dialogue=state,
    )

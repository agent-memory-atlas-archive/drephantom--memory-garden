"""认知回溯答案契约：自然语言回答之外，强制附带可审计的结构化结果。

回答包括回溯结论、证据不足/澄清，以及不作结论的 conversation（回应、补充、暂停）。
关键约束：
- traced_change 必须同时具备早期端点与较近端点，且可定位引用；
- 较近记录永远标注 latest_memory_candidate，不冒充当前观点；
- 区间事件只标 within_interval，不自动构成因果；
- 每条回答最多一个问题；拒答（abstain）是一等公民。
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

ANSWER_TYPES = Literal[
    "traced_change", "no_clear_change", "insufficient_evidence", "clarification_needed", "conversation", "source_answer"
]


class SourceCitation(BaseModel):
    atom_id: int | None = None
    source_uid: str | None = None
    title: str = "未命名来源"
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    recorded_at: str | None = None
    event_time: str | None = None
    authorship: str = "user"
    sender: str = ""
    source_kind: str = "vault"
    excerpt: str | None = None


class PositionEvidence(BaseModel):
    statement: str
    status: Literal["source_grounded", "user_stated_now", "latest_memory_candidate"]
    citations: list[SourceCitation] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    statement: str
    relation: Literal["support", "challenge", "within_interval", "context"]
    citations: list[SourceCitation] = Field(default_factory=list)


class DialogueState(BaseModel):
    """只属于当前对话的状态；用户补充不自动升级为长期判定或 Vault 来源。"""
    topic_query: str = ""
    phase: Literal["awaiting_current_view", "reflecting", "paused"] = "reflecting"
    current_statement: str | None = None
    current_statement_message_id: int | None = None


class CognitiveAnswer(BaseModel):
    answer_type: ANSWER_TYPES
    summary: str
    topic: str | None = None
    early_position: PositionEvidence | None = None
    recent_position: PositionEvidence | None = None
    interval_events: list[EvidenceItem] = Field(default_factory=list)
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    counter_evidence: list[EvidenceItem] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    question_to_user: str | None = None
    citations: list[SourceCitation] = Field(default_factory=list)
    dialogue: DialogueState | None = None
    source_contract: Literal['model_grounded_v1'] | None = None

    @property
    def abstained(self) -> bool:
        return self.answer_type in {"insufficient_evidence", "clarification_needed"}


class VerdictChoice(StrEnum):
    accurate = "accurate"
    partly_accurate = "partly_accurate"
    no_change = "no_change"
    not_my_view = "not_my_view"
    insufficient_evidence = "insufficient_evidence"
    defer = "defer"


VERDICT_LABELS = {
    "accurate": "基本准确",
    "partly_accurate": "部分准确",
    "no_change": "不构成变化",
    "not_my_view": "不是我的观点",
    "insufficient_evidence": "证据不足",
    "defer": "暂时不判断",
}

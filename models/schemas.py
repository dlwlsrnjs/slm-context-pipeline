"""
SLM Minimal Sufficient Context Pipeline - Data Schemas
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class QuestionType(str, Enum):
    FACTOID = "factoid"
    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    MULTI_HOP = "multi-hop"
    AMBIGUOUS = "ambiguous"
    OPINION = "opinion"
    DEFINITION = "definition"


class ContextType(str, Enum):
    C_POS = "c_pos"        # 압축된 핵심 컨텍스트
    C_LONG = "c_long"      # 장황한 설명형 컨텍스트
    C_NOISY = "c_noisy"    # 관련·비관련 정보 혼합
    C_NULL = "c_null"      # 컨텍스트 없음
    C_LEAKY = "c_leaky"    # 답 누설 컨텍스트


@dataclass
class StructuredContext:
    """Teacher LLM이 생성하는 구조화된 컨텍스트"""
    need_context: bool
    question_type: QuestionType
    entities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    subquestions: list[str] = field(default_factory=list)
    useful_facts: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    answer_hint: str = ""
    
    def to_dict(self) -> dict:
        return {
            "need_context": self.need_context,
            "question_type": self.question_type.value if isinstance(self.question_type, QuestionType) else self.question_type,
            "entities": self.entities,
            "constraints": self.constraints,
            "subquestions": self.subquestions,
            "useful_facts": self.useful_facts,
            "missing_info": self.missing_info,
            "answer_hint": self.answer_hint
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StructuredContext":
        question_type = data.get("question_type", "factoid")
        if isinstance(question_type, str):
            try:
                question_type = QuestionType(question_type)
            except ValueError:
                question_type = QuestionType.FACTOID
        
        return cls(
            need_context=data.get("need_context", True),
            question_type=question_type,
            entities=data.get("entities", []),
            constraints=data.get("constraints", []),
            subquestions=data.get("subquestions", []),
            useful_facts=data.get("useful_facts", []),
            missing_info=data.get("missing_info", []),
            answer_hint=data.get("answer_hint", "")
        )


@dataclass
class PlannerOutput:
    """Stage 1: Planner 출력"""
    question_type: QuestionType
    need_external_context: bool
    entities: list[str]
    constraints: list[str]
    subquestions: list[str]
    retrieval_queries: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "question_type": self.question_type.value,
            "need_external_context": self.need_external_context,
            "entities": self.entities,
            "constraints": self.constraints,
            "subquestions": self.subquestions,
            "retrieval_queries": self.retrieval_queries
        }


@dataclass
class Evidence:
    """Stage 2: Evidence Builder가 수집한 증거"""
    source: str  # "retrieved" or "generated"
    content: str
    relevance_score: float = 0.0
    is_distractor: bool = False
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "is_distractor": self.is_distractor
        }


@dataclass
class JudgeOutput:
    """Stage 3: Judge 출력"""
    selected_facts: list[str]
    rejected_facts: list[str]
    information_density_score: float
    has_answer_leakage: bool
    distractor_ratio: float
    final_context: StructuredContext
    
    def to_dict(self) -> dict:
        return {
            "selected_facts": self.selected_facts,
            "rejected_facts": self.rejected_facts,
            "information_density_score": self.information_density_score,
            "has_answer_leakage": self.has_answer_leakage,
            "distractor_ratio": self.distractor_ratio,
            "final_context": self.final_context.to_dict()
        }


@dataclass
class ContextCandidate:
    """5종 컨텍스트 후보 중 하나"""
    question_id: str
    question: str
    context_type: ContextType
    context: StructuredContext
    raw_text: str = ""  # 직렬화된 컨텍스트 텍스트
    
    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "context_type": self.context_type.value,
            "context": self.context.to_dict(),
            "raw_text": self.raw_text
        }


@dataclass
class EvaluationResult:
    """다운스트림 평가 결과"""
    question_id: str
    context_type: ContextType
    accuracy: float
    answer_probability: float
    calibration_score: float
    token_count: int
    latency_ms: float
    
    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "context_type": self.context_type.value,
            "accuracy": self.accuracy,
            "answer_probability": self.answer_probability,
            "calibration_score": self.calibration_score,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms
        }


@dataclass
class PreferenceLabel:
    """학습용 preference label"""
    question_id: str
    question: str
    chosen_context: StructuredContext
    rejected_context: StructuredContext
    chosen_type: ContextType
    rejected_type: ContextType
    performance_delta: float
    
    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "chosen_context": self.chosen_context.to_dict(),
            "rejected_context": self.rejected_context.to_dict(),
            "chosen_type": self.chosen_type.value,
            "rejected_type": self.rejected_type.value,
            "performance_delta": self.performance_delta
        }


@dataclass
class TrainingExample:
    """SFT 학습용 예제"""
    question: str
    target_context: StructuredContext
    task_type: str  # "necessity", "extraction", "compression", "full_generation"
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "target_context": self.target_context.to_dict(),
            "task_type": self.task_type
        }

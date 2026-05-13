"""
Downstream Evaluator
각 컨텍스트 후보를 answer model에 넣어서 성능 측정 → utility 기반 preference label 생성
"""
import argparse
import logging
import re
import time
from pathlib import Path
from typing import Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, asdict

from tqdm import tqdm

from models.schemas import (
    ContextType, StructuredContext, PreferenceLabel
)
from utils.llm_client import create_llm_client, BaseLLMClient
from utils.file_utils import load_yaml, iter_jsonl, save_jsonl, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ANSWER_PROMPT_TEMPLATE = """You are answering a question. Use the provided context if available.

{context_section}

Question: {question}

Provide a concise, direct answer. If the context is insufficient or unclear, say \"I don't have enough information.\"

Answer:"""


@dataclass
class DownstreamEvalOutput:
    """Single evaluation output"""
    question_id: str
    question: str
    context_type: str
    context_raw: str
    model_answer: str
    ground_truth: Optional[str]
    is_correct: bool
    answer_probability: float
    calibration_score: float
    latency_ms: float
    token_count: int
    utility_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class DownstreamEvaluator:
    """Evaluate context candidates using an answer model"""

    def __init__(
        self,
        answer_fn: Callable[[str], str],
        config: Optional[dict] = None
    ):
        self.answer_fn = answer_fn
        self.config = config or {}
        self.alpha_correct = float(self.config.get("alpha_correct", 2.0))
        self.alpha_prob = float(self.config.get("alpha_prob", 1.0))
        self.beta_tokens = float(self.config.get("beta_tokens", 0.001))
        self.beta_latency = float(self.config.get("beta_latency", 0.0005))

    def evaluate_candidate(
        self,
        question_id: str,
        question: str,
        context_raw: str,
        context_type: str,
        ground_truth: Optional[str] = None
    ) -> DownstreamEvalOutput:
        """Evaluate a single context candidate"""

        if context_raw and context_raw.strip():
            context_section = f"Context:\n{context_raw}"
        else:
            context_section = "Context: (none provided)"

        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context_section=context_section,
            question=question
        )

        start_time = time.time()
        try:
            answer = self.answer_fn(prompt)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = "ERROR"
        latency_ms = (time.time() - start_time) * 1000

        token_count = len(prompt.split()) + len(answer.split())

        is_correct, answer_probability = self._check_correctness(answer, ground_truth)
        calibration_score = self._estimate_calibration(answer, ground_truth)

        utility_score = (
            self.alpha_correct * float(is_correct)
            + self.alpha_prob * answer_probability
            - self.beta_tokens * token_count
            - self.beta_latency * latency_ms
        )

        return DownstreamEvalOutput(
            question_id=question_id,
            question=question,
            context_type=context_type,
            context_raw=context_raw,
            model_answer=answer,
            ground_truth=ground_truth,
            is_correct=is_correct,
            answer_probability=answer_probability,
            calibration_score=calibration_score,
            latency_ms=latency_ms,
            token_count=token_count,
            utility_score=utility_score
        )

    def _check_correctness(
        self,
        model_answer: str,
        ground_truth: Optional[str]
    ) -> tuple[bool, float]:
        """Check correctness and estimate answer probability proxy"""
        if not ground_truth:
            return False, 0.0

        model_norm = self._normalize_text(model_answer)
        truth_norm = self._normalize_text(ground_truth)

        if not truth_norm:
            return False, 0.0

        if truth_norm in model_norm or model_norm in truth_norm:
            return True, 1.0

        truth_words = set(truth_norm.split())
        model_words = set(model_norm.split())
        overlap = 0.0
        if truth_words:
            overlap = len(truth_words & model_words) / len(truth_words)

        truth_nums = set(re.findall(r"-?\d+(?:\.\d+)?", truth_norm))
        model_nums = set(re.findall(r"-?\d+(?:\.\d+)?", model_norm))
        num_match = 0.0
        if truth_nums:
            num_match = len(truth_nums & model_nums) / len(truth_nums)

        score = max(overlap, num_match)
        return score >= 0.6, score

    def _estimate_calibration(self, model_answer: str, ground_truth: Optional[str]) -> float:
        """Simple calibration proxy based on abstention behavior"""
        answer_norm = self._normalize_text(model_answer)
        abstain_patterns = [
            "don't have enough information",
            "not enough information",
            "모르",
            "정보가 부족"
        ]
        abstained = any(pattern in answer_norm for pattern in abstain_patterns)

        if not ground_truth:
            return 1.0 if abstained else 0.5
        return 0.3 if abstained else 0.8

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text


def _safe_context_type(value: str) -> ContextType:
    try:
        return ContextType(value)
    except ValueError:
        return ContextType.C_POS


class LocalSLMResponder:
    """Run local HF SLM inference for utility evaluation"""

    def __init__(self, student_cfg: dict, eval_cfg: dict):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("transformers/torch are required for local SLM evaluation") from exc

        model_name = student_cfg.get("model_name", "Qwen/Qwen2.5-1.5B")
        self.max_new_tokens = int(eval_cfg.get("max_new_tokens", 128))
        self.temperature = float(eval_cfg.get("inference_temperature", 0.0))

        logger.info(f"Loading local SLM for evaluation: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float16 if eval_cfg.get("fp16", True) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=eval_cfg.get("device_map", "auto"),
            trust_remote_code=True
        )

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def create_answer_function(config: dict) -> Callable[[str], str]:
    """Create answer function from either API model or local student SLM"""
    eval_cfg = config.get("evaluation", {})
    backend = eval_cfg.get("backend", "api")

    if backend == "student_hf":
        local_responder = LocalSLMResponder(
            student_cfg=config.get("student", {}),
            eval_cfg=eval_cfg
        )
        return local_responder.generate

    answer_model = create_llm_client(config.get("answer_model", {}))
    return answer_model.generate


def generate_preference_labels(
    grouped: dict[str, list[dict]],
    min_delta: float = 0.05,
    null_tolerance: float = 0.03
) -> tuple[list[PreferenceLabel], dict]:
    """Generate preference labels from evaluation outputs"""
    labels: list[PreferenceLabel] = []
    stats = {
        "questions": len(grouped),
        "pairs": 0,
        "leaky_reject_pairs": 0,
        "need_context_false_candidates": 0,
    }

    for question_id, items in grouped.items():
        if len(items) < 2:
            continue

        non_leaky = [item for item in items if item["candidate"]["context_type"] != ContextType.C_LEAKY.value]
        leaky_items = [item for item in items if item["candidate"]["context_type"] == ContextType.C_LEAKY.value]

        if not non_leaky:
            continue

        ranked_non_leaky = sorted(
            non_leaky,
            key=lambda item: (item["eval"].is_correct, item["eval"].utility_score),
            reverse=True
        )
        chosen = ranked_non_leaky[0]

        for rejected in ranked_non_leaky[1:]:
            delta = chosen["eval"].utility_score - rejected["eval"].utility_score
            if delta < min_delta:
                continue

            labels.append(
                PreferenceLabel(
                    question_id=question_id,
                    question=chosen["eval"].question,
                    chosen_context=StructuredContext.from_dict(chosen["candidate"]["context"]),
                    rejected_context=StructuredContext.from_dict(rejected["candidate"]["context"]),
                    chosen_type=_safe_context_type(chosen["candidate"]["context_type"]),
                    rejected_type=_safe_context_type(rejected["candidate"]["context_type"]),
                    performance_delta=delta
                )
            )

        for leaky in leaky_items:
            delta = chosen["eval"].utility_score - leaky["eval"].utility_score
            labels.append(
                PreferenceLabel(
                    question_id=question_id,
                    question=chosen["eval"].question,
                    chosen_context=StructuredContext.from_dict(chosen["candidate"]["context"]),
                    rejected_context=StructuredContext.from_dict(leaky["candidate"]["context"]),
                    chosen_type=_safe_context_type(chosen["candidate"]["context_type"]),
                    rejected_type=ContextType.C_LEAKY,
                    performance_delta=delta
                )
            )
            stats["leaky_reject_pairs"] += 1

        null_items = [item for item in non_leaky if item["candidate"]["context_type"] == ContextType.C_NULL.value]
        non_null_items = [item for item in non_leaky if item["candidate"]["context_type"] != ContextType.C_NULL.value]
        if null_items and non_null_items:
            best_null = sorted(null_items, key=lambda item: item["eval"].utility_score, reverse=True)[0]
            best_non_null = sorted(non_null_items, key=lambda item: item["eval"].utility_score, reverse=True)[0]
            if (best_non_null["eval"].utility_score - best_null["eval"].utility_score) <= null_tolerance:
                stats["need_context_false_candidates"] += 1

    stats["pairs"] = len(labels)
    return labels, stats


def select_helpful_contexts(
    grouped: dict[str, list[dict]],
    null_tolerance: float = 0.03,
    min_probability: float = 0.2
) -> list[dict]:
    """Select contexts that demonstrably help SLM over null context"""
    helpful = []

    for question_id, items in grouped.items():
        non_leaky = [item for item in items if item["candidate"]["context_type"] != ContextType.C_LEAKY.value]
        if not non_leaky:
            continue

        non_null = [item for item in non_leaky if item["candidate"]["context_type"] != ContextType.C_NULL.value]
        if not non_null:
            continue

        best_non_null = sorted(
            non_null,
            key=lambda item: (item["eval"].is_correct, item["eval"].utility_score),
            reverse=True
        )[0]

        null_items = [item for item in non_leaky if item["candidate"]["context_type"] == ContextType.C_NULL.value]
        if null_items:
            best_null = sorted(null_items, key=lambda item: item["eval"].utility_score, reverse=True)[0]
            utility_gain = best_non_null["eval"].utility_score - best_null["eval"].utility_score
        else:
            utility_gain = best_non_null["eval"].utility_score

        if utility_gain <= null_tolerance:
            continue
        if best_non_null["eval"].answer_probability < min_probability:
            continue

        helpful.append(
            {
                "question_id": question_id,
                "question": best_non_null["eval"].question,
                "ground_truth_answer": best_non_null["eval"].ground_truth,
                "context_type": best_non_null["candidate"]["context_type"],
                "context": best_non_null["candidate"]["context"],
                "raw_text": best_non_null["candidate"].get("raw_text", ""),
                "utility_score": best_non_null["eval"].utility_score,
                "answer_probability": best_non_null["eval"].answer_probability,
                "is_correct": best_non_null["eval"].is_correct,
                "selection_reason": "utility_over_null"
            }
        )

    return helpful


def main():
    parser = argparse.ArgumentParser(description="Evaluate Context Candidates")
    parser.add_argument("--input", type=str, required=True, help="Candidates JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL with eval traces")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--min-delta", type=float, default=0.05, help="Minimum utility gap for preference pair")
    parser.add_argument("--null-tolerance", type=float, default=0.03, help="Tolerance to mark need_context=false candidate")
    parser.add_argument("--min-probability", type=float, default=0.2, help="Minimum answer probability proxy for keeping helpful context")
    args = parser.parse_args()

    config = load_yaml(args.config)
    answer_fn = create_answer_function(config)

    evaluator = DownstreamEvaluator(
        answer_fn,
        config=config.get("evaluation", {})
    )

    candidates_by_question: dict[str, list[dict]] = defaultdict(list)
    for record in iter_jsonl(args.input):
        qid = record["question_id"]
        candidates_by_question[qid].append(record)

    grouped_eval: dict[str, list[dict]] = defaultdict(list)
    all_outputs = []

    for qid, candidates in tqdm(candidates_by_question.items(), desc="Evaluating"):
        for candidate in candidates:
            result = evaluator.evaluate_candidate(
                question_id=qid,
                question=candidate["question"],
                context_raw=candidate.get("raw_text", ""),
                context_type=candidate["context_type"],
                ground_truth=candidate.get("ground_truth_answer")
            )
            grouped_eval[qid].append({"eval": result, "candidate": candidate})
            all_outputs.append(result.to_dict())

    preference_labels, stats = generate_preference_labels(
        grouped_eval,
        min_delta=args.min_delta,
        null_tolerance=args.null_tolerance
    )

    helpful_contexts = select_helpful_contexts(
        grouped_eval,
        null_tolerance=args.null_tolerance,
        min_probability=args.min_probability
    )

    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    save_jsonl(all_outputs, output_path)

    labels_path = output_path.parent / f"{output_path.stem}_preferences.jsonl"
    save_jsonl([label.to_dict() for label in preference_labels], labels_path)

    helpful_path = output_path.parent / f"{output_path.stem}_helpful_contexts.jsonl"
    save_jsonl(helpful_contexts, helpful_path)

    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    summary = {
        "total_evaluations": len(all_outputs),
        "total_questions": len(candidates_by_question),
        "preference_pairs": len(preference_labels),
        "helpful_contexts": len(helpful_contexts),
        **stats,
    }
    import json
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Evaluation results: {output_path}")
    print(f"Preference labels: {labels_path}")
    print(f"Helpful contexts: {helpful_path}")
    print(f"Summary: {summary_path}")
    print(f"Total evaluations: {len(all_outputs)}")
    print(f"Preference pairs: {len(preference_labels)}")


if __name__ == "__main__":
    main()

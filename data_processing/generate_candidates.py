"""
Context Candidate Generator
5종 컨텍스트 후보 생성:
- C_pos: 압축된 핵심 컨텍스트 (from Judge output)
- C_long: 장황한 설명형 컨텍스트
- C_noisy: 관련·비관련 정보 혼합
- C_null: 컨텍스트 없음
- C_leaky: 답 누설 컨텍스트
"""
import argparse
import logging
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from models.schemas import StructuredContext, ContextType, QuestionType, ContextCandidate
from utils.llm_client import create_llm_client, BaseLLMClient
from utils.file_utils import load_yaml, iter_jsonl, append_jsonl, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VERBOSE_CONTEXT_PROMPT = """You are generating a VERBOSE, DETAILED context for a question.
Unlike concise contexts, you should:
1. Include background information and history
2. Add related but tangential facts
3. Use flowing prose instead of bullet points
4. Be comprehensive rather than minimal

Question: {question}
Core facts to include: {useful_facts}

Generate a verbose paragraph (150-300 words) that includes all the core facts but adds substantial background context.
Output the paragraph only, no JSON."""


NOISY_CONTEXT_PROMPT = """You are generating a NOISY context that mixes relevant and irrelevant information.
You should:
1. Include some correct, relevant facts
2. Mix in some tangentially related but unhelpful facts
3. Add some completely irrelevant facts that might confuse a model
4. Make it hard to distinguish useful from useless information

Question: {question}
Some relevant facts: {useful_facts}

Generate a noisy paragraph (100-200 words) that includes distractors.
Output the paragraph only, no JSON."""


LEAKY_CONTEXT_PROMPT = """You are generating a context that REVEALS THE ANSWER too directly.
This is for training the model to recognize and reject such contexts.

Question: {question}
Answer: {answer}

Generate a short context (50-100 words) that essentially gives away the answer.
Make it obvious what the answer is from reading the context.
Output the paragraph only, no JSON."""


class CandidateGenerator:
    """Generate 5 types of context candidates for each question"""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: Optional[dict] = None
    ):
        self.llm = llm_client
        self.config = config or {}
        self.context_types = self.config.get("context_types", [
            "c_pos", "c_long", "c_noisy", "c_null", "c_leaky"
        ])
        self.use_llm_variants = self.config.get("use_llm_variants", True)
    
    def generate_all(
        self,
        question_id: str,
        question: str,
        teacher_output: dict,
        ground_truth_answer: Optional[str] = None
    ) -> list[ContextCandidate]:
        """Generate all types of context candidates"""
        candidates = []
        
        final_context = teacher_output.get("final_context", {})
        useful_facts = final_context.get("useful_facts", [])
        
        for ctx_type in self.context_types:
            try:
                candidate = self._generate_candidate(
                    ctx_type=ContextType(ctx_type),
                    question_id=question_id,
                    question=question,
                    final_context=final_context,
                    useful_facts=useful_facts,
                    ground_truth_answer=ground_truth_answer
                )
                candidates.append(candidate)
            except Exception as e:
                logger.error(f"Failed to generate {ctx_type} for {question_id}: {e}")
        
        return candidates
    
    def _generate_candidate(
        self,
        ctx_type: ContextType,
        question_id: str,
        question: str,
        final_context: dict,
        useful_facts: list[str],
        ground_truth_answer: Optional[str]
    ) -> ContextCandidate:
        """Generate a single candidate of the specified type"""
        
        if ctx_type == ContextType.C_POS:
            # Use the judge's output directly
            context = StructuredContext.from_dict(final_context)
            raw_text = self._serialize_context(context)
        
        elif ctx_type == ContextType.C_LONG:
            # Generate verbose context
            raw_text = self._generate_verbose(question, useful_facts)
            context = StructuredContext(
                need_context=True,
                question_type=QuestionType(final_context.get("question_type", "factoid")),
                entities=final_context.get("entities", []),
                constraints=final_context.get("constraints", []),
                useful_facts=[raw_text],  # Store as single verbose fact
                answer_hint=final_context.get("answer_hint", "")
            )
        
        elif ctx_type == ContextType.C_NOISY:
            # Generate noisy context
            raw_text = self._generate_noisy(question, useful_facts)
            context = StructuredContext(
                need_context=True,
                question_type=QuestionType(final_context.get("question_type", "factoid")),
                entities=final_context.get("entities", []),
                constraints=final_context.get("constraints", []),
                useful_facts=[raw_text],
                answer_hint=""
            )
        
        elif ctx_type == ContextType.C_NULL:
            # Empty context
            context = StructuredContext(
                need_context=False,
                question_type=QuestionType(final_context.get("question_type", "factoid")),
                entities=[],
                constraints=[],
                useful_facts=[],
                answer_hint=""
            )
            raw_text = ""
        
        elif ctx_type == ContextType.C_LEAKY:
            # Generate answer-leaking context
            if ground_truth_answer:
                raw_text = self._generate_leaky(question, ground_truth_answer)
            else:
                # Fallback: use hints that are too specific
                raw_text = f"The answer to '{question}' can be directly found by looking at: {', '.join(useful_facts)}"
            
            context = StructuredContext(
                need_context=True,
                question_type=QuestionType(final_context.get("question_type", "factoid")),
                entities=final_context.get("entities", []),
                constraints=final_context.get("constraints", []),
                useful_facts=[raw_text],
                answer_hint="Answer is directly stated above"  # Obvious leakage marker
            )
        
        else:
            raise ValueError(f"Unknown context type: {ctx_type}")
        
        return ContextCandidate(
            question_id=question_id,
            question=question,
            context_type=ctx_type,
            context=context,
            raw_text=raw_text
        )
    
    def _serialize_context(self, context: StructuredContext) -> str:
        """Convert structured context to raw text for SLM input"""
        parts = []
        
        if context.entities:
            parts.append(f"Entities: {', '.join(context.entities)}")
        
        if context.constraints:
            parts.append(f"Constraints: {', '.join(context.constraints)}")
        
        if context.useful_facts:
            parts.append("Facts:")
            for fact in context.useful_facts:
                parts.append(f"- {fact}")
        
        if context.answer_hint:
            parts.append(f"Hint: {context.answer_hint}")
        
        return "\n".join(parts)
    
    def _generate_verbose(self, question: str, useful_facts: list[str]) -> str:
        """Generate verbose/long context"""
        if not self.use_llm_variants:
            if not useful_facts:
                return "This topic includes multiple background considerations and related context."
            joined = " ".join(useful_facts)
            return (
                f"To understand the question, consider the broader background first. {joined} "
                f"In addition, historical context, related concepts, and surrounding details can be useful "
                f"before deciding the final answer."
            )

        prompt = VERBOSE_CONTEXT_PROMPT.format(
            question=question,
            useful_facts="\n".join(f"- {f}" for f in useful_facts)
        )
        
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Verbose generation failed: {e}")
            # Fallback: just join facts with filler
            return f"Background information: {' '.join(useful_facts)}. " + \
                   "This information is relevant to understanding the question and its context."
    
    def _generate_noisy(self, question: str, useful_facts: list[str]) -> str:
        """Generate noisy context with distractors"""
        if not self.use_llm_variants:
            noise = [
                "Some people discuss this from unrelated cultural perspectives.",
                "There are debates that do not directly affect this specific answer.",
                "General background may sound relevant but often does not change the result.",
            ]
            mixed = list(useful_facts)
            mixed.extend(random.sample(noise, min(2, len(noise))))
            random.shuffle(mixed)
            return " ".join(mixed) if mixed else "This contains mixed relevant and irrelevant information."

        prompt = NOISY_CONTEXT_PROMPT.format(
            question=question,
            useful_facts="\n".join(f"- {f}" for f in useful_facts)
        )
        
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Noisy generation failed: {e}")
            # Fallback: add random noise
            noise = [
                "This is additional background that may or may not be relevant.",
                "Some say this topic is complicated.",
                "There are many perspectives on this issue."
            ]
            mixed = useful_facts + random.sample(noise, min(2, len(noise)))
            random.shuffle(mixed)
            return " ".join(mixed)
    
    def _generate_leaky(self, question: str, answer: str) -> str:
        """Generate context that leaks the answer"""
        if not self.use_llm_variants:
            return f"Direct answer leakage: The correct answer to the question is {answer}."

        prompt = LEAKY_CONTEXT_PROMPT.format(
            question=question,
            answer=answer
        )
        
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Leaky generation failed: {e}")
            # Fallback: directly state the answer
            return f"The answer is {answer}. This is the correct response to the question."


def main():
    parser = argparse.ArgumentParser(description="Generate Context Candidates")
    parser.add_argument("--input", type=str, required=True, help="Teacher output JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    llm_client = create_llm_client(config.get("teacher", {}))
    
    generator = CandidateGenerator(
        llm_client,
        config=config.get("candidate_generation", {})
    )
    
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    
    for record in tqdm(iter_jsonl(args.input), desc="Generating candidates"):
        if not record.get("success", False):
            continue
        
        candidates = generator.generate_all(
            question_id=record["question_id"],
            question=record["question"],
            teacher_output=record,
            ground_truth_answer=record.get("ground_truth_answer")
        )
        
        for candidate in candidates:
            candidate_record = candidate.to_dict()
            candidate_record["ground_truth_answer"] = record.get("ground_truth_answer")
            candidate_record["teacher_success"] = record.get("success", False)
            append_jsonl(candidate_record, output_path)
    
    print(f"Candidates saved to {output_path}")


if __name__ == "__main__":
    main()

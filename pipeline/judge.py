"""
Stage 3: Judge
- 정보 밀도 평가
- distractor 탐지
- answer leakage 탐지
- 최종 컨텍스트 선별
"""
from typing import Optional
import logging

from models.schemas import (
    PlannerOutput, Evidence, JudgeOutput, StructuredContext, QuestionType
)
from pipeline.evidence_builder import EvidenceBuilderOutput
from utils.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """You are a context quality judge for small language models (SLMs).

Your job is to evaluate and filter evidence to create the MINIMUM SUFFICIENT CONTEXT that helps an SLM answer correctly.

Evaluation criteria:
1. RELEVANCE: Is this fact directly needed to answer the question?
2. INFORMATION DENSITY: Does this add new, useful information vs token cost?
3. DISTRACTOR CHECK: Is this tangentially related but actually unhelpful?
4. LEAKAGE CHECK: Does this reveal the answer too directly?

Output JSON format:
{
    "selected_facts": ["fact 1 that passed all checks", "fact 2", ...],
    "rejected_facts": ["rejected fact 1", ...],
    "rejection_reasons": {"rejected fact 1": "reason", ...},
    "information_density_score": 0.0-1.0,
    "has_answer_leakage": true/false,
    "distractor_ratio": 0.0-1.0,
    "final_context": {
        "need_context": true/false,
        "question_type": "factoid|procedural|comparison|multi-hop|ambiguous",
        "entities": [...],
        "constraints": [...],
        "subquestions": [...],
        "useful_facts": [...],
        "missing_info": [...],
        "answer_hint": "1-2 sentence reasoning direction WITHOUT revealing the answer"
    }
}

Rules for selected_facts:
1. Keep ONLY facts that change the answer
2. Prefer concrete over generic
3. Maximum 5 facts for simple questions, 8 for complex ones
4. Each fact should be atomic (one piece of information)

Rules for answer_hint:
1. Provide reasoning DIRECTION, not the answer
2. Example good hint: "Compare the 2023 sales figures for both companies"
3. Example bad hint: "BYD sold more than Tesla" (this leaks the answer!)

LANGUAGE REQUIREMENT:
- All output text must be in English only.
- selected_facts, rejected_facts, rejection_reasons, useful_facts, missing_info, and answer_hint must be English.
- Never output Korean or mixed-language strings."""


JUDGE_FEW_SHOT = """
Example 1:
Question: "Which company sold more electric vehicles in 2023, Tesla or BYD?"
Evidence:
- "Tesla is a U.S. EV manufacturer founded in 2003."
- "BYD sold about 3 million electric vehicles in 2023."
- "Tesla sold about 1.8 million vehicles in 2023."
- "Electric vehicles are environmentally friendly transportation."
- "BYD also manufactures batteries."

{
    "selected_facts": [
        "BYD sold about 3 million electric vehicles in 2023.",
        "Tesla sold about 1.8 million vehicles in 2023."
    ],
    "rejected_facts": [
        "Tesla is a U.S. EV manufacturer founded in 2003.",
        "Electric vehicles are environmentally friendly transportation.",
        "BYD also manufactures batteries."
    ],
    "rejection_reasons": {
        "Tesla is a U.S. EV manufacturer founded in 2003.": "Does not affect the comparison",
        "Electric vehicles are environmentally friendly transportation.": "Generic, distractor",
        "BYD also manufactures batteries.": "Irrelevant to sales comparison"
    },
    "information_density_score": 0.85,
    "has_answer_leakage": false,
    "distractor_ratio": 0.6,
    "final_context": {
        "need_context": true,
        "question_type": "comparison",
        "entities": ["Tesla", "BYD"],
        "constraints": ["time=2023", "metric=EV sales"],
        "subquestions": [],
        "useful_facts": [
            "BYD 2023 EV sales: about 3 million",
            "Tesla 2023 EV sales: about 1.8 million"
        ],
        "missing_info": [],
        "answer_hint": "Compare the 2023 EV sales figures for both companies."
    }
}

Example 2:
Question: "What is the Pythagorean theorem?"
Evidence:
- "The Pythagorean theorem is a² + b² = c²."
- "Pythagoras was an ancient Greek mathematician."
- "In a right triangle, c denotes the hypotenuse."

{
    "selected_facts": [
        "The Pythagorean theorem is a² + b² = c².",
        "In a right triangle, c denotes the hypotenuse."
    ],
    "rejected_facts": [
        "Pythagoras was an ancient Greek mathematician."
    ],
    "rejection_reasons": {
        "Pythagoras was an ancient Greek mathematician.": "Historical trivia, not needed for definition"
    },
    "information_density_score": 0.9,
    "has_answer_leakage": true,
    "distractor_ratio": 0.33,
    "final_context": {
        "need_context": false,
        "question_type": "definition",
        "entities": ["Pythagorean theorem", "right triangle", "hypotenuse"],
        "constraints": [],
        "subquestions": [],
        "useful_facts": [
            "Formula for right triangles",
            "a² + b² = c² where c is the hypotenuse"
        ],
        "missing_info": [],
        "answer_hint": "Describe the relationship among the three sides of a right triangle."
    }
}
"""


class Judge:
    """Stage 3: Context Quality Evaluation and Filtering"""
    
    def __init__(self, llm_client: BaseLLMClient, config: Optional[dict] = None):
        self.llm = llm_client
        self.config = config or {}
        self.min_information_density = self.config.get("min_information_density", 0.5)
        self.max_distractor_ratio = self.config.get("max_distractor_ratio", 0.3)
        self.leakage_detection = self.config.get("leakage_detection", True)
    
    def judge(
        self,
        question: str,
        planner_output: PlannerOutput,
        evidence_output: EvidenceBuilderOutput
    ) -> JudgeOutput:
        """Evaluate and filter evidence to produce final context"""
        
        # Format evidence for LLM
        evidence_text = self._format_evidence(evidence_output)
        
        prompt = f"""{JUDGE_FEW_SHOT}

Now judge this:
Question: "{question}"
Question Analysis: {planner_output.to_dict()}
Evidence:
{evidence_text}

Output JSON only (no code blocks):"""
        
        try:
            result = self.llm.generate_json(prompt, JUDGE_SYSTEM_PROMPT)
            return self._parse_result(result, planner_output)
        except Exception as e:
            logger.error(f"Judge failed: {e}")
            return self._default_output(planner_output, evidence_output)
    
    def _format_evidence(self, evidence_output: EvidenceBuilderOutput) -> str:
        """Format evidence list for LLM prompt"""
        lines = []
        for ev in evidence_output.combined_evidence:
            lines.append(f"- \"{ev.content}\"")
        
        if evidence_output.pseudo_document:
            lines.append(f"\nPseudo-document: \"{evidence_output.pseudo_document}\"")
        
        return "\n".join(lines)
    
    def _parse_result(self, result: dict, planner_output: PlannerOutput) -> JudgeOutput:
        """Parse LLM response into JudgeOutput"""
        final_context_data = result.get("final_context", {})
        
        # Parse question type
        q_type_str = final_context_data.get("question_type", planner_output.question_type.value)
        try:
            q_type = QuestionType(q_type_str)
        except ValueError:
            q_type = planner_output.question_type
        
        final_context = StructuredContext(
            need_context=final_context_data.get("need_context", True),
            question_type=q_type,
            entities=final_context_data.get("entities", planner_output.entities),
            constraints=final_context_data.get("constraints", planner_output.constraints),
            subquestions=final_context_data.get("subquestions", planner_output.subquestions),
            useful_facts=final_context_data.get("useful_facts", []),
            missing_info=final_context_data.get("missing_info", []),
            answer_hint=final_context_data.get("answer_hint", "")
        )
        
        return JudgeOutput(
            selected_facts=result.get("selected_facts", []),
            rejected_facts=result.get("rejected_facts", []),
            information_density_score=result.get("information_density_score", 0.5),
            has_answer_leakage=result.get("has_answer_leakage", False),
            distractor_ratio=result.get("distractor_ratio", 0.0),
            final_context=final_context
        )
    
    def _default_output(
        self,
        planner_output: PlannerOutput,
        evidence_output: EvidenceBuilderOutput
    ) -> JudgeOutput:
        """Return default output when judgment fails"""
        # Use top evidence as useful facts
        useful_facts = [
            ev.content for ev in evidence_output.combined_evidence[:5]
        ]
        
        final_context = StructuredContext(
            need_context=planner_output.need_external_context,
            question_type=planner_output.question_type,
            entities=planner_output.entities,
            constraints=planner_output.constraints,
            subquestions=planner_output.subquestions,
            useful_facts=useful_facts,
            missing_info=[],
            answer_hint=""
        )
        
        return JudgeOutput(
            selected_facts=useful_facts,
            rejected_facts=[],
            information_density_score=0.5,
            has_answer_leakage=False,
            distractor_ratio=0.0,
            final_context=final_context
        )
    
    def passes_quality_threshold(self, output: JudgeOutput) -> bool:
        """Check if the judge output passes quality thresholds"""
        if output.information_density_score < self.min_information_density:
            return False
        if output.distractor_ratio > self.max_distractor_ratio:
            return False
        if self.leakage_detection and output.has_answer_leakage:
            return False
        return True


if __name__ == "__main__":
    from utils.llm_client import MockLLMClient
    from pipeline.planner import Planner
    from pipeline.evidence_builder import EvidenceBuilder
    
    mock_client = MockLLMClient()
    planner = Planner(mock_client)
    evidence_builder = EvidenceBuilder(mock_client)
    judge = Judge(mock_client)
    
    question = "Tesla와 BYD 중 2023년에 더 많은 전기차를 판매한 회사는?"
    
    planner_output = planner.analyze(question)
    evidence_output = evidence_builder.build(question, planner_output)
    judge_output = judge.judge(question, planner_output, evidence_output)
    
    print(f"Question: {question}")
    print(f"Final context: {judge_output.final_context.to_dict()}")
    print(f"Passes threshold: {judge.passes_quality_threshold(judge_output)}")

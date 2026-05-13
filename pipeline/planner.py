"""
Stage 1: Planner - 질문 해부
- 질문 유형 분류
- external context 필요 여부 판단
- 핵심 엔티티/제약 추출
- 서브질문 분해
"""
from typing import Optional
import logging

from models.schemas import PlannerOutput, QuestionType
from utils.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """You are a question analysis expert. Your job is to dissect questions to help a small language model (SLM) understand what information it needs.

You must output JSON only. No explanations, no markdown code blocks around it.

Output format:
{
    "question_type": "factoid|procedural|comparison|multi-hop|ambiguous|opinion|definition",
    "need_external_context": true/false,
    "entities": ["list of key entities mentioned or implied"],
    "constraints": ["time constraints", "region constraints", "numerical limits", "exclusions", etc.],
    "subquestions": ["if multi-hop, list sub-questions to answer first"],
    "retrieval_queries": ["suggested search queries if retrieval needed"]
}

Guidelines:
1. question_type: Choose the most appropriate type
   - factoid: Simple fact lookup (who, what, when, where)
   - procedural: How to do something
   - comparison: Comparing two or more things
   - multi-hop: Requires multiple reasoning steps
   - ambiguous: Question is unclear or underspecified
   - opinion: Subjective question
   - definition: What is X

2. need_external_context: 
   - false: Common knowledge, simple definitions, basic math
   - true: Recent events, specific data, domain expertise, comparisons

3. entities: Extract ALL relevant named entities, concepts, technical terms

4. constraints: Extract explicit and implicit constraints
   - Time: "in 2024", "recently", "during WWII"
   - Region: "in Korea", "in Europe"
   - Numerical: "top 5", "under $100", "at least 3"
   - Scope: "excluding X", "only for Y"

5. subquestions: For complex questions, break down into simpler parts

6. retrieval_queries: If context is needed, suggest 1-3 search queries

7. LANGUAGE REQUIREMENT: All generated string values in JSON must be in English only.
    - entities, constraints, subquestions, retrieval_queries must be English.
    - Never output Korean or mixed-language strings.
    - Keep terminology concise and machine-readable.
"""


PLANNER_FEW_SHOT = """
Example 1:
Question: "What was South Korea's GDP growth rate in 2024?"
{
    "question_type": "factoid",
    "need_external_context": true,
    "entities": ["South Korea", "GDP growth rate"],
    "constraints": ["time=2024", "region=South Korea"],
    "subquestions": [],
    "retrieval_queries": ["South Korea GDP growth rate 2024", "Korea GDP growth 2024"]
}

Example 2:
Question: "What is the Pythagorean theorem?"
{
    "question_type": "definition",
    "need_external_context": false,
    "entities": ["Pythagorean theorem", "right triangle"],
    "constraints": [],
    "subquestions": [],
    "retrieval_queries": []
}

Example 3:
Question: "Which company sold more electric vehicles, Tesla or BYD?"
{
    "question_type": "comparison",
    "need_external_context": true,
    "entities": ["Tesla", "BYD", "electric vehicles", "sales"],
    "constraints": ["comparison_axis=EV sales"],
    "subquestions": ["What was Tesla's EV sales figure?", "What was BYD's EV sales figure?"],
    "retrieval_queries": ["Tesla EV sales", "BYD EV sales", "EV sales comparison Tesla BYD"]
}

Example 4:
Question: "What is a good programming language?"
{
    "question_type": "ambiguous",
    "need_external_context": false,
    "entities": ["programming language"],
    "constraints": [],
    "subquestions": ["What is the target use case?", "Is this for beginners or experts?"],
    "retrieval_queries": []
}
"""


class Planner:
    """Stage 1: Question Dissection"""
    
    def __init__(self, llm_client: BaseLLMClient, config: Optional[dict] = None):
        self.llm = llm_client
        self.config = config or {}
        self.max_entities = self.config.get("max_entities", 10)
        self.max_constraints = self.config.get("max_constraints", 8)
        self.max_subquestions = self.config.get("max_subquestions", 5)
    
    def analyze(self, question: str) -> PlannerOutput:
        """Analyze a question and extract structured information"""
        prompt = f"""{PLANNER_FEW_SHOT}

Now analyze this question:
Question: "{question}"

Output JSON only:"""
        
        try:
            result = self.llm.generate_json(prompt, PLANNER_SYSTEM_PROMPT)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Planner failed for question: {question}, error: {e}")
            return self._default_output(question)
    
    def _parse_result(self, result: dict) -> PlannerOutput:
        """Parse LLM response into PlannerOutput"""
        question_type_str = result.get("question_type", "factoid")
        try:
            question_type = QuestionType(question_type_str)
        except ValueError:
            question_type = QuestionType.FACTOID
        
        entities = result.get("entities", [])[:self.max_entities]
        constraints = result.get("constraints", [])[:self.max_constraints]
        subquestions = result.get("subquestions", [])[:self.max_subquestions]
        
        return PlannerOutput(
            question_type=question_type,
            need_external_context=result.get("need_external_context", True),
            entities=entities,
            constraints=constraints,
            subquestions=subquestions,
            retrieval_queries=result.get("retrieval_queries", [])
        )
    
    def _default_output(self, question: str) -> PlannerOutput:
        """Return default output when analysis fails"""
        return PlannerOutput(
            question_type=QuestionType.FACTOID,
            need_external_context=True,
            entities=[],
            constraints=[],
            subquestions=[],
            retrieval_queries=[question]
        )
    
    def batch_analyze(self, questions: list[str]) -> list[PlannerOutput]:
        """Analyze multiple questions"""
        results = []
        for question in questions:
            result = self.analyze(question)
            results.append(result)
        return results


if __name__ == "__main__":
    # Test with mock client
    from utils.llm_client import MockLLMClient
    
    mock_client = MockLLMClient()
    planner = Planner(mock_client)
    
    test_questions = [
        "2024년 한국의 GDP 성장률은?",
        "피타고라스 정리가 뭐야?",
        "Tesla vs BYD 누가 더 잘 팔아?"
    ]
    
    for q in test_questions:
        result = planner.analyze(q)
        print(f"Q: {q}")
        print(f"Result: {result.to_dict()}")
        print()

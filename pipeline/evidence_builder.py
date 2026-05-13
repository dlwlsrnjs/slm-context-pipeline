"""
Stage 2: Evidence Builder
- Retrieved evidence: 실제 문서/지식원에서 가져온 정보
- Generated pseudo-context: teacher가 생성한 배경문 (Query2doc, GenRead 방식)
- COMBO 방식으로 두 소스를 결합
"""
from typing import Optional
from dataclasses import dataclass
import logging

from models.schemas import PlannerOutput, Evidence
from utils.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


EVIDENCE_GENERATION_SYSTEM_PROMPT = """You are an evidence generation expert. Given a question and its analysis, generate relevant factual evidence that would help answer the question.

Rules:
1. Generate FACTUAL information only - no speculation
2. Keep each fact atomic and concise (1-2 sentences max)
3. Do NOT include the answer directly
4. Focus on background facts that enable reasoning toward the answer
5. Include relevant definitions, relationships, and context
6. If the question involves comparison, provide facts about each entity separately
7. LANGUAGE REQUIREMENT: All output text must be in English only.
    - generated_evidence[].content must be English.
    - pseudo_document must be English.
    - Never output Korean or mixed-language content.

Output JSON format:
{
    "generated_evidence": [
        {"content": "fact 1", "relevance_score": 0.9},
        {"content": "fact 2", "relevance_score": 0.8},
        ...
    ],
    "pseudo_document": "A brief paragraph combining key facts (max 200 words)"
}"""


EVIDENCE_GENERATION_FEW_SHOT = """
Example 1:
Question: "Which company sold more electric vehicles in 2023, Tesla or BYD?"
Analysis: {"question_type": "comparison", "entities": ["Tesla", "BYD"], "constraints": ["time=2023"]}

{
    "generated_evidence": [
        {"content": "Tesla is a U.S. electric vehicle manufacturer founded in 2003 and sells models such as Model 3 and Model Y.", "relevance_score": 0.7},
        {"content": "BYD is a Chinese EV and battery manufacturer and became one of the global EV sales leaders in 2023.", "relevance_score": 0.8},
        {"content": "Global EV market share for Chinese brands increased significantly in 2023.", "relevance_score": 0.6},
        {"content": "EV sales comparisons may differ depending on whether BEV-only or BEV+PHEV counts are used.", "relevance_score": 0.7}
    ],
    "pseudo_document": "Tesla and BYD are major players in the global EV market. Tesla primarily sells battery electric vehicles such as Model 3 and Model Y, while BYD sells both battery electric and plug-in hybrid vehicles. The 2023 EV market showed strong growth from Chinese brands."
}

Example 2:
Question: "What is the Pythagorean theorem?"
Analysis: {"question_type": "definition", "need_external_context": false}

{
    "generated_evidence": [
        {"content": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.", "relevance_score": 0.95},
        {"content": "It is written as a² + b² = c², where c is the hypotenuse.", "relevance_score": 0.9},
        {"content": "Although named after Pythagoras, related forms of the theorem were known earlier in other civilizations.", "relevance_score": 0.6}
    ],
    "pseudo_document": "The Pythagorean theorem is a geometric relationship among the sides of a right triangle. The square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c²."
}
"""


@dataclass
class EvidenceBuilderOutput:
    """Evidence Builder의 출력"""
    retrieved_evidence: list[Evidence]
    generated_evidence: list[Evidence]
    pseudo_document: str
    combined_evidence: list[Evidence]
    
    def to_dict(self) -> dict:
        return {
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "generated_evidence": [e.to_dict() for e in self.generated_evidence],
            "pseudo_document": self.pseudo_document,
            "combined_evidence": [e.to_dict() for e in self.combined_evidence]
        }


class DummyRetriever:
    """Placeholder retriever - 실제 구현에서는 BM25 또는 Dense Retriever로 교체"""
    
    def retrieve(self, queries: list[str], top_k: int = 5) -> list[Evidence]:
        """Return empty results - implement actual retrieval logic"""
        logger.warning("Using dummy retriever - no actual retrieval performed")
        return []


class EvidenceBuilder:
    """Stage 2: Evidence Collection and Generation"""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: Optional[DummyRetriever] = None,
        config: Optional[dict] = None
    ):
        self.llm = llm_client
        self.retriever = retriever or DummyRetriever()
        self.config = config or {}
        self.use_retrieval = self.config.get("use_retrieval", False)
        self.use_generation = self.config.get("use_generation", True)
        self.max_evidences = self.config.get("max_evidences", 10)
        self.pseudo_doc_length = self.config.get("pseudo_doc_length", 200)
    
    def build(self, question: str, planner_output: PlannerOutput) -> EvidenceBuilderOutput:
        """Build evidence from both retrieval and generation"""
        retrieved_evidence = []
        generated_evidence = []
        pseudo_document = ""
        
        # Step 1: Retrieval (if enabled and needed)
        if self.use_retrieval and planner_output.need_external_context:
            queries = planner_output.retrieval_queries or [question]
            retrieved_evidence = self.retriever.retrieve(queries, top_k=self.max_evidences)
        
        # Step 2: Generation (Query2doc / GenRead style)
        if self.use_generation:
            gen_result = self._generate_evidence(question, planner_output)
            generated_evidence = gen_result["evidence"]
            pseudo_document = gen_result["pseudo_document"]
        
        # Step 3: Combine (COMBO style)
        combined = self._combine_evidence(retrieved_evidence, generated_evidence)
        
        return EvidenceBuilderOutput(
            retrieved_evidence=retrieved_evidence,
            generated_evidence=generated_evidence,
            pseudo_document=pseudo_document,
            combined_evidence=combined
        )
    
    def _generate_evidence(self, question: str, planner_output: PlannerOutput) -> dict:
        """Generate pseudo-evidence using teacher LLM"""
        prompt = f"""{EVIDENCE_GENERATION_FEW_SHOT}

Now generate evidence for:
Question: "{question}"
Analysis: {planner_output.to_dict()}

Output JSON only (no code blocks):"""
        
        try:
            result = self.llm.generate_json(prompt, EVIDENCE_GENERATION_SYSTEM_PROMPT)
            
            evidence_list = []
            for item in result.get("generated_evidence", []):
                evidence_list.append(Evidence(
                    source="generated",
                    content=item.get("content", ""),
                    relevance_score=item.get("relevance_score", 0.5),
                    is_distractor=False
                ))
            
            return {
                "evidence": evidence_list[:self.max_evidences],
                "pseudo_document": result.get("pseudo_document", "")[:self.pseudo_doc_length * 2]
            }
        except Exception as e:
            logger.error(f"Evidence generation failed: {e}")
            return {"evidence": [], "pseudo_document": ""}
    
    def _combine_evidence(
        self,
        retrieved: list[Evidence],
        generated: list[Evidence]
    ) -> list[Evidence]:
        """Combine and deduplicate evidence from both sources"""
        # Simple combination - in production, use more sophisticated deduplication
        combined = []
        seen_content = set()
        
        # Prioritize retrieved evidence (usually more reliable)
        for ev in retrieved:
            content_key = ev.content.lower().strip()[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                combined.append(ev)
        
        # Add generated evidence
        for ev in generated:
            content_key = ev.content.lower().strip()[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                combined.append(ev)
        
        # Sort by relevance score
        combined.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return combined[:self.max_evidences]


if __name__ == "__main__":
    from utils.llm_client import MockLLMClient
    from pipeline.planner import Planner
    
    mock_client = MockLLMClient()
    planner = Planner(mock_client)
    evidence_builder = EvidenceBuilder(mock_client, config={"use_generation": True})
    
    question = "Tesla와 BYD 중 2023년에 더 많은 전기차를 판매한 회사는?"
    planner_output = planner.analyze(question)
    
    result = evidence_builder.build(question, planner_output)
    print(f"Question: {question}")
    print(f"Evidence count: {len(result.combined_evidence)}")
    print(f"Pseudo document: {result.pseudo_document[:100]}...")

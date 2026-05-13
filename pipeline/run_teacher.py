"""
Teacher Pipeline - 3단계 파이프라인 실행
Planner → Evidence Builder → Judge
"""
import argparse
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from tqdm import tqdm

from pipeline.planner import Planner
from pipeline.evidence_builder import EvidenceBuilder
from pipeline.judge import Judge
from utils.llm_client import create_llm_client, BaseLLMClient
from utils.file_utils import load_yaml, iter_jsonl, append_jsonl, ensure_dir
from models.schemas import StructuredContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TeacherPipelineOutput:
    """전체 파이프라인 출력"""
    question_id: str
    question: str
    ground_truth_answer: Optional[str]
    planner_output: dict
    evidence_output: dict
    judge_output: dict
    final_context: dict
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


class TeacherPipeline:
    """3-Stage Teacher Pipeline"""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: dict
    ):
        self.config = config
        pipeline_config = config.get("pipeline", {})
        
        self.planner = Planner(
            llm_client,
            config=pipeline_config.get("planner", {})
        )
        self.evidence_builder = EvidenceBuilder(
            llm_client,
            config=pipeline_config.get("evidence_builder", {})
        )
        self.judge = Judge(
            llm_client,
            config=pipeline_config.get("judge", {})
        )
    
    def process(
        self,
        question_id: str,
        question: str,
        ground_truth_answer: Optional[str] = None
    ) -> TeacherPipelineOutput:
        """Process a single question through the full pipeline"""
        try:
            # Stage 1: Planner
            logger.debug(f"Processing question: {question[:50]}...")
            planner_output = self.planner.analyze(question)
            
            # Stage 2: Evidence Builder
            evidence_output = self.evidence_builder.build(question, planner_output)
            
            # Stage 3: Judge
            judge_output = self.judge.judge(question, planner_output, evidence_output)
            
            return TeacherPipelineOutput(
                question_id=question_id,
                question=question,
                ground_truth_answer=ground_truth_answer,
                planner_output=planner_output.to_dict(),
                evidence_output=evidence_output.to_dict(),
                judge_output=judge_output.to_dict(),
                final_context=judge_output.final_context.to_dict(),
                success=True
            )
        
        except Exception as e:
            logger.error(f"Pipeline failed for {question_id}: {e}")
            return TeacherPipelineOutput(
                question_id=question_id,
                question=question,
                ground_truth_answer=ground_truth_answer,
                planner_output={},
                evidence_output={},
                judge_output={},
                final_context={},
                success=False,
                error_message=str(e)
            )
    
    def process_batch(
        self,
        input_path: Path,
        output_path: Path,
        resume: bool = True
    ) -> dict:
        """Process multiple questions from a JSONL file"""
        ensure_dir(output_path.parent)
        
        # Track processed IDs for resume
        processed_ids = set()
        if resume and output_path.exists():
            for record in iter_jsonl(output_path):
                processed_ids.add(record.get("question_id"))
            logger.info(f"Resuming: {len(processed_ids)} already processed")
        
        # Process questions
        stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
        
        for record in tqdm(iter_jsonl(input_path), desc="Processing questions"):
            question_id = record.get("id", record.get("question_id", str(stats["total"])))
            
            if question_id in processed_ids:
                stats["skipped"] += 1
                continue
            
            question = record.get("question", record.get("text", ""))
            answer = record.get("answer", record.get("ground_truth", None))
            
            result = self.process(question_id, question, answer)
            append_jsonl(result.to_dict(), output_path)
            
            stats["total"] += 1
            if result.success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
        
        logger.info(f"Completed: {stats}")
        return stats


def main():
    parser = argparse.ArgumentParser(description="Run Teacher Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous run")
    args = parser.parse_args()
    
    # Load config
    config = load_yaml(args.config)
    
    # Create LLM client
    llm_client = create_llm_client(config.get("teacher", {}))
    
    # Create and run pipeline
    pipeline = TeacherPipeline(llm_client, config)
    
    stats = pipeline.process_batch(
        input_path=Path(args.input),
        output_path=Path(args.output),
        resume=not args.no_resume
    )
    
    print(f"\nPipeline completed: {stats}")


if __name__ == "__main__":
    main()

"""
Training Data Preparation
4개 하위 태스크로 데이터 분할:
1. Context Necessity Classification
2. Constraint/Entity Extraction  
3. Evidence Compression
4. Full Context Generation
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
import random

from tqdm import tqdm

from models.schemas import StructuredContext, QuestionType
from utils.file_utils import load_yaml, iter_jsonl, save_jsonl, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataPreparer:
    """Prepare training data for 4 subtasks"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        random.seed(self.config.get("seed", 42))
    
    def prepare_all(
        self,
        teacher_output_path: Path,
        labeled_path: Path,
        output_dir: Path,
        helpful_contexts_path: Optional[Path] = None
    ) -> dict:
        """Prepare all training datasets"""
        ensure_dir(output_dir)
        
        # Load data
        teacher_outputs = list(iter_jsonl(teacher_output_path))
        logger.info(f"Loaded {len(teacher_outputs)} teacher outputs")

        helpful_ids: Optional[set[str]] = None
        if helpful_contexts_path and helpful_contexts_path.exists():
            helpful_ids = set()
            for row in iter_jsonl(helpful_contexts_path):
                qid = row.get("question_id")
                if qid:
                    helpful_ids.add(qid)
            logger.info(f"Loaded {len(helpful_ids)} SLM-helpful question ids")
        
        # Prepare each task
        stats = {}
        
        # Task A: Context Necessity Classification
        necessity_data = self._prepare_necessity_task(teacher_outputs)
        save_jsonl(necessity_data, output_dir / "task_a_necessity.jsonl")
        stats["necessity"] = len(necessity_data)
        
        # Task B: Constraint/Entity Extraction
        extraction_data = self._prepare_extraction_task(teacher_outputs)
        save_jsonl(extraction_data, output_dir / "task_b_extraction.jsonl")
        stats["extraction"] = len(extraction_data)
        
        # Task C: Evidence Compression
        compression_data = self._prepare_compression_task(teacher_outputs, helpful_ids)
        save_jsonl(compression_data, output_dir / "task_c_compression.jsonl")
        stats["compression"] = len(compression_data)
        
        # Task D: Full Context Generation
        full_gen_data = self._prepare_full_generation_task(teacher_outputs, helpful_ids)
        save_jsonl(full_gen_data, output_dir / "task_d_full_generation.jsonl")
        stats["full_generation"] = len(full_gen_data)
        
        # Combined SFT data
        combined = necessity_data + extraction_data + compression_data + full_gen_data
        random.shuffle(combined)
        save_jsonl(combined, output_dir / "sft_combined.jsonl")
        stats["combined"] = len(combined)
        stats["helpful_filter_enabled"] = helpful_ids is not None
        
        # Prepare DPO data if preferences available
        if labeled_path.exists():
            dpo_data = self._prepare_dpo_data(labeled_path)
            save_jsonl(dpo_data, output_dir / "dpo_preferences.jsonl")
            stats["dpo_pairs"] = len(dpo_data)
        
        return stats
    
    def _prepare_necessity_task(self, teacher_outputs: list[dict]) -> list[dict]:
        """Task A: Context Necessity Classification"""
        data = []
        
        for record in teacher_outputs:
            if not record.get("success"):
                continue
            
            question = record["question"]
            final_context = record.get("final_context", {})
            need_context = final_context.get("need_context", True)
            question_type = final_context.get("question_type", "factoid")
            
            # Format as instruction
            instruction = "Determine if this question requires external context to answer properly."
            
            input_text = f"Question: {question}"
            
            output_text = json.dumps({
                "need_context": need_context,
                "question_type": question_type,
                "reasoning": self._generate_necessity_reasoning(need_context, question_type)
            }, ensure_ascii=False)
            
            data.append({
                "task": "necessity_classification",
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
        
        return data
    
    def _generate_necessity_reasoning(self, need_context: bool, question_type: str) -> str:
        """Generate reasoning for necessity classification"""
        if not need_context:
            return "This question can be answered with common knowledge or basic reasoning."
        
        type_reasons = {
            "factoid": "Specific factual information may require verification.",
            "comparison": "Comparing entities requires concrete data for each.",
            "multi-hop": "Multiple reasoning steps require supporting facts.",
            "procedural": "Step-by-step procedures need accurate details.",
            "ambiguous": "Clarification or additional context may help.",
        }
        
        return type_reasons.get(question_type, "External context would improve answer quality.")
    
    def _prepare_extraction_task(self, teacher_outputs: list[dict]) -> list[dict]:
        """Task B: Constraint/Entity Extraction"""
        data = []
        
        for record in teacher_outputs:
            if not record.get("success"):
                continue
            
            question = record["question"]
            planner = record.get("planner_output", {})
            final_context = record.get("final_context", {})
            
            entities = planner.get("entities", []) or final_context.get("entities", [])
            constraints = planner.get("constraints", []) or final_context.get("constraints", [])
            
            instruction = "Extract key entities and constraints from this question."
            input_text = f"Question: {question}"
            
            output_text = json.dumps({
                "entities": entities,
                "constraints": constraints
            }, ensure_ascii=False)
            
            data.append({
                "task": "extraction",
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
        
        return data
    
    def _prepare_compression_task(self, teacher_outputs: list[dict], helpful_ids: Optional[set[str]] = None) -> list[dict]:
        """Task C: Evidence Compression"""
        data = []
        
        for record in teacher_outputs:
            if not record.get("success"):
                continue
            if helpful_ids is not None and record.get("question_id") not in helpful_ids:
                continue
            
            question = record["question"]
            evidence_output = record.get("evidence_output", {})
            judge_output = record.get("judge_output", {})
            
            # Get raw evidence
            combined_evidence = evidence_output.get("combined_evidence", [])
            pseudo_doc = evidence_output.get("pseudo_document", "")
            
            # Get compressed output
            selected_facts = judge_output.get("selected_facts", [])
            
            if not combined_evidence and not pseudo_doc:
                continue
            
            # Build input from raw evidence
            evidence_texts = [e.get("content", "") for e in combined_evidence if e.get("content")]
            if pseudo_doc:
                evidence_texts.append(pseudo_doc)
            
            raw_evidence = "\n".join(f"- {t}" for t in evidence_texts[:10])
            
            instruction = "Compress the given evidence into minimal sufficient facts for answering the question."
            input_text = f"Question: {question}\n\nEvidence:\n{raw_evidence}"
            
            output_text = json.dumps({
                "useful_facts": selected_facts
            }, ensure_ascii=False)
            
            data.append({
                "task": "compression",
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
        
        return data
    
    def _prepare_full_generation_task(self, teacher_outputs: list[dict], helpful_ids: Optional[set[str]] = None) -> list[dict]:
        """Task D: Full Context Generation"""
        data = []
        
        for record in teacher_outputs:
            if not record.get("success"):
                continue
            if helpful_ids is not None and record.get("question_id") not in helpful_ids:
                continue
            
            question = record["question"]
            final_context = record.get("final_context", {})
            
            instruction = """Generate minimal sufficient context for a small language model to answer this question.
Output JSON with: need_context, question_type, entities, constraints, useful_facts, missing_info, answer_hint.
Do NOT include the answer directly."""
            
            input_text = f"Question: {question}"
            output_text = json.dumps(final_context, ensure_ascii=False)
            
            data.append({
                "task": "full_generation",
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
        
        return data
    
    def _prepare_dpo_data(self, labeled_path: Path) -> list[dict]:
        """Prepare DPO preference pairs"""
        dpo_data = []
        
        for record in iter_jsonl(labeled_path):
            # Expect preference label format
            if "chosen_context" not in record or "rejected_context" not in record:
                continue
            
            question = record["question"]
            
            instruction = "Generate minimal sufficient context for answering this question."
            input_text = f"Question: {question}"
            
            dpo_data.append({
                "instruction": instruction,
                "input": input_text,
                "chosen": json.dumps(record["chosen_context"], ensure_ascii=False),
                "rejected": json.dumps(record["rejected_context"], ensure_ascii=False),
                "chosen_type": record.get("chosen_type", ""),
                "rejected_type": record.get("rejected_type", ""),
                "performance_delta": record.get("performance_delta", 0.0)
            })
        
        return dpo_data


def main():
    parser = argparse.ArgumentParser(description="Prepare Training Data")
    parser.add_argument("--teacher-output", type=str, required=True)
    parser.add_argument("--labeled", type=str, default="")
    parser.add_argument("--helpful-contexts", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    args = parser.parse_args()
    
    config = load_yaml(args.config) if Path(args.config).exists() else {}
    
    preparer = TrainingDataPreparer(config.get("training", {}))
    
    labeled_path = Path(args.labeled) if args.labeled else Path("")
    helpful_contexts_path = Path(args.helpful_contexts) if args.helpful_contexts else None
    
    stats = preparer.prepare_all(
        teacher_output_path=Path(args.teacher_output),
        labeled_path=labeled_path,
        output_dir=Path(args.output_dir),
        helpful_contexts_path=helpful_contexts_path
    )
    
    print("\nTraining data prepared:")
    for task, count in stats.items():
        print(f"  {task}: {count} examples")


if __name__ == "__main__":
    main()

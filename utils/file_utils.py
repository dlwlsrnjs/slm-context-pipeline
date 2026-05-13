"""
File utilities for the pipeline
"""
import json
import yaml
from pathlib import Path
from typing import Iterator, Any
import logging

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    """Load YAML configuration file"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    """Save data to YAML file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate over records in a JSONL file (memory efficient)"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(records: list[dict], path: str | Path, append: bool = False) -> None:
    """Save records to a JSONL file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def append_jsonl(record: dict, path: str | Path) -> None:
    """Append a single record to a JSONL file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_json(path: str | Path) -> dict | list:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict | list, path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_lines(path: str | Path) -> int:
    """Count number of lines in a file"""
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def split_file(
    input_path: str | Path,
    output_dir: str | Path,
    num_splits: int = 10,
    prefix: str = "split"
) -> list[Path]:
    """Split a JSONL file into multiple parts"""
    records = load_jsonl(input_path)
    output_dir = ensure_dir(output_dir)
    
    chunk_size = len(records) // num_splits + 1
    output_paths = []
    
    for i in range(num_splits):
        start = i * chunk_size
        end = min(start + chunk_size, len(records))
        if start >= len(records):
            break
        
        chunk = records[start:end]
        output_path = output_dir / f"{prefix}_{i:03d}.jsonl"
        save_jsonl(chunk, output_path)
        output_paths.append(output_path)
        logger.info(f"Saved {len(chunk)} records to {output_path}")
    
    return output_paths


def merge_files(input_paths: list[str | Path], output_path: str | Path) -> None:
    """Merge multiple JSONL files into one"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for input_path in input_paths:
            for record in iter_jsonl(input_path):
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total += 1
    
    logger.info(f"Merged {total} records into {output_path}")

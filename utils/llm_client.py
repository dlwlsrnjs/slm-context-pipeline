"""
LLM Client for Teacher Model and Answer Model
"""
import os
import json
import time
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API Client"""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=api_key, base_url=api_base)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        response = self.generate(prompt, system_prompt)
        
        # JSON 파싱 시도
        try:
            # ```json ... ``` 블록 처리
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nResponse: {response}")
            return {}


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API Client"""
    
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                }
                if system_prompt:
                    kwargs["system"] = system_prompt
                
                response = self.client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        response = self.generate(prompt, system_prompt)
        
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nResponse: {response}")
            return {}


class MockLLMClient(BaseLLMClient):
    """Mock client for testing"""
    
    def __init__(self, default_response: str = "Mock response"):
        self.default_response = default_response
        self.call_count = 0
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.call_count += 1
        return self.default_response
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        self.call_count += 1
        system_lower = (system_prompt or "").lower()

        if "question analysis expert" in system_lower:
            return {
                "question_type": "factoid",
                "need_external_context": True,
                "entities": ["entity1", "entity2"],
                "constraints": ["time=2024"],
                "subquestions": ["핵심 사실은 무엇인가?"],
                "retrieval_queries": ["entity1 2024 fact"]
            }

        if "evidence generation expert" in system_lower:
            return {
                "generated_evidence": [
                    {"content": "entity1 관련 핵심 사실 A", "relevance_score": 0.9},
                    {"content": "entity2 관련 핵심 사실 B", "relevance_score": 0.8},
                    {"content": "비교를 위해 동일 시점 데이터가 필요", "relevance_score": 0.7}
                ],
                "pseudo_document": "entity1과 entity2를 비교하려면 동일 시점의 핵심 지표를 확인해야 한다."
            }

        if "context quality judge" in system_lower:
            return {
                "selected_facts": [
                    "entity1 관련 핵심 사실 A",
                    "entity2 관련 핵심 사실 B"
                ],
                "rejected_facts": ["비교를 위해 동일 시점 데이터가 필요"],
                "rejection_reasons": {
                    "비교를 위해 동일 시점 데이터가 필요": "Generic procedural note"
                },
                "information_density_score": 0.82,
                "has_answer_leakage": False,
                "distractor_ratio": 0.15,
                "final_context": {
                    "need_context": True,
                    "question_type": "factoid",
                    "entities": ["entity1", "entity2"],
                    "constraints": ["time=2024"],
                    "subquestions": [],
                    "useful_facts": [
                        "entity1 관련 핵심 사실 A",
                        "entity2 관련 핵심 사실 B"
                    ],
                    "missing_info": [],
                    "answer_hint": "두 사실을 시간 조건에 맞춰 비교/검증하라"
                }
            }

        return {
            "need_context": True,
            "question_type": "factoid",
            "entities": ["entity1"],
            "constraints": [],
            "subquestions": [],
            "useful_facts": ["Fact 1"],
            "missing_info": [],
            "answer_hint": "This is a hint"
        }


def create_llm_client(config: dict) -> BaseLLMClient:
    """Factory function to create LLM client from config"""
    model = config.get("model", "")
    model_lower = model.lower()
    api_key_env = config.get("api_key_env", "")
    api_key = os.getenv(api_key_env) if api_key_env else None
    api_base = config.get("api_base", "")
    api_base_lower = api_base.lower()

    if model_lower in {"mock", "dummy", "test"}:
        return MockLLMClient(default_response=config.get("mock_response", "Mock response"))
    
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower or "o4" in model_lower or "openai" in api_base_lower:
        return OpenAIClient(
            model=model,
            api_key=api_key,
            api_base=config.get("api_base", "https://api.openai.com/v1"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048)
        )
    elif "claude" in model_lower:
        return AnthropicClient(
            model=model,
            api_key=api_key,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048)
        )
    else:
        logger.warning(f"Unknown model {model}, using OpenAI client as default")
        return OpenAIClient(
            model=model,
            api_key=api_key,
            api_base=config.get("api_base", "https://api.openai.com/v1"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048)
        )

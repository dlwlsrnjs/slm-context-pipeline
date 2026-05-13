from setuptools import setup, find_packages

setup(
    name="slm-context-pipeline",
    version="0.1.0",
    description="SLM Minimal Sufficient Context Generation Pipeline",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "openai>=1.0.0",
    ],
    extras_require={
        "training": [
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "datasets>=2.16.0",
            "peft>=0.7.0",
            "trl>=0.7.0",
            "accelerate>=0.25.0",
        ],
        "anthropic": [
            "anthropic>=0.18.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slm-teacher=pipeline.run_teacher:main",
            "slm-candidates=data_processing.generate_candidates:main",
            "slm-evaluate=evaluation.evaluate_downstream:main",
            "slm-prepare=training.prepare_training_data:main",
            "slm-sft=training.train_sft:main",
            "slm-dpo=training.train_dpo:main",
        ],
    },
)

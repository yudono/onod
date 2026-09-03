"""
Training package — PRD §14-17 pipeline stub.
Re-exports from fast_rag.benchmarks.datasets and fast_rag.benchmarks.benchmark.
"""

from fast_rag.benchmarks.datasets import TRAINING_DATASETS, TEACHER_STUDENT_CONFIG, DATASET_COMBINATION_STRATEGY
from fast_rag.benchmarks.benchmark import TrainingPipelineStub

__all__ = ["TRAINING_DATASETS", "TEACHER_STUDENT_CONFIG", "DATASET_COMBINATION_STRATEGY", "TrainingPipelineStub"]

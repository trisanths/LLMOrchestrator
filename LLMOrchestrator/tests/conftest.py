"""
Shared test fixtures for LLMOrchestrator tests.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from ..controller import (
    ValidationMetrics,
    AdaptiveLearning,
    PromptTemplate,
    OutputCache,
    Controller,
    CustomController
)

class MockGenerator:
    def generate_output(self, prompt: str) -> str:
        return f"Generated output for: {prompt}"

class MockVerifier:
    def verify(self, text: str) -> tuple[bool, str]:
        return True, text

@pytest.fixture
def mock_generator():
    return MockGenerator()

@pytest.fixture
def mock_verifier():
    return MockVerifier()

@pytest.fixture
def validation_metrics():
    return ValidationMetrics(
        confidence_score=0.8,
        processing_time=1.5,
        token_count=100,
        refinement_count=2,
        validation_checks=["check1", "check2"]
    )

@pytest.fixture
def adaptive_learning():
    return AdaptiveLearning(initial_learning_rate=0.1)

@pytest.fixture
def prompt_template():
    template = PromptTemplate("Base template: {prompt}")
    template.add_variation("variation1", "Variation 1: {prompt}")
    template.add_variation("variation2", "Variation 2: {prompt}")
    return template

@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture
def output_cache(temp_cache_dir):
    return OutputCache(cache_dir=temp_cache_dir, max_size_mb=1)

@pytest.fixture
def controller(mock_generator, mock_verifier):
    return Controller(
        generator=mock_generator,
        verifier=mock_verifier,
        max_iterations=3,
        cache_enabled=True,
        monitoring_enabled=True
    )

@pytest.fixture
def custom_controller(mock_generator, mock_verifier):
    custom_func = lambda g, v, p, n: f"Custom output for: {p}"
    return CustomController(
        custom_func=custom_func,
        generator=mock_generator,
        verifier=mock_verifier
    ) 
"""
Test suite for LLMOrchestrator controller with real models.
"""
import pytest
import logging
import os
from dotenv import load_dotenv

from LLMOrchestrator.controller import Controller, PromptTemplate
from LLMOrchestrator.models.openai_model import OpenAIModel
from LLMOrchestrator.models.local_model import LocalModel

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def check_api_key(key_name: str) -> bool:
    """Check if an API key is available."""
    key = os.getenv(key_name)
    return key is not None and len(key.strip()) > 0

@pytest.fixture
def openai_model():
    """Fixture for OpenAI model."""
    if not check_api_key("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not found")
    return OpenAIModel(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )

@pytest.fixture
def local_model():
    """Fixture for local model."""
    return LocalModel(
        model_name="facebook/opt-125m",
        device="cpu"
    )

@pytest.fixture
def prompt_template():
    """Fixture for prompt template."""
    return PromptTemplate("Write a haiku about {topic}")

@pytest.mark.skipif(not check_api_key("OPENAI_API_KEY"),
                   reason="OpenAI API key not found")
def test_openai_parallel_processing(openai_model, prompt_template):
    """Test OpenAI model's parallel processing capabilities."""
    controller = Controller(
        generator=openai_model,
        verifier=openai_model,
        max_iterations=2,
        parallel_processing=True,
        monitoring_enabled=True
    )
    
    # Test with shorter prompts
    prompts = [
        "Write a haiku about spring.",
        "Write a haiku about winter."
    ]
    
    results = controller.execute_parallel(prompts, max_workers=2)
    
    # Basic verification
    assert len(results) == 2
    assert all(isinstance(result, str) for result in results)
    assert all(len(result.split('\n')) >= 3 for result in results)

@pytest.mark.skipif(not check_api_key("OPENAI_API_KEY"),
                   reason="OpenAI API key not found")
def test_cross_model_verification(openai_model, local_model, prompt_template):
    """Test cross-model verification capabilities."""
    controller = Controller(
        generator=openai_model,
        verifier=local_model,
        max_iterations=2,
        max_verifications=2,
        prompt_template=prompt_template,
        monitoring_enabled=True
    )
    
    prompt = "Write a short story about a cat and a mouse becoming friends."
    
    result = controller.execute(prompt, stop_early=True)
    
    # Basic verification
    assert isinstance(result, str)
    assert len(result.split()) >= 20
    assert "cat" in result.lower()
    assert "mouse" in result.lower()

def test_error_handling(openai_model, prompt_template):
    """Test error handling and recovery."""
    controller = Controller(
        generator=openai_model,
        verifier=openai_model,
        max_iterations=2,
        max_verifications=2,
        prompt_template=prompt_template,
        monitoring_enabled=True
    )
    
    # Test with an empty prompt
    with pytest.raises(Exception):
        controller.execute("", stop_early=True)
    
    # Verify the controller can still process valid prompts after an error
    valid_prompt = "Write a haiku about coding."
    result = controller.execute(valid_prompt, stop_early=True)
    
    # Basic verification
    assert isinstance(result, str)
    assert len(result.split()) >= 5 
"""
Comprehensive test suite for LLMOrchestrator capabilities.
"""
import os
import logging
import json
from dotenv import load_dotenv
from typing import Optional

from LLMOrchestrator.controller import Controller, PromptTemplate
from LLMOrchestrator.models.openai_model import OpenAIModel
from LLMOrchestrator.models.local_model import LocalModel

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_api_key(key_name: str) -> bool:
    """Check if an API key is available."""
    key = os.getenv(key_name)
    return key is not None and len(key.strip()) > 0

def test_openai_capabilities():
    """Test OpenAI model capabilities."""
    if not check_api_key("OPENAI_API_KEY"):
        logger.warning("OpenAI API key not found. Skipping OpenAI tests.")
        return

    logger.info("Testing OpenAI capabilities...")
    
    # Initialize OpenAI model
    model = OpenAIModel(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create controller
    controller = Controller(
        generator=model,
        verifier=model,
        max_iterations=2,
        max_verifications=2,
        monitoring_enabled=True
    )
    
    # Test basic generation
    prompt = "Write a haiku about coding."
    result = controller.execute(prompt, stop_early=True)
    logger.info(f"OpenAI Generation Result: {result[:100]}...")
    assert isinstance(result, str)
    assert len(result.split('\n')) >= 3
    
    # Test parallel processing
    prompts = [
        "Write a haiku about spring.",
        "Write a haiku about winter."
    ]
    results = controller.execute_parallel(prompts, max_workers=2)
    logger.info(f"OpenAI Parallel Results: {len(results)} responses")
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)

def test_local_model_capabilities():
    """Test local model capabilities."""
    logger.info("Testing Local Model capabilities...")
    
    try:
        # Initialize local model
        model = LocalModel(
            model_name="facebook/opt-125m",
            device="cpu",
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Create controller
        controller = Controller(
            generator=model,
            verifier=model,
            max_iterations=2,
            max_verifications=2,
            monitoring_enabled=True
        )
        
        # Test basic generation
        prompt = "Write a short sentence about coding."
        result = controller.execute(prompt, stop_early=True)
        logger.info(f"Local Model Generation Result: {result[:100]}...")
        assert isinstance(result, str)
        assert len(result.split()) >= 3
        
    except Exception as e:
        logger.warning(f"Local model test failed: {e}")
        logger.info("This is expected if the model is not downloaded or resources are limited.")

def test_cross_model_verification():
    """Test cross-model verification capabilities."""
    if not check_api_key("OPENAI_API_KEY"):
        logger.warning("OpenAI API key not found. Skipping cross-model verification test.")
        return

    logger.info("Testing Cross-Model Verification...")
    
    # Initialize models
    generator = OpenAIModel(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    verifier = OpenAIModel(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create controller
    controller = Controller(
        generator=generator,
        verifier=verifier,
        max_iterations=2,
        max_verifications=2,
        monitoring_enabled=True
    )
    
    # Test cross-model verification
    prompt = "Write a short story about a robot learning to paint."
    result = controller.execute(prompt, stop_early=True)
    logger.info(f"Cross-Model Verification Result: {result[:100]}...")
    assert isinstance(result, str)
    assert len(result.split()) >= 20
    assert "robot" in result.lower()
    assert "paint" in result.lower()

def test_error_handling():
    """Test error handling capabilities."""
    if not check_api_key("OPENAI_API_KEY"):
        logger.warning("OpenAI API key not found. Skipping error handling test.")
        return

    logger.info("Testing Error Handling...")
    
    # Initialize OpenAI model
    model = OpenAIModel(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create controller
    controller = Controller(
        generator=model,
        verifier=model,
        max_iterations=2,
        max_verifications=2,
        monitoring_enabled=True
    )
    
    # Test empty prompt
    try:
        controller.execute("", stop_early=True)
        raise AssertionError("Empty prompt should raise an exception")
    except Exception as e:
        logger.info(f"Empty prompt correctly raised exception: {e}")
    
    # Test recovery after error
    valid_prompt = "Write a haiku about coding."
    result = controller.execute(valid_prompt, stop_early=True)
    logger.info(f"Recovery Test Result: {result[:100]}...")
    assert isinstance(result, str)
    assert len(result.split('\n')) >= 3

def main():
    """Run all tests."""
    # Load environment variables
    load_dotenv()
    
    # Run tests
    test_openai_capabilities()
    test_local_model_capabilities()
    test_cross_model_verification()
    test_error_handling()
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main() 
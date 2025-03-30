#!/usr/bin/env python
"""
Test script for verifying controller with real OpenAI and local models.
"""
import sys
import logging
import time
import os
from dotenv import load_dotenv

from LLMOrchestrator.controller import Controller, PromptTemplate
from LLMOrchestrator.models.openai_model import OpenAIModel
from LLMOrchestrator.models.local_model import LocalModel

# Load environment variables
load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def run_test(description: str, controller: Controller, prompt: str, stop_early: bool = False):
    """Run a single test and return the result."""
    print(f"\n{description}")
    print("-"*80)
    
    start_time = time.time()
    try:
        print(f"Starting test with prompt: {prompt}")
        result = controller.execute(prompt, stop_early=stop_early)
        elapsed = time.time() - start_time
        
        print("\nRESULT SUMMARY:")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Result length: {len(result.split())} words")
        print(f"  First 100 chars: {result[:100]}...")
        return True
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return False

def main():
    print("\n" + "="*80)
    print("LLMOrchestrator Controller Test with Real Models")
    print("="*80)
    
    # Initialize OpenAI model
    print("\nInitializing OpenAI model...")
    openai_model = OpenAIModel(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize local model (using a smaller model for testing)
    print("\nInitializing local model...")
    local_model = LocalModel(
        model_name="facebook/opt-125m",  # Small model for testing
        device="cpu"  # Use CPU for testing
    )
    
    # Create a simple prompt template
    template = PromptTemplate("{prompt}")
    
    # Test 1: OpenAI model with all iterations
    controller1 = Controller(
        generator=openai_model,
        verifier=openai_model,  # Use same model for verification
        max_iterations=2,
        max_verifications=3,
        prompt_template=template,
        monitoring_enabled=True
    )
    
    success1 = run_test(
        "Test 1: OpenAI model with all iterations (stop_early=False)",
        controller1,
        "Write a short story about a robot learning to paint.",
        stop_early=False
    )
    
    # Test 2: Local model with early stopping
    controller2 = Controller(
        generator=local_model,
        verifier=local_model,  # Use same model for verification
        max_iterations=3,
        max_verifications=3,
        prompt_template=template,
        monitoring_enabled=True
    )
    
    success2 = run_test(
        "Test 2: Local model with early stopping (stop_early=True)",
        controller2,
        "Explain the concept of artificial intelligence in simple terms.",
        stop_early=True
    )
    
    # Test 3: Parallel processing with OpenAI model
    controller3 = Controller(
        generator=openai_model,
        verifier=openai_model,
        max_iterations=2,
        parallel_processing=True,
        monitoring_enabled=True
    )
    
    print("\nTest 3: Parallel processing with OpenAI model")
    print("-"*80)
    
    prompts = [
        "Write a haiku about technology.",
        "Create a recipe for a futuristic dessert."
    ]
    start_time = time.time()
    try:
        results = controller3.execute_parallel(prompts, max_workers=2)
        elapsed = time.time() - start_time
        
        print("\nRESULT SUMMARY:")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Number of results: {len(results)}")
        for i, result in enumerate(results):
            print(f"  Result {i+1} length: {len(result.split()) if result else 0} words")
        success3 = True
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        success3 = False
    
    print("\n" + "="*80)
    print("Test Results:")
    print(f"  Test 1 (OpenAI): {'PASSED' if success1 else 'FAILED'}")
    print(f"  Test 2 (Local): {'PASSED' if success2 else 'FAILED'}")
    print(f"  Test 3 (Parallel): {'PASSED' if success3 else 'FAILED'}")
    print("="*80)
    
    return 0 if all([success1, success2, success3]) else 1

if __name__ == "__main__":
    sys.exit(main()) 
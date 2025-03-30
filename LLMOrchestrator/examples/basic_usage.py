"""
Basic usage example of LLMOrchestrator.
"""
import time
from typing import Optional
from LLMOrchestrator.controller import Controller, PromptTemplate
from LLMOrchestrator.examples.mock_llm import MockLLMGenerator, MockLLMVerifier

def main():
    try:
        # Initialize components with shorter delays
        generator = MockLLMGenerator(delay_range=(0.01, 0.1))
        verifier = MockLLMVerifier(delay_range=(0.01, 0.05))
        
        # Create a prompt template
        template = PromptTemplate(
            "Please analyze the following text and provide insights: {prompt}"
        )
        
        # Initialize the controller with shorter max iterations
        controller = Controller(
            generator=generator,
            verifier=verifier,
            max_iterations=2,  # Reduced from 3 to 2
            prompt_template=template,
            cache_enabled=True,
            monitoring_enabled=True
        )
        
        # Execute a prompt with timeout
        prompt = "The impact of artificial intelligence on healthcare"
        print(f"Processing prompt: {prompt}")
        
        start_time = time.time()
        result = controller.execute(prompt)
        processing_time = time.time() - start_time
        
        # Get performance metrics
        metrics = controller.get_validation_metrics()
        print(f"\nGenerated output: {result}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Quality score: {metrics.quality_score:.2f}")
        
        # Get performance report
        report = controller.get_performance_report()
        print("\nPerformance Report:")
        print(f"Cache size: {report['cache_stats']['size']} bytes")
        print(f"Cache entries: {report['cache_stats']['entries']}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
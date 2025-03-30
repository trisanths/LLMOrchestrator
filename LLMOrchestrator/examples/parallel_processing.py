"""
Example demonstrating parallel processing and caching capabilities.
"""
import time
from LLMOrchestrator.controller import Controller, PromptTemplate
from LLMOrchestrator.examples.mock_llm import MockLLMGenerator, MockLLMVerifier

def main():
    # Initialize components
    generator = MockLLMGenerator()
    verifier = MockLLMVerifier()
    
    # Create a prompt template
    template = PromptTemplate(
        "Analyze the following topic and provide detailed insights: {prompt}"
    )
    
    # Initialize the controller with parallel processing enabled
    controller = Controller(
        generator=generator,
        verifier=verifier,
        max_iterations=3,
        prompt_template=template,
        parallel_processing=True,
        cache_enabled=True,
        monitoring_enabled=True
    )
    
    # Define multiple prompts to process in parallel
    prompts = [
        "The future of renewable energy",
        "Impact of social media on society",
        "Advances in quantum computing",
        "The role of blockchain in finance",
        "Climate change and its effects"
    ]
    
    # Process prompts sequentially (for comparison)
    print("Processing prompts sequentially...")
    start_time = time.time()
    sequential_results = [controller.execute(prompt) for prompt in prompts]
    sequential_time = time.time() - start_time
    
    # Process prompts in parallel
    print("\nProcessing prompts in parallel...")
    start_time = time.time()
    parallel_results = controller.execute_parallel(prompts, max_workers=3)
    parallel_time = time.time() - start_time
    
    # Print results and timing comparison
    print("\nResults:")
    print(f"Sequential processing time: {sequential_time:.2f} seconds")
    print(f"Parallel processing time: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Demonstrate caching
    print("\nDemonstrating caching...")
    start_time = time.time()
    cached_results = [controller.execute(prompt) for prompt in prompts]
    cached_time = time.time() - start_time
    
    print(f"Cached processing time: {cached_time:.2f} seconds")
    print(f"Cache speedup: {sequential_time/cached_time:.2f}x")
    
    # Get cache statistics
    report = controller.get_performance_report()
    print("\nCache Statistics:")
    print(f"Total cache entries: {report['cache_stats']['entries']}")
    print(f"Cache size: {report['cache_stats']['size']} bytes")

if __name__ == "__main__":
    main() 
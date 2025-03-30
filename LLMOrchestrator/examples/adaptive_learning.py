"""
Example demonstrating adaptive learning and prompt template optimization.
"""
import time
from LLMOrchestrator.controller import Controller, PromptTemplate
from LLMOrchestrator.examples.mock_llm import MockLLMGenerator, MockLLMVerifier

def main():
    # Initialize components
    generator = MockLLMGenerator()
    verifier = MockLLMVerifier()
    
    # Create a prompt template with multiple variations
    template = PromptTemplate(
        "Analyze the following topic: {prompt}"
    )
    
    # Add different template variations
    template.add_variation(
        "detailed",
        "Please provide a detailed analysis of the following topic: {prompt}"
    )
    template.add_variation(
        "concise",
        "Give a concise summary of: {prompt}"
    )
    template.add_variation(
        "technical",
        "Provide a technical analysis of: {prompt}"
    )
    
    # Initialize the controller with adaptive learning enabled
    controller = Controller(
        generator=generator,
        verifier=verifier,
        max_iterations=3,
        prompt_template=template,
        adaptive_learning=True,
        monitoring_enabled=True
    )
    
    # Define test prompts
    prompts = [
        "The impact of AI on healthcare",
        "Quantum computing applications",
        "Blockchain technology",
        "Climate change solutions",
        "Space exploration"
    ]
    
    # Initial learning phase
    print("Initial learning phase...")
    for prompt in prompts:
        result = controller.execute(prompt)
        metrics = controller.get_validation_metrics()
        print(f"\nPrompt: {prompt}")
        print(f"Quality score: {metrics.quality_score:.2f}")
        print(f"Best template variation: {template.get_best_variation()}")
    
    # Get learning parameters
    learning_params = controller.adaptive_learning.get_optimal_parameters()
    print("\nLearning Parameters:")
    print(f"Learning rate: {learning_params['learning_rate']:.3f}")
    print(f"Performance trend: {learning_params['performance_trend']:.3f}")
    
    # Test with new prompts after learning
    print("\nTesting with new prompts after learning...")
    new_prompts = [
        "Machine learning in finance",
        "Renewable energy technology",
        "Digital privacy concerns"
    ]
    
    for prompt in new_prompts:
        result = controller.execute(prompt)
        metrics = controller.get_validation_metrics()
        print(f"\nPrompt: {prompt}")
        print(f"Quality score: {metrics.quality_score:.2f}")
        print(f"Best template variation: {template.get_best_variation()}")
    
    # Get final performance report
    report = controller.get_performance_report()
    print("\nFinal Performance Report:")
    print(f"Average quality score: {metrics.quality_score:.2f}")
    print(f"Total prompts processed: {len(prompts) + len(new_prompts)}")
    print(f"Best performing template: {template.get_best_variation()}")
    
    # Show template performance metrics
    print("\nTemplate Performance Metrics:")
    for variation in template.variations:
        scores = template.performance_metrics.get(variation, [])
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"{variation}: {avg_score:.2f}")

if __name__ == "__main__":
    main() 
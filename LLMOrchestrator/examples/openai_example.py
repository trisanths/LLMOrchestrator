"""
Example using OpenAI models with LLMOrchestrator.
"""
import os
import time
import sys
from typing import Optional
from dotenv import load_dotenv
import logging

from LLMOrchestrator.controller import Controller, PromptTemplate
from LLMOrchestrator.models.openai_model import OpenAIModel
from LLMOrchestrator.verifier import Verifier

def create_relevance_verifier():
    """Create a verifier that checks if output is relevant to the input prompt."""
    def verify_quality(text: str, prompt: str = None) -> tuple[bool, str]:
        if not text or not text.strip():
            return False, "Output is empty"
        
        # Add a very basic length check just for demonstration
        if len(text.strip()) < 10:
            return False, f"Output too short ({len(text.strip())} chars)"
        
        # Check if the response contains keywords from the prompt
        # For simple prompts like "What is water?", the response should 
        # contain the word "water"
        if prompt:
            keywords = [word.lower() for word in prompt.split() if len(word) > 3]
            found_keywords = [keyword for keyword in keywords if keyword in text.lower()]
            
            if not found_keywords and len(keywords) > 0:
                return False, f"Response doesn't seem related to the prompt about {', '.join(keywords)}"
            
        return True, text
    
    return Verifier(custom_verifier=verify_quality)

def main():
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set it in your .env file or environment variables")
            sys.exit(1)

        print("\n=== LLMOrchestrator Demo ===")
        print("Initializing OpenAI model...")
        
        try:
            # Initialize OpenAI model with better parameters
            model = OpenAIModel(
                api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.2,  # Lower temperature for more predictable responses
                max_tokens=100  # Enough tokens for a brief answer
            )
            
            # Create a prompt template
            template = PromptTemplate(
                "Answer this question directly: {prompt}"
            )
            
            # Create a relevance verifier
            verifier = create_relevance_verifier()
            
            # Initialize the controller with verification limit
            controller = Controller(
                generator=model,
                verifier=verifier,
                max_iterations=2,  # Allow for retries
                max_verifications=3,  # Allow multiple verification attempts
                prompt_template=template,
                cache_enabled=True,
                monitoring_enabled=True
            )
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
        
        # Example prompts - extremely simple
        prompts = [
            "What is water?",
            "Define AI",
            "Explain colors"
        ]
        
        print(f"Maximum iterations: {controller.max_iterations}")
        print(f"Maximum verifications per prompt: {controller.max_verifications}")
        print("Starting processing...\n")
        
        # Initialize metrics
        metrics = None
        successful_prompts = 0
        
        for prompt in prompts:
            print(f"\n{'='*50}")
            print(f"Processing: {prompt}")
            print(f"{'='*50}")
            start_time = time.time()
            
            try:
                result = controller.execute(prompt)
                processing_time = time.time() - start_time
                
                # Get performance metrics
                metrics = controller.get_validation_metrics()
                successful_prompts += 1
                
                print(f"\nFINAL OUTPUT: {result}")
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"Quality score: {metrics.quality_score:.2f}")
                
            except Exception as e:
                print(f"Error processing prompt: {str(e)}")
                # Continue to the next prompt even if this one fails
                continue
        
        # Get final performance report
        try:
            report = controller.get_performance_report()
            print("\n=== Final Performance Report ===")
            print(f"Total prompts processed: {len(prompts)}")
            print(f"Successful prompts: {successful_prompts}")
            print(f"Cache entries: {report['cache_stats']['entries']}")
            
            if metrics is not None:
                print(f"Average quality score: {metrics.quality_score:.2f}")
            else:
                print("No successful prompts processed - quality score unavailable")
        except Exception as e:
            print(f"Error generating performance report: {str(e)}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
"""
Example using local models with LLMOrchestrator.
"""
import time
import sys
from typing import Optional
import torch
import logging

from LLMOrchestrator.controller import Controller, PromptTemplate
from LLMOrchestrator.models.local_model import LocalModel
from LLMOrchestrator.verifier import Verifier

def create_simple_verifier():
    """Create a very simple verifier that just checks if output exists."""
    def verify_quality(text: str) -> tuple[bool, str]:
        if not text or not text.strip():
            return False, "Output is empty"
        
        # Add a very basic length check just for demonstration
        if len(text.strip()) < 10:
            return False, f"Output too short ({len(text.strip())} chars)"
            
        return True, text
    
    return Verifier(custom_verifier=verify_quality)

def main():
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== LLMOrchestrator Demo (Local Model) ===")
        print(f"Using device: {device}")
        
        try:
            # Initialize local model with intentionally small max_length
            # to demonstrate verification failures and retries
            model = LocalModel(
                model_name="facebook/opt-125m",  # Small model for testing
                device=device,
                max_length=25,  # Intentionally small to potentially trigger retries
                temperature=0.7
            )
            
            # Create a prompt template
            template = PromptTemplate(
                "In one sentence only: {prompt}"
            )
            
            # Create a simple verifier
            verifier = create_simple_verifier()
            
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
                
            # Print memory usage if using CUDA
            if device == "cuda":
                print(f"\n=== GPU Memory Usage ===")
                print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error generating performance report: {str(e)}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
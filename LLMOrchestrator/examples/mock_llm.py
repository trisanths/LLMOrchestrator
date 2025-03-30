"""
Mock LLM components for examples.
"""
import time
import random
from typing import Optional, Tuple

class MockLLMGenerator:
    """Mock LLM generator that simulates API calls with realistic delays."""
    def __init__(self, delay_range: tuple[float, float] = (0.01, 0.1)):
        self.delay_range = delay_range
        
    def generate_output(self, prompt: str) -> str:
        # Simulate API call delay with shorter times
        time.sleep(random.uniform(*self.delay_range))
        
        # Generate mock response based on prompt
        if "invalid" in prompt.lower():
            return "This is an invalid response that should fail verification."
        
        # Generate a more realistic response
        topics = {
            "ai": ["machine learning", "neural networks", "deep learning"],
            "quantum": ["quantum computing", "quantum mechanics", "quantum physics"],
            "blockchain": ["distributed ledger", "cryptocurrency", "smart contracts"],
            "climate": ["global warming", "renewable energy", "carbon emissions"],
            "space": ["astronomy", "space exploration", "cosmology"],
            "finance": ["investment", "trading", "market analysis"],
            "healthcare": ["medical technology", "patient care", "health systems"],
            "privacy": ["data protection", "cybersecurity", "personal information"]
        }
        
        # Extract relevant topics from prompt
        prompt_topics = []
        for topic, keywords in topics.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                prompt_topics.append(topic)
        
        if not prompt_topics:
            prompt_topics = ["general"]
        
        # Generate response
        response = f"Analysis of {prompt}:\n\n"
        for topic in prompt_topics:
            if topic in topics:
                response += f"- Key aspects of {topic}: {', '.join(random.sample(topics[topic], 2))}\n"
        
        response += "\nThis is a mock response generated for demonstration purposes."
        return response

class MockLLMVerifier:
    """Mock verifier that checks output quality and validity."""
    def __init__(self, delay_range: tuple[float, float] = (0.01, 0.05)):
        self.delay_range = delay_range
        
    def verify(self, text: str, prompt: str = None) -> Tuple[bool, str]:
        # Simulate verification delay with shorter times
        time.sleep(random.uniform(*self.delay_range))
        
        try:
            # Check for invalid responses
            if "invalid" in text.lower() or "warning" in text.lower():
                return False, "Response failed verification checks"
            
            # Check for relevance if prompt is provided
            if prompt and len(prompt) > 5:
                prompt_keywords = [word.lower() for word in prompt.split() if len(word) > 3]
                if prompt_keywords:
                    found_keywords = [kw for kw in prompt_keywords if kw in text.lower()]
                    if not found_keywords:
                        return False, f"Response not relevant to prompt about {', '.join(prompt_keywords)}"
            
            # Basic quality checks
            if len(text.split()) < 10:
                return False, "Response too short"
            
            if not any(char in text for char in ['.', '!', '?']):
                return False, "Response lacks proper sentence structure"
            
            # Simulate quality score
            quality_score = random.uniform(0.6, 0.95)
            
            # Add quality metrics to response
            enhanced_response = f"{text}\n\nQuality Score: {quality_score:.2f}"
            return True, enhanced_response
            
        except Exception as e:
            return False, f"Verification error: {str(e)}" 
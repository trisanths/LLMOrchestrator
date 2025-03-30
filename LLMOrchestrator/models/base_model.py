from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseModel(ABC):
    """Base class for all LLM models."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        
    @abstractmethod
    def generate_output(self, prompt: str, **kwargs) -> str:
        """Generate output from the model given a prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated output as string
        """
        pass
    
    @abstractmethod
    def validate_output(self, output: str) -> tuple[bool, str]:
        """Validate the generated output.
        
        Args:
            output: The generated output to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "config": self.config
        } 
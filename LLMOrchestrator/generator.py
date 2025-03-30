from .models import BaseModel
import logging
import time

class Generator:
    """
    Default generator that uses an LLM model to produce outputs.
    
    Methods:
      - generate_output(prompt: str = None) -> str
    """
    DEFAULT_PROMPT = "Please provide your detailed reasoning on the given topic."

    def __init__(self, model: BaseModel):
        self.model = model
        self.logger = logging.getLogger(__name__)

    def generate_output(self, prompt: str = None) -> str:
        """Generate output using the provided model."""
        prompt = prompt or self.DEFAULT_PROMPT
        try:
            start_time = time.time()
            self.logger.debug(f"Generating output for prompt: {prompt[:50]}...")
            output = self.model.generate_output(prompt)
            elapsed = time.time() - start_time
            self.logger.debug(f"Generation completed in {elapsed:.2f}s")
            return output
        except Exception as e:
            self.logger.error(f"Error generating output: {str(e)}")
            raise Exception(f"Generation failed: {str(e)}")

class CustomGenerator:
    """
    Custom generator that uses a provided function to produce outputs.
    
    Methods:
      - generate_output(prompt: str = None) -> str
    """
    def __init__(self, custom_func):
        self.custom_func = custom_func
        self.logger = logging.getLogger(__name__)

    def generate_output(self, prompt: str = None) -> str:
        """Generate output using the custom function."""
        try:
            start_time = time.time()
            self.logger.debug(f"Using custom generator for prompt: {prompt[:50]}...")
            output = self.custom_func(prompt)
            elapsed = time.time() - start_time
            self.logger.debug(f"Custom generation completed in {elapsed:.2f}s")
            return output
        except Exception as e:
            self.logger.error(f"Error in custom generator: {str(e)}")
            raise Exception(f"Custom generation failed: {str(e)}")

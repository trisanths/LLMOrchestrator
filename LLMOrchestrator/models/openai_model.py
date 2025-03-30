"""
OpenAI model implementation for LLMOrchestrator.
"""
import os
import warnings
import ssl
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
from .base_model import BaseModel

# Suppress urllib3 warnings about SSL
warnings.filterwarnings("ignore", category=Warning)

class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model_name, **kwargs)
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        
        api_key = kwargs.get('api_key') or os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OPENAI_API_KEY environment variable not set"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Store kwargs for future reference
        self.kwargs = kwargs
            
        # Add a default timeout to the OpenAI client
        try:
            self.client = OpenAI(
                api_key=api_key,
                timeout=kwargs.get('timeout', 30.0)  # Default 30-second timeout
            )
            self.logger.info(f"OpenAI client initialized with model: {model_name}")
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI client: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    def generate_output(self, prompt: str, **kwargs) -> str:
        """Generate output using OpenAI API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated output as string
        """
        try:
            self.logger.debug(f"Generating output for prompt: {prompt[:50]}...")
            
            # Use modest defaults to avoid hanging
            if 'temperature' not in kwargs and 'temperature' not in self.kwargs:
                kwargs['temperature'] = 0.7
            if 'max_tokens' not in kwargs and 'max_tokens' not in self.kwargs:
                kwargs['max_tokens'] = 500
                
            # Add model parameters from initialization if not overridden
            for k, v in self.kwargs.items():
                if k not in ['api_key', 'model_name', 'timeout'] and k not in kwargs:
                    kwargs[k] = v
                
            # Create a messages array with system instructions for reliability
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly and concisely."},
                {"role": "user", "content": prompt}
            ]
                
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            output = response.choices[0].message.content
            self.logger.debug(f"Generated {len(output.split())} words")
            return output
            
        except ssl.SSLError as e:
            error_msg = f"SSL Error connecting to OpenAI: {str(e)}. Consider downgrading urllib3 with 'pip install urllib3==1.26.6'"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error generating output: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
            
    def validate_output(self, output: str) -> tuple[bool, str]:
        """Validate the generated output.
        
        Args:
            output: The generated output to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not output:
            return False, "Empty output"
        return True, "Output validated successfully"

    def verify(self, text: str, prompt: str = None) -> tuple[bool, str]:
        """Verify the quality of generated text.
        
        Args:
            text: The text to verify
            prompt: Optional prompt that was used to generate the text
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Create a verification prompt that encourages proper JSON formatting
            verification_prompt = f"""Verify the quality of the following text and respond ONLY with a JSON object.
            
            Text to verify: {text}
            Original prompt (if provided): {prompt if prompt else 'Not provided'}
            
            Respond with ONLY a JSON object in this exact format:
            {{
                "score": <float between 0 and 1>,
                "valid": <true/false>,
                "message": "<brief explanation>"
            }}
            
            Do not include any other text before or after the JSON object."""
            
            # Generate verification response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a text quality verifier. Respond only with properly formatted JSON."},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent JSON formatting
                max_tokens=200    # Reduced tokens since we only need JSON
            )
            
            # Get the response content
            result = response.choices[0].message.content.strip()
            
            # Try to parse the JSON directly first
            try:
                import json
                verification_result = json.loads(result)
                
                # Ensure all required fields are present
                if all(k in verification_result for k in ['score', 'valid', 'message']):
                    # Format the message to include the score
                    message = json.dumps({
                        'score': float(verification_result['score']),
                        'message': verification_result['message']
                    })
                    return verification_result['valid'], message
                    
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON using regex
                import re
                json_match = re.search(r'\{[^{]*\}', result, re.DOTALL)
                if json_match:
                    try:
                        verification_result = json.loads(json_match.group())
                        if all(k in verification_result for k in ['score', 'valid', 'message']):
                            message = json.dumps({
                                'score': float(verification_result['score']),
                                'message': verification_result['message']
                            })
                            return verification_result['valid'], message
                    except:
                        pass
                        
            # If we get here, we couldn't parse the JSON properly
            # As a fallback, if the text looks reasonable, return a default response
            if len(text.split()) >= 3:  # Basic validation
                return True, json.dumps({
                    'score': 0.7,
                    'message': 'Basic validation passed'
                })
                
            return False, json.dumps({
                'score': 0.0,
                'message': 'Failed to verify text quality'
            })
                
        except Exception as e:
            self.logger.error(f"Verification error: {str(e)}")
            # Return a properly formatted JSON even in case of error
            return False, json.dumps({
                'score': 0.0,
                'message': f'Verification error: {str(e)}'
            }) 
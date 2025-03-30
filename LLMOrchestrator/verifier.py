import logging
import time

class Verifier:
    """
    Default verifier that checks if generated output is non-empty.
    
    Methods:
      - verify(text: str, prompt: str = None) -> (bool, str)
    """
    def __init__(self, custom_verifier=None):
        self.custom_verifier = custom_verifier
        self.logger = logging.getLogger(__name__)

    def verify(self, text: str, prompt: str = None) -> tuple[bool, str]:
        try:
            start_time = time.time()
            self.logger.debug(f"Starting verification, text length: {len(text)}")
            
            if self.custom_verifier:
                # Pass the prompt to the custom verifier if it accepts it
                try:
                    result = self.custom_verifier(text, prompt)
                    elapsed = time.time() - start_time
                    self.logger.debug(f"Custom verification completed in {elapsed:.2f}s, result: {result[0]}")
                    return result
                except TypeError:
                    # Fallback if the custom verifier doesn't accept prompt parameter
                    result = self.custom_verifier(text)
                    elapsed = time.time() - start_time
                    self.logger.debug(f"Custom verification (no prompt) completed in {elapsed:.2f}s, result: {result[0]}")
                    return result
                
            # Basic validation - extremely lenient
            if not text or not text.strip():
                self.logger.warning("Verification failed: Empty output")
                return False, "Output is empty"
                
            # Any non-empty text passes verification by default
            elapsed = time.time() - start_time
            self.logger.debug(f"Basic verification completed in {elapsed:.2f}s")
            return True, text
        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            return False, f"Verification error: {str(e)}"

class CustomVerifier:
    """
    Wrapper for a custom verification function.
    
    Parameters:
      custom_func: A function that takes output (str) and returns (bool, str).
    """
    def __init__(self, custom_func):
        self.custom_func = custom_func
        self.logger = logging.getLogger(__name__)

    def verify(self, text: str, prompt: str = None) -> tuple[bool, str]:
        try:
            start_time = time.time()
            self.logger.debug(f"Starting custom verification, text length: {len(text)}")
            
            try:
                result = self.custom_func(text, prompt)
                elapsed = time.time() - start_time
                self.logger.debug(f"Custom verification completed in {elapsed:.2f}s, result: {result[0]}")
                return result
            except TypeError:
                # Fallback if the custom verifier doesn't accept prompt parameter
                result = self.custom_func(text)
                elapsed = time.time() - start_time
                self.logger.debug(f"Custom verification (no prompt) completed in {elapsed:.2f}s, result: {result[0]}")
                return result
        except Exception as e:
            self.logger.error(f"Error in custom verification: {str(e)}")
            return False, f"Custom verification error: {str(e)}"

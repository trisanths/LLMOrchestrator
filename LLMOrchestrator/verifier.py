import logging
import time
import json

class Verifier:
    """
    Default verifier that checks if generated output is non-empty
    and returns a boolean plus a JSON message with a numeric score.
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

            # Delegate to custom verifier if provided
            if self.custom_verifier:
                try:
                    valid, score = self.custom_verifier(text, prompt)
                except TypeError:
                    valid, score = self.custom_verifier(text)
                elapsed = time.time() - start_time
                self.logger.debug(f"Custom verification completed in {elapsed:.2f}s, valid={valid}, score={score}")
                # Ensure score is numeric
                try:
                    score = float(score)
                except Exception:
                    score = 0.0
                message = json.dumps({"score": score})
                return valid, message

            # Basic validation: non-empty text passes with perfect score
            if not text or not text.strip():
                self.logger.warning("Verification failed: Empty output")
                return False, json.dumps({"score": 0.0})

            # Any non-empty text passes with score 1.0
            elapsed = time.time() - start_time
            self.logger.debug(f"Basic verification completed in {elapsed:.2f}s")
            return True, json.dumps({"score": 1.0})

        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            # On exception, treat as failure with zero score
            return False, json.dumps({"score": 0.0})

class CustomVerifier:
    """
    Wrapper for a custom verification function.
    Parameters:
      custom_func: A function that takes output (str) [, prompt: str] and returns (bool, float).
    """
    def __init__(self, custom_func):
        self.custom_func = custom_func
        self.logger = logging.getLogger(__name__)

    def verify(self, text: str, prompt: str = None) -> tuple[bool, str]:
        try:
            start_time = time.time()
            self.logger.debug(f"Starting custom verification, text length: {len(text)}")

            try:
                valid, score = self.custom_func(text, prompt)
            except TypeError:
                valid, score = self.custom_func(text)

            elapsed = time.time() - start_time
            self.logger.debug(f"Custom verification completed in {elapsed:.2f}s, valid={valid}, score={score}")

            # Ensure score is numeric
            try:
                score = float(score)
            except Exception:
                score = 0.0
            message = json.dumps({"score": score})
            return valid, message

        except Exception as e:
            self.logger.error(f"Error in custom verification: {str(e)}")
            return False, json.dumps({"score": 0.0})

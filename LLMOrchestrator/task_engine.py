import logging
from typing import Optional, Dict, Any
from .controller import Controller

class TaskEngine:
    """
    High-level API to run the reasoning task using the controller.
    
    Methods:
      - run_task(prompt: str = None, n: int = None, stop_early: bool = False) -> str
    """
    def __init__(
        self, 
        generator, 
        verifier, 
        max_iterations: int = 3, 
        max_verifications: int = 5,
        refinement_generator = None, 
        dynamic_iterations = None,
        cache_enabled: bool = True,
        monitoring_enabled: bool = True
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing TaskEngine")
        
        self.controller = Controller(
            generator=generator, 
            verifier=verifier, 
            max_iterations=max_iterations,
            max_verifications=max_verifications,
            refinement_generator=refinement_generator, 
            dynamic_iterations=dynamic_iterations,
            cache_enabled=cache_enabled,
            monitoring_enabled=monitoring_enabled
        )
        
        self.logger.debug(f"Controller initialized with max_iterations={max_iterations}, max_verifications={max_verifications}")

    def run_task(self, prompt: str = None, n: int = None, stop_early: bool = False) -> str:
        """
        Run a task with the given prompt.
        
        Args:
            prompt: The input prompt
            n: Optional number of iterations to override controller's default
            stop_early: If True, will return early when a good enough result is found
            
        Returns:
            Generated and verified output
        """
        try:
            self.logger.info(f"Running task with prompt: {prompt[:50]}...")
            result = self.controller.execute(prompt, n, stop_early)
            self.logger.info("Task completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error running task: {str(e)}")
            raise
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current performance metrics."""
        try:
            metrics = self.controller.get_validation_metrics()
            report = self.controller.get_performance_report()
            self.logger.debug("Retrieved performance metrics")
            return {
                'validation_metrics': metrics.__dict__ if metrics else {},
                'performance_report': report
            }
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {
                'error': str(e)
            }

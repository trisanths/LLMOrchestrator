from typing import Optional, Callable, Dict, Any, List, Tuple, Set, Union
import concurrent.futures
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from enum import Enum
import threading
from collections import defaultdict
import re

class OutputQuality(Enum):
    EXCELLENT = 4
    GOOD = 3
    FAIR = 2
    POOR = 1

@dataclass
class ValidationMetrics:
    """Enhanced metrics for output validation and quality assessment."""
    confidence_score: float
    processing_time: float
    token_count: int
    refinement_count: int
    validation_checks: List[str]
    quality_score: float = 0.0
    semantic_similarity: float = 0.0
    coherence_score: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class AdaptiveLearning:
    """Manages adaptive learning parameters based on performance."""
    def __init__(self, initial_learning_rate: float = 0.1):
        self.learning_rate = initial_learning_rate
        self.performance_history: List[float] = []
        self.parameter_history: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def update_parameters(self, metrics: ValidationMetrics):
        """Update learning parameters based on performance metrics."""
        with self.lock:
            self.performance_history.append(metrics.quality_score)
            if len(self.performance_history) > 10:
                self.performance_history.pop(0)
            
            # Adjust learning rate based on performance trend
            if len(self.performance_history) >= 2:
                trend = self.performance_history[-1] - self.performance_history[-2]
                self.learning_rate *= (1 + trend * 0.1)
                self.learning_rate = max(0.01, min(1.0, self.learning_rate))

    def get_optimal_parameters(self) -> Dict[str, float]:
        """Get optimized parameters based on learning history."""
        return {
            'learning_rate': self.learning_rate,
            'performance_trend': np.mean(self.performance_history) if self.performance_history else 0.0
        }

class PromptTemplate:
    """Enhanced prompt template management with dynamic adaptation."""
    def __init__(self, base_template: str):
        self.base_template = base_template
        self.variations: Dict[str, str] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def add_variation(self, name: str, template: str):
        self.variations[name] = template
    
    def get_template(self, variation: str = None) -> str:
        return self.variations.get(variation, self.base_template)
    
    def record_performance(self, variation: str, quality_score: float):
        """Record performance metrics for template variations."""
        with self.lock:
            self.performance_metrics[variation].append(quality_score)
            if len(self.performance_metrics[variation]) > 100:
                self.performance_metrics[variation].pop(0)
    
    def get_best_variation(self) -> str:
        """Get the best performing template variation."""
        if not self.performance_metrics:
            return None
        avg_scores = {
            var: np.mean(scores) 
            for var, scores in self.performance_metrics.items()
        }
        return max(avg_scores.items(), key=lambda x: x[1])[0]

class OutputCache:
    """Enhanced caching system with quality-based retention."""
    def __init__(self, cache_dir: str = ".cache", max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "output_cache.json"
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.lock = threading.Lock()
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        with self.lock:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: Dict[str, Any]):
        with self.lock:
            # Check cache size and remove low-quality entries if needed
            if self._get_cache_size() > self.max_size_bytes:
                self._prune_cache()
            
            self.cache[key] = value
            self._save_cache()
    
    def _get_cache_size(self) -> int:
        return sum(len(str(v).encode('utf-8')) for v in self.cache.values())
    
    def _prune_cache(self):
        """Remove low-quality entries to maintain size limit."""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].get('metrics', {}).get('quality_score', 0),
            reverse=True
        )
        while self._get_cache_size() > self.max_size_bytes and sorted_entries:
            self.cache.pop(sorted_entries.pop()[0])

class Controller:
    """
    Enhanced controller with advanced features for LLM orchestration.
    """
    def __init__(
        self,
        generator,
        verifier,
        max_iterations: int = 3,
        max_verifications: int = 5,  # New parameter for verification limit
        refinement_generator=None,
        dynamic_iterations=None,
        parallel_processing: bool = False,
        cache_enabled: bool = True,
        retry_attempts: int = 2,
        prompt_template: Optional[PromptTemplate] = None,
        adaptive_learning: bool = True,
        monitoring_enabled: bool = True
    ):
        self.generator = generator
        self.verifier = verifier
        self.max_iterations = max_iterations
        self.max_verifications = max_verifications  # Store the verification limit
        self.refinement_generator = refinement_generator
        self.dynamic_iterations = dynamic_iterations
        self.parallel_processing = parallel_processing
        self.cache_enabled = cache_enabled
        self.retry_attempts = retry_attempts
        self.prompt_template = prompt_template
        self.cache = OutputCache() if cache_enabled else None
        self.metrics = ValidationMetrics(0.0, 0.0, 0, 0, [])
        self.adaptive_learning = AdaptiveLearning() if adaptive_learning else None
        self.monitoring_enabled = monitoring_enabled
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for monitoring and debugging."""
        if self.monitoring_enabled:
            # Create a filter to exclude HTTP request logs
            class HTTPFilter(logging.Filter):
                def filter(self, record):
                    return not any(x in record.msg.lower() for x in [
                        'http', 'request', 'response', 'status', 'headers',
                        'urllib3', 'requests', 'openai', 'api'
                    ])

            # Setup root logger
            root_logger = logging.getLogger()
            # Revert level to INFO for normal operation
            root_logger.setLevel(logging.INFO)

            # Clear any existing handlers
            root_logger.handlers = []

            # Create console handler with standard formatter
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            console_handler.addFilter(HTTPFilter())
            root_logger.addHandler(console_handler)

            # Create file handler for all logs (including HTTP)
            file_handler = logging.FileHandler('llm_orchestrator.log')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(file_handler)

            # Create a separate logger for this class
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

    def _process_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Enhanced retry logic with adaptive backoff."""
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    raise
                backoff = 2 ** attempt * (1 + np.random.random())  # Jittered exponential backoff
                time.sleep(backoff)

    def _get_cached_output(self, prompt: str) -> Optional[str]:
        """Enhanced cache retrieval with quality checks."""
        if not self.cache_enabled or not self.cache:
            return None
        cached = self.cache.get(prompt)
        if cached:
            if time.time() - cached['timestamp'] < 3600:  # 1-hour cache
                if cached.get('metrics', {}).get('quality_score', 0) >= 0.7:  # Quality threshold
                    return cached['output']
        return None

    def _cache_output(self, prompt: str, output: str):
        """Enhanced caching with quality metrics."""
        if not self.cache_enabled or not self.cache:
            return
        self.cache.set(prompt, {
            'output': output,
            'timestamp': time.time(),
            'metrics': self.metrics.__dict__
        })

    def _calculate_quality_metrics(self, output: str) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for output."""
        # Placeholder for actual quality calculation logic
        return {
            'quality_score': 0.8,
            'semantic_similarity': 0.85,
            'coherence_score': 0.9,
            'error_rate': 0.05
        }

    def execute(self, prompt: str, stop_early: bool = False) -> str:
        """Execute a single prompt with the configured models."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        self.logger.info(f"Starting execution for prompt: {prompt[:50]}...")
        
        best_output = None
        best_quality = 0.0
        completed_iterations = 0
        
        for i in range(self.max_iterations):
            self.logger.info(f"Starting iteration {i+1}/{self.max_iterations}")
            print(f"\nITERATION {i+1}/{self.max_iterations}")
            
            # Generate response
            print("   Generating response... (timeout=30s)")
            try:
                response = self.generator.generate_output(prompt)
                if not response or not response.strip():
                    self.logger.warning("Empty response from generator")
                    continue
                print(f"   Generated: {response[:100]}")
            except Exception as e:
                self.logger.error(f"Generation error: {e}")
                continue
                
            # Verify response
            valid_refined = False
            for v in range(self.max_verifications):
                print(f"   Verification attempt {v+1}/{self.max_verifications}")
                try:
                    valid, message = self.verifier.verify(response, prompt)
                    self.logger.debug(f"Verification result: valid={valid}, message={message}")
                    
                    if valid:
                        try:
                            # Try to extract quality score from message
                            quality_match = re.search(r'"score":\s*(\d+\.?\d*)', message)
                            if quality_match:
                                current_quality = float(quality_match.group(1))
                                print(f"   Verification PASSED! Quality: {current_quality:.2f}")
                                self.logger.info(f"Verification successful, quality score: {current_quality:.2f}")
                                
                                if current_quality > best_quality:
                                    best_quality = current_quality
                                    best_output = response
                                    print("   New best quality output")
                                    
                                    # If quality is high enough and stop_early is True, return immediately
                                    if current_quality > 0.9 and stop_early:
                                        self.logger.info("High quality achieved, stopping early")
                                        return best_output
                                        
                                valid_refined = True
                                break
                            else:
                                self.logger.warning("No quality score found in verification message")
                                # If verification passed but no score found, assume decent quality
                                current_quality = 0.7
                                if current_quality > best_quality:
                                    best_quality = current_quality
                                    best_output = response
                                    print("   New best quality output")
                                valid_refined = True
                                break
                        except (ValueError, IndexError) as e:
                            self.logger.warning(f"Error parsing quality score: {e}")
                            # If we can't parse the score but verification passed, assume decent quality
                            current_quality = 0.7
                            if current_quality > best_quality:
                                best_quality = current_quality
                                best_output = response
                                print("   New best quality output")
                            valid_refined = True
                            break
                    else:
                        print(f"   Verification FAILED: {message}")
                        self.logger.warning(f"Verification failed: {message}")
                except Exception as e:
                    self.logger.error(f"Verification error: {e}")
                    continue
                    
            if valid_refined:
                self.logger.info("Continuing to next iteration for potential quality improvement")
                print("   Continuing to next iteration for potential quality improvement")
            else:
                self.logger.warning("No valid refinement achieved in this iteration")
                print("   No valid refinement achieved in this iteration")
                
            completed_iterations = i
            
        # After all iterations, return the best output if we have one
        if best_output is not None:
            self.logger.info(f"Returning best output with quality score: {best_quality:.2f}")
            return best_output
        else:
            self.logger.warning("No valid outputs generated after all iterations")
            raise RuntimeError("No valid outputs generated after all iterations")

    def execute_parallel(self, prompts: List[str], max_workers: int = 3, stop_early: bool = False) -> List[str]:
        """
        Enhanced parallel execution with progress tracking.
        
        Args:
            prompts: List of prompts to process in parallel
            max_workers: Maximum number of parallel workers
            stop_early: If True, will return early when a good enough result is found
            
        Returns:
            List of generated outputs
        """
        if not self.parallel_processing:
            return [self.execute(prompt, stop_early=stop_early) for prompt in prompts]
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {executor.submit(self.execute, prompt, stop_early): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                    if self.logger:
                        self.logger.info(f"Completed parallel execution for prompt: {prompt[:50]}...")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in parallel execution for {prompt[:50]}: {str(e)}")
                    results.append(None)
        return results

    def _update_metrics(self, start_time: float, output: str):
        """Enhanced metrics update with comprehensive quality assessment."""
        quality_metrics = self._calculate_quality_metrics(output)
        self.metrics.processing_time = time.time() - start_time
        self.metrics.token_count = len(output.split())
        self.metrics.validation_checks.append("basic_validation")
        self.metrics.quality_score = quality_metrics['quality_score']
        self.metrics.semantic_similarity = quality_metrics['semantic_similarity']
        self.metrics.coherence_score = quality_metrics['coherence_score']
        self.metrics.error_rate = quality_metrics['error_rate']
        self.metrics.last_updated = datetime.now()

        if self.adaptive_learning:
            self.adaptive_learning.update_parameters(self.metrics)

        if self.prompt_template:
            self.prompt_template.record_performance(
                self.prompt_template.get_best_variation(),
                quality_metrics['quality_score']
            )

    def get_validation_metrics(self) -> ValidationMetrics:
        """Return current validation metrics."""
        return self.metrics

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'metrics': self.metrics.__dict__,
            'adaptive_learning': self.adaptive_learning.get_optimal_parameters() if self.adaptive_learning else None,
            'cache_stats': {
                'size': self.cache._get_cache_size() if self.cache else 0,
                'entries': len(self.cache.cache) if self.cache else 0
            },
            'prompt_template_stats': {
                'variations': len(self.prompt_template.variations) if self.prompt_template else 0,
                'best_variation': self.prompt_template.get_best_variation() if self.prompt_template else None
            }
        }

class CustomController(Controller):
    """
    Enhanced custom controller with additional features.
    """
    def __init__(
        self,
        custom_func: Callable,
        generator,
        verifier,
        parallel_processing: bool = False,
        cache_enabled: bool = True,
        adaptive_learning: bool = True,
        monitoring_enabled: bool = True
    ):
        super().__init__(
            generator,
            verifier,
            parallel_processing=parallel_processing,
            cache_enabled=cache_enabled,
            adaptive_learning=adaptive_learning,
            monitoring_enabled=monitoring_enabled
        )
        self.custom_func = custom_func

    def execute(self, prompt: str = None, n: int = None, stop_early: bool = False) -> str:
        """Execute custom processing function.
        
        Args:
            prompt: The input prompt
            n: Optional number of iterations to override controller's default
            stop_early: If True, will return early when a good enough result is found
            
        Returns:
            Generated and verified output
        """
        start_time = time.time()
        
        if cached_output := self._get_cached_output(prompt):
            return cached_output

        result = self.custom_func(self.generator, self.verifier, prompt, n)
        self._update_metrics(start_time, result)
        self._cache_output(prompt, result)
        return result

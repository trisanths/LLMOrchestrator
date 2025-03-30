import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import time

from ..controller import (
    Controller,
    CustomController,
    ValidationMetrics,
    AdaptiveLearning,
    PromptTemplate,
    OutputCache,
    OutputQuality
)

class MockGenerator:
    def generate_output(self, prompt: str) -> str:
        return f"Generated output for: {prompt}"

class MockVerifier:
    def verify(self, text: str) -> tuple[bool, str]:
        return True, text

class TestValidationMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = ValidationMetrics(
            confidence_score=0.8,
            processing_time=1.5,
            token_count=100,
            refinement_count=2,
            validation_checks=["check1", "check2"]
        )

    def test_metrics_initialization(self):
        self.assertEqual(self.metrics.confidence_score, 0.8)
        self.assertEqual(self.metrics.processing_time, 1.5)
        self.assertEqual(self.metrics.token_count, 100)
        self.assertEqual(self.metrics.refinement_count, 2)
        self.assertEqual(self.metrics.validation_checks, ["check1", "check2"])
        self.assertEqual(self.metrics.quality_score, 0.0)
        self.assertEqual(self.metrics.semantic_similarity, 0.0)
        self.assertEqual(self.metrics.coherence_score, 0.0)
        self.assertEqual(self.metrics.error_rate, 0.0)
        self.assertIsInstance(self.metrics.last_updated, datetime)

    def test_metrics_update(self):
        self.metrics.quality_score = 0.9
        self.metrics.semantic_similarity = 0.85
        self.metrics.coherence_score = 0.95
        self.metrics.error_rate = 0.05
        
        self.assertEqual(self.metrics.quality_score, 0.9)
        self.assertEqual(self.metrics.semantic_similarity, 0.85)
        self.assertEqual(self.metrics.coherence_score, 0.95)
        self.assertEqual(self.metrics.error_rate, 0.05)

class TestAdaptiveLearning(unittest.TestCase):
    def setUp(self):
        self.learning = AdaptiveLearning(initial_learning_rate=0.1)
        self.metrics = ValidationMetrics(
            confidence_score=0.8,
            processing_time=1.5,
            token_count=100,
            refinement_count=2,
            validation_checks=["check1"]
        )

    def test_initialization(self):
        self.assertEqual(self.learning.learning_rate, 0.1)
        self.assertEqual(len(self.learning.performance_history), 0)
        self.assertEqual(len(self.learning.parameter_history), 0)

    def test_update_parameters(self):
        self.metrics.quality_score = 0.8
        self.learning.update_parameters(self.metrics)
        self.assertEqual(len(self.learning.performance_history), 1)
        self.assertEqual(self.learning.performance_history[0], 0.8)

        self.metrics.quality_score = 0.9
        self.learning.update_parameters(self.metrics)
        self.assertEqual(len(self.learning.performance_history), 2)
        self.assertGreater(self.learning.learning_rate, 0.1)

    def test_get_optimal_parameters(self):
        self.metrics.quality_score = 0.8
        self.learning.update_parameters(self.metrics)
        params = self.learning.get_optimal_parameters()
        self.assertIn('learning_rate', params)
        self.assertIn('performance_trend', params)

    def test_learning_rate_bounds(self):
        # Test learning rate doesn't go below minimum
        self.metrics.quality_score = 0.1
        self.learning.update_parameters(self.metrics)
        self.assertGreaterEqual(self.learning.learning_rate, 0.01)
        
        # Test learning rate doesn't go above maximum
        self.metrics.quality_score = 1.0
        self.learning.update_parameters(self.metrics)
        self.assertLessEqual(self.learning.learning_rate, 1.0)

class TestPromptTemplate(unittest.TestCase):
    def setUp(self):
        self.template = PromptTemplate("Base template: {prompt}")
        self.template.add_variation("variation1", "Variation 1: {prompt}")
        self.template.add_variation("variation2", "Variation 2: {prompt}")

    def test_template_initialization(self):
        self.assertEqual(self.template.base_template, "Base template: {prompt}")
        self.assertEqual(len(self.template.variations), 2)

    def test_get_template(self):
        self.assertEqual(
            self.template.get_template(),
            "Base template: {prompt}"
        )
        self.assertEqual(
            self.template.get_template("variation1"),
            "Variation 1: {prompt}"
        )

    def test_record_performance(self):
        self.template.record_performance("variation1", 0.8)
        self.assertEqual(len(self.template.performance_metrics["variation1"]), 1)
        self.assertEqual(self.template.performance_metrics["variation1"][0], 0.8)

    def test_get_best_variation(self):
        self.template.record_performance("variation1", 0.8)
        self.template.record_performance("variation2", 0.9)
        self.assertEqual(self.template.get_best_variation(), "variation2")

    def test_template_history_limit(self):
        # Test that performance history is limited to 100 entries
        for i in range(150):
            self.template.record_performance("variation1", 0.8)
        self.assertEqual(len(self.template.performance_metrics["variation1"]), 100)

class TestOutputCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = OutputCache(cache_dir=self.temp_dir, max_size_mb=1)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_cache_initialization(self):
        self.assertEqual(len(self.cache.cache), 0)
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_cache_operations(self):
        test_data = {
            'output': 'test output',
            'timestamp': time.time(),
            'metrics': {'quality_score': 0.8}
        }
        self.cache.set('test_key', test_data)
        cached = self.cache.get('test_key')
        self.assertEqual(cached['output'], 'test output')
        self.assertEqual(cached['metrics']['quality_score'], 0.8)

    def test_cache_pruning(self):
        # Add multiple entries to exceed size limit
        for i in range(10):
            self.cache.set(f'key_{i}', {
                'output': 'x' * 100000,  # Large output
                'timestamp': time.time(),
                'metrics': {'quality_score': i / 10}
            })
        self.assertLess(len(self.cache.cache), 10)

    def test_cache_expiration(self):
        # Test cache entry expiration
        test_data = {
            'output': 'test output',
            'timestamp': time.time() - 7200,  # 2 hours old
            'metrics': {'quality_score': 0.8}
        }
        self.cache.set('test_key', test_data)
        cached = self.cache.get('test_key')
        self.assertIsNone(cached)

    def test_cache_quality_threshold(self):
        # Test cache quality threshold
        test_data = {
            'output': 'test output',
            'timestamp': time.time(),
            'metrics': {'quality_score': 0.5}  # Below threshold
        }
        self.cache.set('test_key', test_data)
        cached = self.cache.get('test_key')
        self.assertIsNone(cached)

class TestController(unittest.TestCase):
    def setUp(self):
        self.generator = MockGenerator()
        self.verifier = MockVerifier()
        self.controller = Controller(
            generator=self.generator,
            verifier=self.verifier,
            max_iterations=3,
            cache_enabled=True,
            monitoring_enabled=True
        )

    def test_controller_initialization(self):
        self.assertIsNotNone(self.controller.generator)
        self.assertIsNotNone(self.controller.verifier)
        self.assertEqual(self.controller.max_iterations, 3)
        self.assertTrue(self.controller.cache_enabled)
        self.assertTrue(self.controller.monitoring_enabled)

    def test_execute(self):
        result = self.controller.execute("test prompt")
        self.assertIsNotNone(result)
        self.assertIn("Generated output for: test prompt", result)

    def test_execute_with_cache(self):
        # First execution
        result1 = self.controller.execute("test prompt")
        # Second execution should use cache
        result2 = self.controller.execute("test prompt")
        self.assertEqual(result1, result2)

    def test_execute_parallel(self):
        prompts = ["prompt1", "prompt2", "prompt3"]
        results = self.controller.execute_parallel(prompts)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r is not None for r in results))

    def test_quality_metrics(self):
        result = self.controller.execute("test prompt")
        metrics = self.controller.get_validation_metrics()
        self.assertGreater(metrics.quality_score, 0)
        self.assertGreater(metrics.processing_time, 0)
        self.assertGreater(metrics.token_count, 0)

    def test_max_iterations(self):
        # Test that controller respects max iterations
        self.controller.max_iterations = 1
        result = self.controller.execute("test prompt")
        self.assertIsNotNone(result)

    def test_dynamic_iterations(self):
        # Test dynamic iteration adjustment
        def dynamic_iterations(prompt):
            return 2 if "complex" in prompt else 1
        
        controller = Controller(
            generator=self.generator,
            verifier=self.verifier,
            dynamic_iterations=dynamic_iterations
        )
        
        result1 = controller.execute("simple prompt")
        result2 = controller.execute("complex prompt")
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)

class TestCustomController(unittest.TestCase):
    def setUp(self):
        self.generator = MockGenerator()
        self.verifier = MockVerifier()
        self.custom_func = lambda g, v, p, n: f"Custom output for: {p}"
        self.controller = CustomController(
            custom_func=self.custom_func,
            generator=self.generator,
            verifier=self.verifier
        )

    def test_custom_controller_initialization(self):
        self.assertIsNotNone(self.controller.custom_func)
        self.assertIsNotNone(self.controller.generator)
        self.assertIsNotNone(self.controller.verifier)

    def test_custom_execute(self):
        result = self.controller.execute("test prompt")
        self.assertEqual(result, "Custom output for: test prompt")

    def test_custom_execute_with_cache(self):
        # First execution
        result1 = self.controller.execute("test prompt")
        # Second execution should use cache
        result2 = self.controller.execute("test prompt")
        self.assertEqual(result1, result2)

class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        self.generator = Mock()
        self.verifier = Mock()
        self.controller = Controller(
            generator=self.generator,
            verifier=self.verifier,
            retry_attempts=2
        )

    def test_generator_error(self):
        self.generator.generate_output.side_effect = Exception("Generator error")
        with self.assertRaises(Exception):
            self.controller.execute("test prompt")

    def test_verifier_error(self):
        self.verifier.verify.side_effect = Exception("Verifier error")
        with self.assertRaises(Exception):
            self.controller.execute("test prompt")

    def test_cache_error(self):
        with patch.object(OutputCache, 'get', side_effect=Exception("Cache error")):
            result = self.controller.execute("test prompt")
            self.assertIsNotNone(result)

    def test_retry_mechanism(self):
        # Test that retry mechanism works
        self.generator.generate_output.side_effect = [
            Exception("First attempt failed"),
            "Success on second attempt"
        ]
        result = self.controller.execute("test prompt")
        self.assertEqual(result, "Success on second attempt")

class TestPerformanceMonitoring(unittest.TestCase):
    def setUp(self):
        self.generator = MockGenerator()
        self.verifier = MockVerifier()
        self.controller = Controller(
            generator=self.generator,
            verifier=self.verifier,
            monitoring_enabled=True
        )

    def test_performance_report(self):
        self.controller.execute("test prompt")
        report = self.controller.get_performance_report()
        self.assertIn('metrics', report)
        self.assertIn('cache_stats', report)
        self.assertIn('prompt_template_stats', report)

    def test_metrics_update(self):
        self.controller.execute("test prompt")
        metrics = self.controller.get_validation_metrics()
        self.assertGreater(metrics.processing_time, 0)
        self.assertGreater(metrics.token_count, 0)
        self.assertGreater(metrics.quality_score, 0)

    def test_continuous_monitoring(self):
        # Test metrics accumulation over multiple executions
        for _ in range(3):
            self.controller.execute("test prompt")
        metrics = self.controller.get_validation_metrics()
        self.assertGreater(metrics.processing_time, 0)
        self.assertGreater(metrics.token_count, 0)

if __name__ == '__main__':
    unittest.main() 
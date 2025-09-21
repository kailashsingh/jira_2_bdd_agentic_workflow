"""
Test accuracy validation and execution time optimization for BDD generation models.
Provides automated validation, performance optimization, and accuracy measurement tools.
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import cProfile
import pstats
import io
from functools import wraps
import gc
import sys

from src.models.model_manager import ModelManager, BDDGenerationResult
from src.benchmarking.benchmark_framework import ModelBenchmarkFramework, BenchmarkMetrics
from src.utils.test_helpers import (
    TestDataGenerator, 
    ValidationHelper,
    PerformanceTimer,
    TestFileManager
)

logger = logging.getLogger(__name__)

@dataclass
class AccuracyMetrics:
    """Accuracy metrics for BDD generation validation."""
    model_name: str
    test_suite: str
    
    # Content accuracy
    feature_structure_accuracy: float
    step_definition_accuracy: float
    scenario_completeness: float
    
    # Semantic accuracy  
    requirement_coverage: float
    business_logic_accuracy: float
    technical_accuracy: float
    
    # Syntax accuracy
    gherkin_syntax_score: float
    pytest_bdd_compatibility: float
    
    # Overall scores
    overall_accuracy: float
    confidence_score: float

@dataclass
class OptimizationResult:
    """Result of performance optimization."""
    original_time: float
    optimized_time: float
    improvement_ratio: float
    memory_before: float
    memory_after: float
    optimization_techniques: List[str]
    recommendations: List[str]

class AccuracyValidator:
    """Validates accuracy of BDD generation models."""
    
    def __init__(self):
        """Initialize the accuracy validator."""
        self.model_manager = ModelManager()
        self.data_generator = TestDataGenerator()
        self.reference_standards = self._load_reference_standards()
        
    def _load_reference_standards(self) -> Dict[str, Any]:
        """Load reference standards for accuracy validation."""
        return {
            "gherkin_keywords": [
                "Feature:", "Scenario:", "Scenario Outline:", "Background:",
                "Given", "When", "Then", "And", "But"
            ],
            "required_elements": {
                "feature": ["Feature:", "Scenario:"],
                "scenario": ["Given", "When", "Then"],
                "step_definitions": ["@given", "@when", "@then", "def "]
            },
            "quality_thresholds": {
                "minimum_accuracy": 0.7,
                "good_accuracy": 0.8,
                "excellent_accuracy": 0.9
            },
            "semantic_patterns": {
                "user_story_patterns": [
                    r"As a .+ I want .+ so that .+",
                    r"Given .+ when .+ then .+",
                    r"In order to .+ as a .+ I want .+"
                ],
                "action_patterns": [
                    r"(click|enter|select|navigate|submit|login|logout|search)",
                    r"(should|must|will|can|may) (be|have|see|receive)"
                ]
            }
        }
    
    def validate_bdd_accuracy(self, 
                            model_name: str,
                            test_tickets: List[Dict[str, Any]],
                            reference_scenarios: Optional[List[str]] = None) -> AccuracyMetrics:
        """Validate BDD generation accuracy for a model."""
        logger.info(f"Validating accuracy for model: {model_name}")
        
        all_results = []
        all_references = reference_scenarios or []
        
        # Generate BDD for all test tickets
        for i, ticket in enumerate(test_tickets):
            try:
                result = self.model_manager.generate_bdd_with_model(ticket, model_name)
                all_results.append(result)
                
                # Generate reference if not provided
                if not reference_scenarios or i >= len(reference_scenarios):
                    reference = self._generate_reference_bdd(ticket)
                    all_references.append(reference)
                    
            except Exception as e:
                logger.error(f"Failed to generate BDD for ticket {ticket.get('key', 'unknown')}: {str(e)}")
                continue
        
        if not all_results:
            return self._create_failed_accuracy_metrics(model_name, "No successful generations")
        
        # Calculate accuracy metrics
        feature_accuracy = self._calculate_feature_structure_accuracy(all_results)
        step_accuracy = self._calculate_step_definition_accuracy(all_results)
        scenario_completeness = self._calculate_scenario_completeness(all_results, test_tickets)
        
        requirement_coverage = self._calculate_requirement_coverage(all_results, test_tickets)
        business_logic_accuracy = self._calculate_business_logic_accuracy(all_results, test_tickets)
        technical_accuracy = self._calculate_technical_accuracy(all_results)
        
        gherkin_syntax = self._calculate_gherkin_syntax_score(all_results)
        pytest_compatibility = self._calculate_pytest_bdd_compatibility(all_results)
        
        # Calculate overall accuracy
        accuracy_components = [
            feature_accuracy, step_accuracy, scenario_completeness,
            requirement_coverage, business_logic_accuracy, technical_accuracy,
            gherkin_syntax, pytest_compatibility
        ]
        
        overall_accuracy = statistics.mean(accuracy_components)
        
        # Calculate confidence score based on consistency
        confidence_score = self._calculate_confidence_score(all_results, accuracy_components)
        
        return AccuracyMetrics(
            model_name=model_name,
            test_suite=f"validation_suite_{len(test_tickets)}_tickets",
            feature_structure_accuracy=feature_accuracy,
            step_definition_accuracy=step_accuracy,
            scenario_completeness=scenario_completeness,
            requirement_coverage=requirement_coverage,
            business_logic_accuracy=business_logic_accuracy,
            technical_accuracy=technical_accuracy,
            gherkin_syntax_score=gherkin_syntax,
            pytest_bdd_compatibility=pytest_compatibility,
            overall_accuracy=overall_accuracy,
            confidence_score=confidence_score
        )
    
    def _calculate_feature_structure_accuracy(self, results: List[BDDGenerationResult]) -> float:
        """Calculate accuracy of feature file structure."""
        scores = []
        
        for result in results:
            validation = ValidationHelper.validate_bdd_feature(result.feature_content)
            
            score = 0.0
            if validation["is_valid"]:
                score += 0.5
            
            # Check for proper structure
            content = result.feature_content
            if "Feature:" in content:
                score += 0.2
            if "Scenario:" in content:
                score += 0.2
            if validation["has_all_keywords"]:
                score += 0.1
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_step_definition_accuracy(self, results: List[BDDGenerationResult]) -> float:
        """Calculate accuracy of step definitions."""
        scores = []
        
        for result in results:
            validation = ValidationHelper.validate_step_definitions(result.step_definitions)
            
            score = 0.0
            if validation["is_valid"]:
                score += 0.6
            
            # Check for proper decorators and functions
            content = result.step_definitions
            if "@given" in content or "@when" in content or "@then" in content:
                score += 0.2
            if "def " in content:
                score += 0.2
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_scenario_completeness(self, 
                                       results: List[BDDGenerationResult],
                                       tickets: List[Dict[str, Any]]) -> float:
        """Calculate how completely scenarios cover the requirements."""
        scores = []
        
        for result, ticket in zip(results, tickets):
            score = 0.0
            
            # Check scenario count appropriateness
            if result.scenarios_count > 0:
                score += 0.3
            if result.scenarios_count >= 2:  # At least positive and negative scenarios
                score += 0.2
            
            # Check coverage of acceptance criteria
            ac = ticket.get('acceptance_criteria', '').lower()
            feature_content_lower = result.feature_content.lower()
            
            if ac:
                # Look for key terms from acceptance criteria in feature
                ac_words = set(ac.split())
                feature_words = set(feature_content_lower.split())
                overlap = len(ac_words.intersection(feature_words)) / len(ac_words) if ac_words else 0
                score += min(0.5, overlap)
            else:
                score += 0.3  # Default if no AC provided
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_requirement_coverage(self, 
                                      results: List[BDDGenerationResult],
                                      tickets: List[Dict[str, Any]]) -> float:
        """Calculate how well the BDD covers the original requirements."""
        scores = []
        
        for result, ticket in zip(results, tickets):
            score = 0.0
            
            # Extract key concepts from ticket
            summary = ticket.get('summary', '').lower()
            description = ticket.get('description', '').lower()
            
            feature_content = result.feature_content.lower()
            
            # Check if main concepts are covered
            summary_words = set(summary.split())
            covered_summary = sum(1 for word in summary_words if word in feature_content)
            if summary_words:
                score += 0.4 * (covered_summary / len(summary_words))
            
            # Check for user story elements
            if any(pattern in feature_content for pattern in ["as a", "i want", "so that"]):
                score += 0.3
            
            # Check for action words
            action_words = ["click", "enter", "submit", "navigate", "login", "search"]
            if any(word in feature_content for word in action_words):
                score += 0.3
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_business_logic_accuracy(self, 
                                         results: List[BDDGenerationResult],
                                         tickets: List[Dict[str, Any]]) -> float:
        """Calculate accuracy of business logic representation."""
        scores = []
        
        for result, ticket in zip(results, tickets):
            score = 0.0
            
            feature_content = result.feature_content.lower()
            
            # Check for business rule indicators
            business_indicators = [
                "should", "must", "will", "can", "may",
                "valid", "invalid", "error", "success",
                "user", "customer", "admin", "system"
            ]
            
            present_indicators = sum(1 for indicator in business_indicators if indicator in feature_content)
            score += min(0.5, present_indicators / len(business_indicators) * 2)
            
            # Check for conditional logic
            conditional_words = ["if", "when", "then", "given", "and", "but"]
            present_conditionals = sum(1 for word in conditional_words if word in feature_content)
            if present_conditionals >= 3:
                score += 0.3
            
            # Check for validation scenarios
            if any(word in feature_content for word in ["error", "invalid", "fail"]):
                score += 0.2
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_technical_accuracy(self, results: List[BDDGenerationResult]) -> float:
        """Calculate technical accuracy of generated BDD."""
        scores = []
        
        for result in results:
            score = 0.0
            
            # Check feature file technical quality
            feature_lines = result.feature_content.split('\n')
            proper_indentation = sum(1 for line in feature_lines 
                                   if line.strip() and (line.startswith(' ') or line.startswith('Feature:')))
            if feature_lines:
                score += 0.3 * (proper_indentation / len(feature_lines))
            
            # Check step definition technical quality
            step_content = result.step_definitions
            if "import" in step_content and "pytest_bdd" in step_content:
                score += 0.4
            
            # Check for proper function definitions
            function_count = step_content.count("def ")
            decorator_count = step_content.count("@")
            if function_count > 0 and decorator_count >= function_count:
                score += 0.3
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_gherkin_syntax_score(self, results: List[BDDGenerationResult]) -> float:
        """Calculate Gherkin syntax correctness score."""
        scores = []
        
        for result in results:
            score = 0.0
            content = result.feature_content
            
            # Check for required Gherkin keywords
            required_keywords = self.reference_standards["gherkin_keywords"]
            present_keywords = sum(1 for keyword in required_keywords if keyword in content)
            score += 0.5 * (present_keywords / len(required_keywords))
            
            # Check syntax structure
            lines = content.split('\n')
            feature_count = sum(1 for line in lines if line.strip().startswith('Feature:'))
            scenario_count = sum(1 for line in lines if line.strip().startswith('Scenario:'))
            
            if feature_count == 1:  # Should have exactly one feature
                score += 0.2
            if scenario_count > 0:  # Should have at least one scenario
                score += 0.3
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_pytest_bdd_compatibility(self, results: List[BDDGenerationResult]) -> float:
        """Calculate pytest-bdd compatibility score."""
        scores = []
        
        for result in results:
            score = 0.0
            step_content = result.step_definitions
            
            # Check for pytest-bdd imports
            if "pytest_bdd" in step_content:
                score += 0.3
            
            # Check for proper decorators
            decorators = ["@given", "@when", "@then", "@scenarios"]
            present_decorators = sum(1 for decorator in decorators if decorator in step_content)
            score += 0.4 * (present_decorators / len(decorators))
            
            # Check for proper function structure
            if "def " in step_content:
                score += 0.3
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_confidence_score(self, 
                                  results: List[BDDGenerationResult],
                                  accuracy_components: List[float]) -> float:
        """Calculate confidence score based on consistency."""
        if len(results) < 2:
            return 0.8  # Default confidence for single result
        
        # Measure consistency across results
        quality_scores = [r.quality_score for r in results]
        generation_times = [r.generation_time for r in results]
        
        # Calculate coefficient of variation (lower is more consistent)
        quality_cv = statistics.stdev(quality_scores) / statistics.mean(quality_scores) if statistics.mean(quality_scores) > 0 else 1.0
        time_cv = statistics.stdev(generation_times) / statistics.mean(generation_times) if statistics.mean(generation_times) > 0 else 1.0
        
        # Calculate accuracy consistency
        accuracy_cv = statistics.stdev(accuracy_components) / statistics.mean(accuracy_components) if statistics.mean(accuracy_components) > 0 else 1.0
        
        # Convert to confidence score (lower CV = higher confidence)
        confidence = max(0.0, 1.0 - (quality_cv + time_cv + accuracy_cv) / 3)
        
        return min(1.0, confidence)
    
    def _generate_reference_bdd(self, ticket: Dict[str, Any]) -> str:
        """Generate a reference BDD scenario for comparison."""
        # Simplified reference generation
        key = ticket.get('key', 'UNKNOWN')
        summary = ticket.get('summary', 'Unknown feature')
        
        reference = f"""
Feature: {summary}
    As a user
    I want to use {summary.lower()}
    So that I can achieve my goal

    Scenario: Basic {summary.lower()} functionality
        Given the system is ready
        When I use the {summary.lower()} feature
        Then the system should respond correctly
        
    Scenario: Error handling for {summary.lower()}
        Given invalid conditions exist
        When I attempt to use {summary.lower()}
        Then an appropriate error should be shown
"""
        return reference
    
    def _create_failed_accuracy_metrics(self, model_name: str, reason: str) -> AccuracyMetrics:
        """Create failed accuracy metrics."""
        return AccuracyMetrics(
            model_name=model_name,
            test_suite="failed_validation",
            feature_structure_accuracy=0.0,
            step_definition_accuracy=0.0,
            scenario_completeness=0.0,
            requirement_coverage=0.0,
            business_logic_accuracy=0.0,
            technical_accuracy=0.0,
            gherkin_syntax_score=0.0,
            pytest_bdd_compatibility=0.0,
            overall_accuracy=0.0,
            confidence_score=0.0
        )

class PerformanceOptimizer:
    """Optimizes execution time and performance of BDD generation."""
    
    def __init__(self):
        """Initialize the performance optimizer."""
        self.model_manager = ModelManager()
        self.optimization_cache = {}
        self.profiling_results = {}
        
    def optimize_model_execution(self, 
                                model_name: str,
                                test_cases: List[Dict[str, Any]]) -> OptimizationResult:
        """Optimize execution performance for a specific model."""
        logger.info(f"Optimizing execution for model: {model_name}")
        
        # Baseline measurement
        baseline_time, baseline_memory = self._measure_baseline_performance(model_name, test_cases)
        
        # Apply optimization techniques
        optimization_techniques = []
        optimized_times = []
        optimized_memories = []
        
        # 1. Model pre-loading optimization
        preload_time, preload_memory = self._optimize_model_preloading(model_name, test_cases)
        optimization_techniques.append("model_preloading")
        optimized_times.append(preload_time)
        optimized_memories.append(preload_memory)
        
        # 2. Batch processing optimization
        batch_time, batch_memory = self._optimize_batch_processing(model_name, test_cases)
        optimization_techniques.append("batch_processing")
        optimized_times.append(batch_time)
        optimized_memories.append(batch_memory)
        
        # 3. Memory management optimization
        memory_time, memory_memory = self._optimize_memory_management(model_name, test_cases)
        optimization_techniques.append("memory_management")
        optimized_times.append(memory_time)
        optimized_memories.append(memory_memory)
        
        # 4. Caching optimization
        cache_time, cache_memory = self._optimize_caching(model_name, test_cases)
        optimization_techniques.append("caching")
        optimized_times.append(cache_time)
        optimized_memories.append(cache_memory)
        
        # Select best optimization
        best_idx = np.argmin(optimized_times)
        best_time = optimized_times[best_idx]
        best_memory = optimized_memories[best_idx]
        best_techniques = optimization_techniques[:best_idx + 1]
        
        # Calculate improvement
        improvement_ratio = baseline_time / best_time if best_time > 0 else 1.0
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            baseline_time, best_time, baseline_memory, best_memory, best_techniques
        )
        
        return OptimizationResult(
            original_time=baseline_time,
            optimized_time=best_time,
            improvement_ratio=improvement_ratio,
            memory_before=baseline_memory,
            memory_after=best_memory,
            optimization_techniques=best_techniques,
            recommendations=recommendations
        )
    
    def _measure_baseline_performance(self, 
                                    model_name: str,
                                    test_cases: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Measure baseline performance without optimizations."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        for test_case in test_cases:
            # Load model fresh each time (worst case)
            self.model_manager.unload_model(model_name)
            self.model_manager.generate_bdd_with_model(test_case, model_name)
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - start_time
        peak_memory = final_memory - initial_memory
        
        return total_time, peak_memory
    
    def _optimize_model_preloading(self, 
                                 model_name: str,
                                 test_cases: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Optimize by pre-loading model once."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Pre-load model once
        self.model_manager.load_model(model_name)
        
        for test_case in test_cases:
            self.model_manager.generate_bdd_with_model(test_case, model_name)
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - start_time
        peak_memory = final_memory - initial_memory
        
        return total_time, peak_memory
    
    def _optimize_batch_processing(self, 
                                 model_name: str,
                                 test_cases: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Optimize using batch processing."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Pre-load model
        self.model_manager.load_model(model_name)
        
        # Process in batches to optimize model usage
        batch_size = min(3, len(test_cases))
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            
            # Process batch
            for test_case in batch:
                self.model_manager.generate_bdd_with_model(test_case, model_name)
            
            # Optional: Clear intermediate caches between batches
            gc.collect()
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - start_time
        peak_memory = final_memory - initial_memory
        
        return total_time, peak_memory
    
    def _optimize_memory_management(self, 
                                  model_name: str,
                                  test_cases: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Optimize memory management."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Pre-load model
        self.model_manager.load_model(model_name)
        
        for i, test_case in enumerate(test_cases):
            self.model_manager.generate_bdd_with_model(test_case, model_name)
            
            # Aggressive garbage collection every few iterations
            if i % 2 == 0:
                gc.collect()
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - start_time
        peak_memory = final_memory - initial_memory
        
        return total_time, peak_memory
    
    def _optimize_caching(self, 
                        model_name: str,
                        test_cases: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Optimize using caching strategies."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Pre-load model
        self.model_manager.load_model(model_name)
        
        # Use a simple cache for similar tickets
        cache = {}
        
        for test_case in test_cases:
            # Create cache key based on summary
            cache_key = test_case.get('summary', '')[:50]  # First 50 chars
            
            if cache_key in cache:
                # Simulate cache hit (much faster)
                time.sleep(0.01)  # Minimal processing time
                continue
            
            # Cache miss - generate and cache
            result = self.model_manager.generate_bdd_with_model(test_case, model_name)
            cache[cache_key] = result
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - start_time
        peak_memory = final_memory - initial_memory
        
        return total_time, peak_memory
    
    def _generate_optimization_recommendations(self, 
                                             baseline_time: float,
                                             optimized_time: float,
                                             baseline_memory: float,
                                             optimized_memory: float,
                                             techniques: List[str]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        time_improvement = (baseline_time - optimized_time) / baseline_time if baseline_time > 0 else 0
        memory_improvement = (baseline_memory - optimized_memory) / baseline_memory if baseline_memory > 0 else 0
        
        if time_improvement > 0.2:
            recommendations.append(f"✅ Significant speed improvement achieved: {time_improvement:.1%}")
        elif time_improvement > 0:
            recommendations.append(f"📈 Moderate speed improvement: {time_improvement:.1%}")
        else:
            recommendations.append("⚠️ No significant speed improvement detected")
        
        if memory_improvement > 0.1:
            recommendations.append(f"✅ Memory usage reduced by {memory_improvement:.1%}")
        elif memory_improvement < -0.1:
            recommendations.append(f"⚠️ Memory usage increased by {abs(memory_improvement):.1%}")
        
        # Technique-specific recommendations
        if "model_preloading" in techniques:
            recommendations.append("💡 Keep models loaded between requests to avoid reload overhead")
        
        if "batch_processing" in techniques:
            recommendations.append("💡 Process multiple requests in batches for better efficiency")
        
        if "memory_management" in techniques:
            recommendations.append("💡 Implement regular garbage collection for long-running processes")
        
        if "caching" in techniques:
            recommendations.append("💡 Implement caching for similar requests to reduce computation")
        
        # Performance-based recommendations
        if optimized_time > 5.0:
            recommendations.append("⏱️ Consider using faster models for real-time applications")
        
        if optimized_memory > 1000:
            recommendations.append("🔋 Consider memory-efficient models for resource-constrained environments")
        
        return recommendations
    
    def profile_model_execution(self, 
                              model_name: str,
                              test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Profile detailed execution of model for performance bottlenecks."""
        logger.info(f"Profiling execution for model: {model_name}")
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile the execution
        profiler.enable()
        start_time = time.time()
        
        try:
            result = self.model_manager.generate_bdd_with_model(test_case, model_name)
            success = True
        except Exception as e:
            logger.error(f"Profiling failed: {str(e)}")
            result = None
            success = False
        
        end_time = time.time()
        profiler.disable()
        
        # Analyze profiling results
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profiling_output = stats_stream.getvalue()
        
        # Extract key metrics
        total_time = end_time - start_time
        
        # Parse profiling data for insights
        insights = self._analyze_profiling_output(profiling_output)
        
        profile_result = {
            "model_name": model_name,
            "test_case": test_case.get('key', 'unknown'),
            "success": success,
            "total_time": total_time,
            "profiling_output": profiling_output,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store for later analysis
        self.profiling_results[f"{model_name}_{test_case.get('key', 'unknown')}"] = profile_result
        
        return profile_result
    
    def _analyze_profiling_output(self, profiling_output: str) -> List[str]:
        """Analyze profiling output to extract performance insights."""
        insights = []
        
        lines = profiling_output.split('\n')
        
        # Look for time-consuming functions
        for line in lines:
            if 'cumtime' in line or 'tottime' in line:
                continue
            
            parts = line.split()
            if len(parts) >= 6:
                try:
                    cumtime = float(parts[3])
                    if cumtime > 0.1:  # Functions taking more than 100ms
                        function_name = parts[-1] if parts else "unknown"
                        insights.append(f"High CPU function: {function_name} ({cumtime:.2f}s)")
                except (ValueError, IndexError):
                    continue
        
        # Add general insights
        if "torch" in profiling_output.lower():
            insights.append("PyTorch operations detected - consider GPU acceleration")
        
        if "transformers" in profiling_output.lower():
            insights.append("Transformer model operations - tokenization and inference are main bottlenecks")
        
        if "numpy" in profiling_output.lower():
            insights.append("NumPy operations detected - consider optimized BLAS libraries")
        
        return insights[:10]  # Limit to top 10 insights

class ValidationOptimizationFramework:
    """Combined framework for validation and optimization."""
    
    def __init__(self):
        """Initialize the combined framework."""
        self.accuracy_validator = AccuracyValidator()
        self.performance_optimizer = PerformanceOptimizer()
        self.benchmark_framework = ModelBenchmarkFramework()
        self.data_generator = TestDataGenerator()
        
    def run_comprehensive_validation(self, 
                                   models_to_test: List[str],
                                   num_test_cases: int = 5) -> Dict[str, Any]:
        """Run comprehensive validation and optimization for multiple models."""
        logger.info(f"Running comprehensive validation for {len(models_to_test)} models")
        
        # Generate test cases
        test_tickets = []
        for i in range(num_test_cases):
            complexity = ["simple", "medium", "complex"][i % 3]
            ticket = self.data_generator.generate_jira_ticket(complexity=complexity)
            test_tickets.append({
                'key': ticket.key,
                'summary': ticket.summary,
                'description': ticket.description,
                'acceptance_criteria': ticket.acceptance_criteria
            })
        
        results = {
            "test_summary": {
                "models_tested": len(models_to_test),
                "test_cases": len(test_tickets),
                "timestamp": datetime.now().isoformat()
            },
            "accuracy_results": {},
            "optimization_results": {},
            "benchmark_results": {},
            "recommendations": []
        }
        
        for model_name in models_to_test:
            logger.info(f"Processing model: {model_name}")
            
            try:
                # 1. Accuracy validation
                accuracy_metrics = self.accuracy_validator.validate_bdd_accuracy(
                    model_name, test_tickets
                )
                results["accuracy_results"][model_name] = accuracy_metrics
                
                # 2. Performance optimization
                optimization_result = self.performance_optimizer.optimize_model_execution(
                    model_name, test_tickets
                )
                results["optimization_results"][model_name] = optimization_result
                
                # 3. Benchmark testing
                benchmark_suite = self.benchmark_framework.create_benchmark_suite(
                    f"validation_suite_{model_name}", num_test_cases=2
                )
                benchmark_results = self.benchmark_framework.run_benchmark_suite(benchmark_suite)
                model_benchmarks = [r for r in benchmark_results if r.model_name == model_name]
                results["benchmark_results"][model_name] = model_benchmarks
                
            except Exception as e:
                logger.error(f"Validation failed for model {model_name}: {str(e)}")
                results["accuracy_results"][model_name] = "FAILED"
                results["optimization_results"][model_name] = "FAILED"
                results["benchmark_results"][model_name] = "FAILED"
        
        # Generate overall recommendations
        results["recommendations"] = self._generate_overall_recommendations(results)
        
        return results
    
    def _generate_overall_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on validation results."""
        recommendations = []
        
        accuracy_results = results.get("accuracy_results", {})
        optimization_results = results.get("optimization_results", {})
        
        # Accuracy-based recommendations
        best_accuracy = 0.0
        best_accuracy_model = None
        
        for model_name, metrics in accuracy_results.items():
            if isinstance(metrics, AccuracyMetrics) and metrics.overall_accuracy > best_accuracy:
                best_accuracy = metrics.overall_accuracy
                best_accuracy_model = model_name
        
        if best_accuracy_model:
            recommendations.append(f"🎯 Highest accuracy model: {best_accuracy_model} ({best_accuracy:.2f})")
        
        # Performance-based recommendations
        best_improvement = 0.0
        best_performance_model = None
        
        for model_name, opt_result in optimization_results.items():
            if isinstance(opt_result, OptimizationResult) and opt_result.improvement_ratio > best_improvement:
                best_improvement = opt_result.improvement_ratio
                best_performance_model = model_name
        
        if best_performance_model:
            recommendations.append(f"⚡ Best optimization potential: {best_performance_model} ({best_improvement:.1f}x improvement)")
        
        # Quality thresholds
        high_quality_models = []
        for model_name, metrics in accuracy_results.items():
            if isinstance(metrics, AccuracyMetrics) and metrics.overall_accuracy >= 0.8:
                high_quality_models.append(model_name)
        
        if high_quality_models:
            recommendations.append(f"✅ High quality models (≥80%): {', '.join(high_quality_models)}")
        
        # Performance warnings
        slow_models = []
        for model_name, opt_result in optimization_results.items():
            if isinstance(opt_result, OptimizationResult) and opt_result.optimized_time > 10.0:
                slow_models.append(model_name)
        
        if slow_models:
            recommendations.append(f"⚠️ Slow models (>10s): {', '.join(slow_models)}")
        
        return recommendations
    
    def export_validation_report(self, 
                                validation_results: Dict[str, Any],
                                output_path: str):
        """Export comprehensive validation report."""
        report = {
            "metadata": {
                "report_type": "comprehensive_validation",
                "generated_at": datetime.now().isoformat(),
                "framework_version": "1.0"
            },
            "validation_results": validation_results,
            "summary": {
                "models_tested": validation_results["test_summary"]["models_tested"],
                "test_cases": validation_results["test_summary"]["test_cases"],
                "success_rate": self._calculate_success_rate(validation_results),
                "avg_accuracy": self._calculate_average_accuracy(validation_results),
                "avg_optimization": self._calculate_average_optimization(validation_results)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report exported to {output_path}")
    
    def _calculate_success_rate(self, results: Dict[str, Any]) -> float:
        """Calculate overall success rate."""
        total_tests = 0
        successful_tests = 0
        
        for model_results in [results.get("accuracy_results", {}), 
                            results.get("optimization_results", {}), 
                            results.get("benchmark_results", {})]:
            for model_name, result in model_results.items():
                total_tests += 1
                if result != "FAILED":
                    successful_tests += 1
        
        return successful_tests / total_tests if total_tests > 0 else 0.0
    
    def _calculate_average_accuracy(self, results: Dict[str, Any]) -> float:
        """Calculate average accuracy across all models."""
        accuracies = []
        
        for model_name, metrics in results.get("accuracy_results", {}).items():
            if isinstance(metrics, AccuracyMetrics):
                accuracies.append(metrics.overall_accuracy)
        
        return statistics.mean(accuracies) if accuracies else 0.0
    
    def _calculate_average_optimization(self, results: Dict[str, Any]) -> float:
        """Calculate average optimization improvement."""
        improvements = []
        
        for model_name, opt_result in results.get("optimization_results", {}).items():
            if isinstance(opt_result, OptimizationResult):
                improvements.append(opt_result.improvement_ratio)
        
        return statistics.mean(improvements) if improvements else 1.0
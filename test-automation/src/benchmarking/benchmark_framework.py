"""
Comprehensive benchmarking framework for model performance comparison and accuracy validation.
Provides metrics, validation, and optimization for BDD generation models.
"""

import time
import psutil
import asyncio
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import csv
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import memory_profiler
import pytest

from src.models.model_manager import ModelManager, BDDGenerationResult
from src.models.huggingface_research import HuggingFaceModelResearch, get_recommended_models_for_bdd
from src.utils.test_helpers import (
    TestDataGenerator, 
    PerformanceTimer,
    ValidationHelper,
    TestFileManager
)

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics for model evaluation."""
    model_name: str
    test_case: str
    
    # Performance metrics
    inference_time: float
    memory_usage_mb: float
    throughput_tokens_per_second: float
    
    # Quality metrics
    bdd_quality_score: float
    semantic_similarity_score: float
    structural_completeness_score: float
    
    # Accuracy metrics
    scenario_count_accuracy: float
    keyword_presence_score: float
    syntax_correctness_score: float
    
    # Resource metrics
    cpu_usage_percent: float
    peak_memory_mb: float
    disk_io_mb: float
    
    # Metadata
    timestamp: str
    model_size_mb: float
    test_complexity: str

@dataclass
class BenchmarkSuite:
    """Collection of benchmark tests and their configurations."""
    name: str
    description: str
    test_cases: List[Dict[str, Any]]
    models_to_test: List[str]
    metrics_to_collect: List[str]
    thresholds: Dict[str, float]
    
class ModelBenchmarkFramework:
    """Comprehensive benchmarking framework for model performance evaluation."""
    
    DEFAULT_THRESHOLDS = {
        "min_bdd_quality": 0.7,
        "max_inference_time": 10.0,
        "max_memory_usage": 2000.0,  # MB
        "min_similarity_score": 0.6,
        "min_completeness": 0.8,
        "min_syntax_score": 0.9
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the benchmark framework."""
        self.model_manager = ModelManager(cache_dir)
        self.research = HuggingFaceModelResearch()
        self.data_generator = TestDataGenerator()
        self.file_manager = TestFileManager()
        
        # Results storage
        self.benchmark_results: List[BenchmarkMetrics] = []
        self.comparison_reports: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.memory_tracker = memory_profiler.memory_usage
        
    def create_benchmark_suite(self, 
                             name: str, 
                             complexity_levels: List[str] = None,
                             num_test_cases: int = 5) -> BenchmarkSuite:
        """Create a comprehensive benchmark suite."""
        if complexity_levels is None:
            complexity_levels = ["simple", "medium", "complex"]
        
        test_cases = []
        
        for complexity in complexity_levels:
            for i in range(num_test_cases):
                ticket = self.data_generator.generate_jira_ticket(complexity=complexity)
                test_cases.append({
                    "name": f"{complexity}_case_{i+1}",
                    "complexity": complexity,
                    "ticket": {
                        "key": ticket.key,
                        "summary": ticket.summary,
                        "description": ticket.description,
                        "acceptance_criteria": ticket.acceptance_criteria
                    },
                    "expected_scenarios": {
                        "simple": 1,
                        "medium": 2,
                        "complex": 4
                    }[complexity]
                })
        
        models_to_test = get_recommended_models_for_bdd()
        
        return BenchmarkSuite(
            name=name,
            description=f"Comprehensive benchmark with {len(test_cases)} test cases across {len(complexity_levels)} complexity levels",
            test_cases=test_cases,
            models_to_test=models_to_test,
            metrics_to_collect=[
                "inference_time", "memory_usage", "bdd_quality", 
                "similarity_score", "completeness", "syntax_correctness"
            ],
            thresholds=self.DEFAULT_THRESHOLDS.copy()
        )
    
    def run_single_benchmark(self, 
                           model_name: str, 
                           test_case: Dict[str, Any]) -> BenchmarkMetrics:
        """Run a single benchmark test for a model on a test case."""
        logger.info(f"Running benchmark: {model_name} on {test_case['name']}")
        
        # Prepare monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start performance monitoring
        start_time = time.time()
        start_cpu = process.cpu_percent()
        
        try:
            # Run BDD generation
            with PerformanceTimer("generation") as timer:
                result = self.model_manager.generate_bdd_with_model(
                    test_case["ticket"], 
                    model_name
                )
            
            # Calculate metrics
            end_time = time.time()
            inference_time = timer.duration
            
            # Memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = peak_memory - initial_memory
            
            # CPU usage
            end_cpu = process.cpu_percent()
            cpu_usage = end_cpu - start_cpu
            
            # Quality metrics
            quality_scores = self._evaluate_quality_metrics(result, test_case)
            
            # Throughput calculation (approximate)
            estimated_tokens = len(result.feature_content.split()) + len(result.step_definitions.split())
            throughput = estimated_tokens / inference_time if inference_time > 0 else 0
            
            # Model size estimation
            model_info = self.model_manager.get_model_info(model_name)
            model_size = self._estimate_model_size(model_info)
            
            return BenchmarkMetrics(
                model_name=model_name,
                test_case=test_case["name"],
                inference_time=inference_time,
                memory_usage_mb=memory_usage,
                throughput_tokens_per_second=throughput,
                bdd_quality_score=quality_scores["bdd_quality"],
                semantic_similarity_score=quality_scores["similarity"],
                structural_completeness_score=quality_scores["completeness"],
                scenario_count_accuracy=quality_scores["scenario_accuracy"],
                keyword_presence_score=quality_scores["keyword_presence"],
                syntax_correctness_score=quality_scores["syntax_correctness"],
                cpu_usage_percent=cpu_usage,
                peak_memory_mb=peak_memory,
                disk_io_mb=0.0,  # Simplified for now
                timestamp=datetime.now().isoformat(),
                model_size_mb=model_size,
                test_complexity=test_case["complexity"]
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name} on {test_case['name']}: {str(e)}")
            return self._create_failed_benchmark(model_name, test_case, str(e))
    
    def _evaluate_quality_metrics(self, 
                                result: BDDGenerationResult, 
                                test_case: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate comprehensive quality metrics for generated BDD."""
        metrics = {}
        
        # BDD Quality Score (structural validity)
        feature_validation = ValidationHelper.validate_bdd_feature(result.feature_content)
        steps_validation = ValidationHelper.validate_step_definitions(result.step_definitions)
        
        bdd_quality = 0.0
        if feature_validation["is_valid"]:
            bdd_quality += 0.5
        if steps_validation["is_valid"]:
            bdd_quality += 0.5
        
        metrics["bdd_quality"] = bdd_quality
        
        # Semantic Similarity Score
        ticket_text = f"{test_case['ticket']['summary']} {test_case['ticket']['description']}"
        feature_text = result.feature_content
        
        try:
            similarity_scores = self.model_manager.compare_models_similarity(
                ticket_text, 
                feature_text, 
                ["sentence-transformer"]
            )
            metrics["similarity"] = similarity_scores.get("sentence-transformer", 0.0)
        except:
            metrics["similarity"] = 0.5  # Default moderate score
        
        # Structural Completeness
        completeness = 0.0
        required_elements = ["Feature:", "Scenario:", "Given", "When", "Then"]
        present_elements = sum(1 for elem in required_elements if elem in result.feature_content)
        completeness = present_elements / len(required_elements)
        metrics["completeness"] = completeness
        
        # Scenario Count Accuracy
        actual_scenarios = result.scenarios_count
        expected_scenarios = test_case.get("expected_scenarios", 2)
        
        if expected_scenarios > 0:
            scenario_accuracy = min(1.0, actual_scenarios / expected_scenarios)
        else:
            scenario_accuracy = 1.0 if actual_scenarios > 0 else 0.0
        
        metrics["scenario_accuracy"] = scenario_accuracy
        
        # Keyword Presence Score
        important_keywords = [
            test_case['ticket']['summary'].lower(),
            "user", "system", "should", "when", "given", "then"
        ]
        
        feature_lower = result.feature_content.lower()
        present_keywords = sum(1 for keyword in important_keywords if keyword in feature_lower)
        metrics["keyword_presence"] = present_keywords / len(important_keywords)
        
        # Syntax Correctness Score
        syntax_score = 1.0
        
        # Check for common syntax errors
        syntax_errors = [
            "undefined" in result.feature_content.lower(),
            "error" in result.feature_content.lower(),
            "failed" in result.feature_content.lower(),
            result.feature_content.count("Given") == 0,
            result.feature_content.count("When") == 0,
            result.feature_content.count("Then") == 0
        ]
        
        error_count = sum(syntax_errors)
        syntax_score = max(0.0, 1.0 - (error_count * 0.2))
        metrics["syntax_correctness"] = syntax_score
        
        return metrics
    
    def _estimate_model_size(self, model_info: Dict[str, Any]) -> float:
        """Estimate model size in MB."""
        # Simplified estimation based on model type and name
        model_name = model_info.get("name", "").lower()
        
        size_estimates = {
            "distilbert": 250,
            "bert-base": 400,
            "roberta-base": 500,
            "codebert": 500,
            "t5-small": 250,
            "t5-base": 850,
            "sentence-transformer": 200,
            "graphcodebert": 800,
            "unixcoder": 800,
            "codet5": 1000,
            "incoder": 4000,
            "code_llama": 15000
        }
        
        for key, size in size_estimates.items():
            if key in model_name:
                return size
        
        return 500  # Default estimate
    
    def _create_failed_benchmark(self, 
                                model_name: str, 
                                test_case: Dict[str, Any], 
                                error: str) -> BenchmarkMetrics:
        """Create a benchmark result for failed tests."""
        return BenchmarkMetrics(
            model_name=model_name,
            test_case=test_case["name"],
            inference_time=float('inf'),
            memory_usage_mb=float('inf'),
            throughput_tokens_per_second=0.0,
            bdd_quality_score=0.0,
            semantic_similarity_score=0.0,
            structural_completeness_score=0.0,
            scenario_count_accuracy=0.0,
            keyword_presence_score=0.0,
            syntax_correctness_score=0.0,
            cpu_usage_percent=0.0,
            peak_memory_mb=0.0,
            disk_io_mb=0.0,
            timestamp=datetime.now().isoformat(),
            model_size_mb=0.0,
            test_complexity=test_case["complexity"]
        )
    
    def run_benchmark_suite(self, 
                          benchmark_suite: BenchmarkSuite,
                          parallel: bool = False) -> List[BenchmarkMetrics]:
        """Run a complete benchmark suite."""
        logger.info(f"Running benchmark suite: {benchmark_suite.name}")
        logger.info(f"Test cases: {len(benchmark_suite.test_cases)}")
        logger.info(f"Models: {benchmark_suite.models_to_test}")
        
        all_results = []
        
        if parallel:
            all_results = self._run_parallel_benchmarks(benchmark_suite)
        else:
            all_results = self._run_sequential_benchmarks(benchmark_suite)
        
        self.benchmark_results.extend(all_results)
        logger.info(f"Completed benchmark suite with {len(all_results)} results")
        
        return all_results
    
    def _run_sequential_benchmarks(self, benchmark_suite: BenchmarkSuite) -> List[BenchmarkMetrics]:
        """Run benchmarks sequentially."""
        results = []
        
        total_tests = len(benchmark_suite.models_to_test) * len(benchmark_suite.test_cases)
        current_test = 0
        
        for model_name in benchmark_suite.models_to_test:
            try:
                # Pre-load model to ensure fair timing
                self.model_manager.load_model(model_name)
                
                for test_case in benchmark_suite.test_cases:
                    current_test += 1
                    logger.info(f"Progress: {current_test}/{total_tests} - {model_name} on {test_case['name']}")
                    
                    result = self.run_single_benchmark(model_name, test_case)
                    results.append(result)
                    
                    # Validate against thresholds
                    self._validate_benchmark_result(result, benchmark_suite.thresholds)
                
                # Unload model to free memory
                self.model_manager.unload_model(model_name)
                
            except Exception as e:
                logger.error(f"Failed to benchmark model {model_name}: {str(e)}")
                # Create failed results for all test cases
                for test_case in benchmark_suite.test_cases:
                    failed_result = self._create_failed_benchmark(model_name, test_case, str(e))
                    results.append(failed_result)
        
        return results
    
    def _run_parallel_benchmarks(self, benchmark_suite: BenchmarkSuite) -> List[BenchmarkMetrics]:
        """Run benchmarks in parallel (limited concurrency to avoid resource conflicts)."""
        results = []
        
        # Limit concurrency to avoid memory issues
        max_workers = min(2, len(benchmark_suite.models_to_test))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create all benchmark tasks
            tasks = []
            for model_name in benchmark_suite.models_to_test:
                for test_case in benchmark_suite.test_cases:
                    task = executor.submit(self.run_single_benchmark, model_name, test_case)
                    tasks.append(task)
            
            # Collect results
            for i, task in enumerate(tasks):
                try:
                    result = task.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                    logger.info(f"Completed parallel benchmark {i+1}/{len(tasks)}")
                except Exception as e:
                    logger.error(f"Parallel benchmark {i+1} failed: {str(e)}")
        
        return results
    
    def _validate_benchmark_result(self, 
                                 result: BenchmarkMetrics, 
                                 thresholds: Dict[str, float]):
        """Validate benchmark result against thresholds."""
        validations = []
        
        if result.inference_time > thresholds.get("max_inference_time", float('inf')):
            validations.append(f"Inference time {result.inference_time:.2f}s exceeds threshold")
        
        if result.memory_usage_mb > thresholds.get("max_memory_usage", float('inf')):
            validations.append(f"Memory usage {result.memory_usage_mb:.1f}MB exceeds threshold")
        
        if result.bdd_quality_score < thresholds.get("min_bdd_quality", 0.0):
            validations.append(f"BDD quality {result.bdd_quality_score:.2f} below threshold")
        
        if validations:
            logger.warning(f"Threshold violations for {result.model_name}: {', '.join(validations)}")
    
    def analyze_benchmark_results(self, 
                                results: List[BenchmarkMetrics] = None) -> Dict[str, Any]:
        """Analyze benchmark results and generate comprehensive insights."""
        if results is None:
            results = self.benchmark_results
        
        if not results:
            return {"error": "No benchmark results available"}
        
        analysis = {
            "summary": self._generate_summary_statistics(results),
            "model_rankings": self._rank_models(results),
            "performance_analysis": self._analyze_performance_trends(results),
            "quality_analysis": self._analyze_quality_metrics(results),
            "recommendations": self._generate_recommendations(results),
            "detailed_results": [asdict(r) for r in results]
        }
        
        return analysis
    
    def _generate_summary_statistics(self, results: List[BenchmarkMetrics]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        if not results:
            return {}
        
        # Group by model
        model_results = {}
        for result in results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)
        
        summary = {
            "total_benchmarks": len(results),
            "models_tested": len(model_results),
            "test_cases": len(set(r.test_case for r in results)),
            "avg_inference_time": statistics.mean([r.inference_time for r in results if r.inference_time != float('inf')]),
            "avg_memory_usage": statistics.mean([r.memory_usage_mb for r in results if r.memory_usage_mb != float('inf')]),
            "avg_bdd_quality": statistics.mean([r.bdd_quality_score for r in results]),
            "success_rate": len([r for r in results if r.inference_time != float('inf')]) / len(results)
        }
        
        # Per-model statistics
        summary["per_model"] = {}
        for model_name, model_res in model_results.items():
            valid_results = [r for r in model_res if r.inference_time != float('inf')]
            
            if valid_results:
                summary["per_model"][model_name] = {
                    "tests_run": len(model_res),
                    "success_rate": len(valid_results) / len(model_res),
                    "avg_inference_time": statistics.mean([r.inference_time for r in valid_results]),
                    "avg_memory_usage": statistics.mean([r.memory_usage_mb for r in valid_results]),
                    "avg_quality_score": statistics.mean([r.bdd_quality_score for r in valid_results]),
                    "avg_similarity_score": statistics.mean([r.semantic_similarity_score for r in valid_results])
                }
            else:
                summary["per_model"][model_name] = {
                    "tests_run": len(model_res),
                    "success_rate": 0.0,
                    "avg_inference_time": float('inf'),
                    "avg_memory_usage": float('inf'),
                    "avg_quality_score": 0.0,
                    "avg_similarity_score": 0.0
                }
        
        return summary
    
    def _rank_models(self, results: List[BenchmarkMetrics]) -> Dict[str, Any]:
        """Rank models based on different criteria."""
        # Group by model and calculate averages
        model_stats = {}
        for result in results:
            if result.model_name not in model_stats:
                model_stats[result.model_name] = []
            
            if result.inference_time != float('inf'):  # Only include successful runs
                model_stats[result.model_name].append(result)
        
        rankings = {
            "overall_score": [],
            "speed": [],
            "quality": [],
            "memory_efficiency": [],
            "reliability": []
        }
        
        for model_name, model_results in model_stats.items():
            if not model_results:
                continue
                
            # Calculate average metrics
            avg_inference = statistics.mean([r.inference_time for r in model_results])
            avg_memory = statistics.mean([r.memory_usage_mb for r in model_results])
            avg_quality = statistics.mean([r.bdd_quality_score for r in model_results])
            avg_similarity = statistics.mean([r.semantic_similarity_score for r in model_results])
            success_rate = len(model_results) / len([r for r in results if r.model_name == model_name])
            
            # Calculate composite scores
            speed_score = max(0, 1 - (avg_inference / 10.0))  # Normalize to 0-1
            memory_score = max(0, 1 - (avg_memory / 2000.0))  # Normalize to 0-1
            quality_score = (avg_quality + avg_similarity) / 2
            reliability_score = success_rate
            
            overall_score = (speed_score + memory_score + quality_score + reliability_score) / 4
            
            rankings["overall_score"].append((model_name, overall_score))
            rankings["speed"].append((model_name, speed_score))
            rankings["quality"].append((model_name, quality_score))
            rankings["memory_efficiency"].append((model_name, memory_score))
            rankings["reliability"].append((model_name, reliability_score))
        
        # Sort rankings
        for category in rankings:
            rankings[category].sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _analyze_performance_trends(self, results: List[BenchmarkMetrics]) -> Dict[str, Any]:
        """Analyze performance trends across different dimensions."""
        trends = {}
        
        # Performance by complexity
        complexity_groups = {"simple": [], "medium": [], "complex": []}
        for result in results:
            if result.test_complexity in complexity_groups:
                complexity_groups[result.test_complexity].append(result)
        
        trends["by_complexity"] = {}
        for complexity, group_results in complexity_groups.items():
            if group_results:
                valid_results = [r for r in group_results if r.inference_time != float('inf')]
                if valid_results:
                    trends["by_complexity"][complexity] = {
                        "avg_inference_time": statistics.mean([r.inference_time for r in valid_results]),
                        "avg_memory_usage": statistics.mean([r.memory_usage_mb for r in valid_results]),
                        "avg_quality_score": statistics.mean([r.bdd_quality_score for r in valid_results])
                    }
        
        # Model size vs performance correlation
        size_performance = []
        for result in results:
            if result.inference_time != float('inf') and result.model_size_mb > 0:
                size_performance.append((result.model_size_mb, result.inference_time))
        
        if size_performance:
            sizes, times = zip(*size_performance)
            correlation = np.corrcoef(sizes, times)[0, 1] if len(sizes) > 1 else 0
            trends["size_vs_performance"] = {
                "correlation": float(correlation),
                "interpretation": "positive" if correlation > 0.3 else "negative" if correlation < -0.3 else "weak"
            }
        
        return trends
    
    def _analyze_quality_metrics(self, results: List[BenchmarkMetrics]) -> Dict[str, Any]:
        """Analyze quality metrics in detail."""
        quality_analysis = {}
        
        # Quality distribution
        quality_scores = [r.bdd_quality_score for r in results]
        similarity_scores = [r.semantic_similarity_score for r in results]
        completeness_scores = [r.structural_completeness_score for r in results]
        
        quality_analysis["distributions"] = {
            "bdd_quality": {
                "mean": statistics.mean(quality_scores),
                "median": statistics.median(quality_scores),
                "std": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                "min": min(quality_scores),
                "max": max(quality_scores)
            },
            "semantic_similarity": {
                "mean": statistics.mean(similarity_scores),
                "median": statistics.median(similarity_scores),
                "std": statistics.stdev(similarity_scores) if len(similarity_scores) > 1 else 0,
                "min": min(similarity_scores),
                "max": max(similarity_scores)
            },
            "structural_completeness": {
                "mean": statistics.mean(completeness_scores),
                "median": statistics.median(completeness_scores),
                "std": statistics.stdev(completeness_scores) if len(completeness_scores) > 1 else 0,
                "min": min(completeness_scores),
                "max": max(completeness_scores)
            }
        }
        
        # Quality by model
        model_quality = {}
        for result in results:
            if result.model_name not in model_quality:
                model_quality[result.model_name] = {
                    "bdd_quality": [],
                    "similarity": [],
                    "completeness": []
                }
            
            model_quality[result.model_name]["bdd_quality"].append(result.bdd_quality_score)
            model_quality[result.model_name]["similarity"].append(result.semantic_similarity_score)
            model_quality[result.model_name]["completeness"].append(result.structural_completeness_score)
        
        quality_analysis["by_model"] = {}
        for model_name, metrics in model_quality.items():
            quality_analysis["by_model"][model_name] = {
                "avg_bdd_quality": statistics.mean(metrics["bdd_quality"]),
                "avg_similarity": statistics.mean(metrics["similarity"]),
                "avg_completeness": statistics.mean(metrics["completeness"]),
                "consistency": {
                    "bdd_quality_std": statistics.stdev(metrics["bdd_quality"]) if len(metrics["bdd_quality"]) > 1 else 0,
                    "similarity_std": statistics.stdev(metrics["similarity"]) if len(metrics["similarity"]) > 1 else 0
                }
            }
        
        return quality_analysis
    
    def _generate_recommendations(self, results: List[BenchmarkMetrics]) -> List[str]:
        """Generate actionable recommendations based on benchmark results."""
        recommendations = []
        
        if not results:
            return ["No benchmark results available for analysis"]
        
        # Find best performers
        valid_results = [r for r in results if r.inference_time != float('inf')]
        if not valid_results:
            recommendations.append("⚠️ All benchmark tests failed - check model configurations and dependencies")
            return recommendations
        
        # Speed recommendations
        fastest_result = min(valid_results, key=lambda x: x.inference_time)
        recommendations.append(f"🚀 Fastest model: {fastest_result.model_name} ({fastest_result.inference_time:.2f}s avg)")
        
        # Quality recommendations
        highest_quality = max(valid_results, key=lambda x: x.bdd_quality_score)
        recommendations.append(f"🏆 Highest quality: {highest_quality.model_name} (score: {highest_quality.bdd_quality_score:.2f})")
        
        # Memory efficiency
        most_efficient = min(valid_results, key=lambda x: x.memory_usage_mb)
        recommendations.append(f"💾 Most memory efficient: {most_efficient.model_name} ({most_efficient.memory_usage_mb:.1f}MB)")
        
        # Balanced recommendation
        balanced_scores = []
        for result in valid_results:
            speed_score = max(0, 1 - (result.inference_time / 10.0))
            memory_score = max(0, 1 - (result.memory_usage_mb / 2000.0))
            quality_score = result.bdd_quality_score
            balanced_score = (speed_score + memory_score + quality_score) / 3
            balanced_scores.append((result.model_name, balanced_score))
        
        if balanced_scores:
            best_balanced = max(balanced_scores, key=lambda x: x[1])
            recommendations.append(f"⚖️ Best balanced model: {best_balanced[0]} (composite score: {best_balanced[1]:.2f})")
        
        # Performance warnings
        slow_models = [r.model_name for r in valid_results if r.inference_time > 5.0]
        if slow_models:
            recommendations.append(f"⏱️ Consider avoiding for real-time use: {', '.join(set(slow_models))}")
        
        memory_heavy = [r.model_name for r in valid_results if r.memory_usage_mb > 1000]
        if memory_heavy:
            recommendations.append(f"🔋 High memory usage models: {', '.join(set(memory_heavy))}")
        
        # Quality warnings
        low_quality = [r.model_name for r in valid_results if r.bdd_quality_score < 0.6]
        if low_quality:
            recommendations.append(f"⚠️ Low quality output from: {', '.join(set(low_quality))}")
        
        # Success rate warnings
        model_success_rates = {}
        for result in results:
            if result.model_name not in model_success_rates:
                model_success_rates[result.model_name] = {"total": 0, "success": 0}
            model_success_rates[result.model_name]["total"] += 1
            if result.inference_time != float('inf'):
                model_success_rates[result.model_name]["success"] += 1
        
        unreliable_models = []
        for model_name, stats in model_success_rates.items():
            success_rate = stats["success"] / stats["total"]
            if success_rate < 0.8:
                unreliable_models.append(f"{model_name} ({success_rate:.1%})")
        
        if unreliable_models:
            recommendations.append(f"🔧 Models with reliability issues: {', '.join(unreliable_models)}")
        
        return recommendations
    
    def export_benchmark_results(self, 
                                filepath: str, 
                                format: str = "json",
                                include_analysis: bool = True):
        """Export benchmark results to various formats."""
        if not self.benchmark_results:
            logger.warning("No benchmark results to export")
            return
        
        filepath = Path(filepath)
        
        if format.lower() == "json":
            data = {
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "total_results": len(self.benchmark_results),
                    "format_version": "1.0"
                },
                "results": [asdict(r) for r in self.benchmark_results]
            }
            
            if include_analysis:
                data["analysis"] = self.analyze_benchmark_results()
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format.lower() == "csv":
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.benchmark_results[0]).keys())
                writer.writeheader()
                for result in self.benchmark_results:
                    writer.writerow(asdict(result))
        
        logger.info(f"Benchmark results exported to {filepath}")
    
    def generate_visualization_report(self, output_dir: str):
        """Generate visualization reports for benchmark results."""
        if not self.benchmark_results:
            logger.warning("No benchmark results to visualize")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance comparison chart
        self._plot_performance_comparison(output_path / "performance_comparison.png")
        
        # 2. Quality metrics heatmap
        self._plot_quality_heatmap(output_path / "quality_heatmap.png")
        
        # 3. Memory vs Speed scatter plot
        self._plot_memory_vs_speed(output_path / "memory_vs_speed.png")
        
        # 4. Model rankings radar chart
        self._plot_model_rankings_radar(output_path / "model_rankings.png")
        
        logger.info(f"Visualization reports generated in {output_path}")
    
    def _plot_performance_comparison(self, filepath: Path):
        """Plot performance comparison across models."""
        # Group results by model
        model_data = {}
        for result in self.benchmark_results:
            if result.inference_time != float('inf'):
                if result.model_name not in model_data:
                    model_data[result.model_name] = []
                model_data[result.model_name].append(result.inference_time)
        
        if not model_data:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(model_data.keys())
        times = [statistics.mean(model_data[model]) for model in models]
        
        bars = ax.bar(models, times)
        ax.set_ylabel('Average Inference Time (seconds)')
        ax.set_title('Model Performance Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_heatmap(self, filepath: Path):
        """Plot quality metrics heatmap."""
        # Prepare data for heatmap
        models = list(set(r.model_name for r in self.benchmark_results))
        metrics = ['BDD Quality', 'Similarity', 'Completeness', 'Syntax']
        
        data = []
        for model in models:
            model_results = [r for r in self.benchmark_results if r.model_name == model]
            if model_results:
                row = [
                    statistics.mean([r.bdd_quality_score for r in model_results]),
                    statistics.mean([r.semantic_similarity_score for r in model_results]),
                    statistics.mean([r.structural_completeness_score for r in model_results]),
                    statistics.mean([r.syntax_correctness_score for r in model_results])
                ]
                data.append(row)
        
        if not data:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metrics)):
                ax.text(j, i, f'{data[i][j]:.2f}', ha='center', va='center', color='black')
        
        ax.set_title('Quality Metrics Heatmap')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_vs_speed(self, filepath: Path):
        """Plot memory usage vs inference speed."""
        valid_results = [r for r in self.benchmark_results if r.inference_time != float('inf')]
        
        if not valid_results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by model for coloring
        models = list(set(r.model_name for r in valid_results))
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_results = [r for r in valid_results if r.model_name == model]
            x = [r.memory_usage_mb for r in model_results]
            y = [r.inference_time for r in model_results]
            ax.scatter(x, y, label=model, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Memory Usage (MB)')
        ax.set_ylabel('Inference Time (seconds)')
        ax.set_title('Memory Usage vs Inference Speed')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_rankings_radar(self, filepath: Path):
        """Plot model rankings as radar chart."""
        rankings = self._rank_models(self.benchmark_results)
        
        # Get top 5 models
        top_models = rankings["overall_score"][:5]
        if not top_models:
            return
        
        # Prepare data
        categories = ['Speed', 'Quality', 'Memory Efficiency', 'Reliability']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        for model_name, _ in top_models:
            values = []
            for category in ['speed', 'quality', 'memory_efficiency', 'reliability']:
                model_score = next((score for name, score in rankings[category] if name == model_name), 0)
                values.append(model_score)
            
            values += values[:1]  # Close the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Top 5 Models - Multi-Criteria Comparison')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def cleanup(self):
        """Clean up resources."""
        self.model_manager.unload_all_models()
        self.file_manager.cleanup()
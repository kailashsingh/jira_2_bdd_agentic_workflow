"""
Model Manager for BDD generation and validation testing.
Focuses on essential models: CodeBERT and Sentence Transformers for BDD scenario creation.
"""

import os
import time
import logging
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel,
    pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class BDDValidationResult:
    """Result of BDD validation against backend output."""
    backend_bdd_quality: float
    expected_bdd_quality: float
    similarity_score: float
    structure_match: bool
    scenario_count_match: bool
    validation_passed: bool
    differences: List[str]

@dataclass
class ModelPerformance:
    """Data class to track model performance metrics for BDD generation."""
    model_name: str
    inference_time: float
    memory_usage: float
    bdd_quality_score: float
    similarity_score: float
    generation_time: float

@dataclass
class BDDGenerationResult:
    """Result of BDD generation process."""
    feature_content: str
    step_definitions: str
    scenarios_count: int
    quality_score: float
    generation_time: float
    model_used: str

class ModelManager:
    """Manages essential models for BDD generation and validation testing."""
    
    # Streamlined model configuration focusing on BDD scenario creation and validation
    SUPPORTED_MODELS = {
        "codebert": {
            "name": "microsoft/codebert-base",
            "type": "encoder",
            "task": "feature-extraction",
            "description": "Primary model for understanding JIRA requirements and code context in BDD scenarios"
        },
        "sentence-transformer": {
            "name": "all-MiniLM-L6-v2",
            "type": "sentence-transformer", 
            "task": "sentence-similarity",
            "description": "Essential for validating BDD scenarios against expected results and similarity analysis"
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the model manager with focus on BDD validation."""
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.loaded_models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.performance_history: List[ModelPerformance] = []
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelManager initialized for BDD generation and validation")
        
    def load_model(self, model_key: str, force_reload: bool = False) -> Tuple[Any, Any]:
        """Load a model and its tokenizer for BDD processing."""
        if model_key not in self.SUPPORTED_MODELS:
            supported = list(self.SUPPORTED_MODELS.keys())
            raise ValueError(f"Unsupported model: {model_key}. Supported models: {supported}")
            
        if model_key in self.loaded_models and not force_reload:
            return self.loaded_models[model_key], self.tokenizers[model_key]
            
        model_config = self.SUPPORTED_MODELS[model_key]
        model_name = model_config["name"]
        model_type = model_config["type"]
        
        logger.info(f"Loading {model_type} model for BDD processing: {model_name}")
        
        try:
            # Load models specifically for BDD generation and validation
            if model_type == "sentence-transformer":
                # Sentence transformers handle tokenization internally
                model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
                tokenizer = None
            elif model_name == "microsoft/codebert-base":
                # CodeBERT for understanding code and requirements context
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                model = AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                logger.info("CodeBERT loaded successfully for BDD scenario understanding")
            else:
                # Standard transformer models
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                model = AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Store in cache
            self.loaded_models[model_key] = model
            self.tokenizers[model_key] = tokenizer
            
            logger.info(f"Successfully loaded {model_name} for BDD testing")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def get_model_embeddings(self, text: str, model_key: str) -> np.ndarray:
        """Get embeddings for text using specified model."""
        model, tokenizer = self.load_model(model_key)
        model_config = self.SUPPORTED_MODELS[model_key]
        
        if model_config["type"] == "sentence-transformer":
            # Sentence transformer
            embeddings = model.encode([text])
            return embeddings[0]
        else:
            # Standard transformer model
            if tokenizer is None:
                raise ValueError(f"Tokenizer not available for {model_key}")
                
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use last hidden state mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
            return embeddings
    
    def compare_models_similarity(self, text1: str, text2: str, model_keys: List[str]) -> Dict[str, float]:
        """Compare similarity scores across different models."""
        results = {}
        
        for model_key in model_keys:
            try:
                emb1 = self.get_model_embeddings(text1, model_key)
                emb2 = self.get_model_embeddings(text2, model_key)
                
                # Calculate cosine similarity
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                results[model_key] = float(similarity)
                
            except Exception as e:
                logger.error(f"Error computing similarity for {model_key}: {str(e)}")
                results[model_key] = 0.0
                
        return results
    
    def benchmark_model_performance(self, model_key: str, test_texts: List[str]) -> ModelPerformance:
        """Benchmark a model's performance on given test texts."""
        start_time = time.time()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            model, tokenizer = self.load_model(model_key)
            
            # Process all texts
            embeddings = []
            token_counts = []
            
            for text in test_texts:
                if self.SUPPORTED_MODELS[model_key]["type"] == "sentence-transformer":
                    emb = model.encode([text])[0]
                    token_counts.append(len(text.split()))  # Approximate
                else:
                    if tokenizer:
                        tokens = tokenizer.encode(text)
                        token_counts.append(len(tokens))
                    
                    emb = self.get_model_embeddings(text, model_key)
                
                embeddings.append(emb)
            
            # Calculate performance metrics
            inference_time = time.time() - start_time
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            # Calculate similarity between consecutive embeddings as quality metric
            quality_scores = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                quality_scores.append(sim)
            
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            
            performance = ModelPerformance(
                model_name=model_key,
                inference_time=inference_time,
                memory_usage=memory_usage,
                accuracy_score=0.0,  # To be set by external evaluation
                similarity_score=avg_quality,
                token_count=sum(token_counts),
                generation_quality=avg_quality
            )
            
            self.performance_history.append(performance)
            return performance
            
        except Exception as e:
            logger.error(f"Benchmarking failed for {model_key}: {str(e)}")
            return ModelPerformance(
                model_name=model_key,
                inference_time=float('inf'),
                memory_usage=float('inf'),
                accuracy_score=0.0,
                similarity_score=0.0,
                token_count=0,
                generation_quality=0.0
            )
    
    def generate_bdd_with_model(self, jira_ticket: Dict[str, Any], model_key: str) -> BDDGenerationResult:
        """Generate BDD scenarios using specified model."""
        start_time = time.time()
        
        try:
            model, tokenizer = self.load_model(model_key)
            model_config = self.SUPPORTED_MODELS[model_key]
            
            # Prepare input text
            input_text = f"""
            Ticket: {jira_ticket.get('key', 'UNKNOWN')}
            Summary: {jira_ticket.get('summary', '')}
            Description: {jira_ticket.get('description', '')}
            Acceptance Criteria: {jira_ticket.get('acceptance_criteria', '')}
            
            Generate BDD scenarios for this ticket.
            """
            
            if model_config["task"] == "text2text-generation" and "t5" in model_config["name"]:
                # T5 model generation
                prompt = f"Generate BDD scenarios: {input_text}"
                inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs, 
                        max_length=200, 
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            elif model_config["task"] == "text-generation":
                # Causal LM generation
                generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
                prompt = f"Generate BDD scenarios for: {input_text[:200]}..."
                
                results = generator(
                    prompt, 
                    max_length=300, 
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                generated_text = results[0]["generated_text"]
                
            else:
                # Feature extraction models - use for similarity/template matching
                embeddings = self.get_model_embeddings(input_text, model_key)
                
                # Simple template-based generation for feature extraction models
                generated_text = self._generate_template_bdd(jira_ticket)
            
            # Parse generated content
            feature_content, step_definitions = self._parse_generated_bdd(generated_text, jira_ticket)
            scenarios_count = feature_content.count("Scenario:")
            
            generation_time = time.time() - start_time
            quality_score = self._evaluate_bdd_quality(feature_content, jira_ticket)
            
            return BDDGenerationResult(
                feature_content=feature_content,
                step_definitions=step_definitions,
                scenarios_count=scenarios_count,
                quality_score=quality_score,
                generation_time=generation_time,
                model_used=model_key
            )
            
        except Exception as e:
            logger.error(f"BDD generation failed with {model_key}: {str(e)}")
            return BDDGenerationResult(
                feature_content="# Generation failed",
                step_definitions="# Generation failed", 
                scenarios_count=0,
                quality_score=0.0,
                generation_time=time.time() - start_time,
                model_used=model_key
            )
    
    def _generate_template_bdd(self, jira_ticket: Dict[str, Any]) -> str:
        """Generate BDD using templates for models that don't support generation."""
        ticket_key = jira_ticket.get('key', 'UNKNOWN')
        summary = jira_ticket.get('summary', 'Unknown feature')
        description = jira_ticket.get('description', '')
        
        template = f"""
Feature: {summary}
    As a user
    I want to {summary.lower()}
    So that I can achieve my goals

    Scenario: Basic {summary.lower()} functionality
        Given the system is ready
        When I perform the {summary.lower()} action
        Then the system should respond appropriately
        
    Scenario: Error handling for {summary.lower()}
        Given invalid input is provided
        When I attempt to {summary.lower()}
        Then an appropriate error message should be displayed
"""
        return template
    
    def _parse_generated_bdd(self, generated_text: str, jira_ticket: Dict[str, Any]) -> Tuple[str, str]:
        """Parse generated text into feature file and step definitions."""
        # Extract feature content
        feature_content = generated_text
        
        # Generate basic step definitions
        ticket_key = jira_ticket.get('key', 'UNKNOWN').lower().replace('-', '_')
        
        step_definitions = f"""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

# Load scenarios from feature file
scenarios('../features/{ticket_key}.feature')

@given('the system is ready')
def system_ready():
    # Implementation for system setup
    pass

@when(parsers.parse('I perform the {{action}} action'))
def perform_action(action):
    # Implementation for performing action
    pass

@then('the system should respond appropriately')
def verify_response():
    # Implementation for response verification
    pass

@given('invalid input is provided')
def invalid_input():
    # Implementation for invalid input setup
    pass

@when(parsers.parse('I attempt to {{action}}'))
def attempt_action(action):
    # Implementation for attempting action
    pass

@then('an appropriate error message should be displayed')
def verify_error_message():
    # Implementation for error verification
    pass
"""
        
        return feature_content, step_definitions
    
    def _evaluate_bdd_quality(self, feature_content: str, jira_ticket: Dict[str, Any]) -> float:
        """Evaluate the quality of generated BDD content."""
        quality_score = 0.0
        
        # Check for required BDD elements
        if "Feature:" in feature_content:
            quality_score += 0.2
        if "Scenario:" in feature_content:
            quality_score += 0.2
        if "Given" in feature_content:
            quality_score += 0.2
        if "When" in feature_content:
            quality_score += 0.2
        if "Then" in feature_content:
            quality_score += 0.2
        
        return quality_score
    
    def compare_models(self, jira_tickets: List[Dict[str, Any]], model_keys: List[str]) -> Dict[str, Any]:
        """Compare multiple models on the same set of JIRA tickets."""
        results = {
            "model_comparisons": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        for model_key in model_keys:
            model_results = []
            total_time = 0
            total_quality = 0
            
            for ticket in jira_tickets:
                result = self.generate_bdd_with_model(ticket, model_key)
                model_results.append(result)
                total_time += result.generation_time
                total_quality += result.quality_score
            
            avg_time = total_time / len(jira_tickets)
            avg_quality = total_quality / len(jira_tickets)
            
            results["model_comparisons"][model_key] = {
                "results": model_results,
                "avg_generation_time": avg_time,
                "avg_quality_score": avg_quality,
                "total_scenarios": sum(r.scenarios_count for r in model_results)
            }
            
            # Benchmark performance
            test_texts = [f"{t['summary']} {t['description']}" for t in jira_tickets]
            performance = self.benchmark_model_performance(model_key, test_texts)
            results["performance_metrics"][model_key] = performance
        
        # Generate recommendations
        results["recommendations"] = self._generate_model_recommendations(results)
        
        return results
    
    def _generate_model_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on model comparison results."""
        recommendations = []
        
        performance_data = comparison_results["performance_metrics"]
        comparison_data = comparison_results["model_comparisons"]
        
        # Find best performing models
        best_speed = min(performance_data.values(), key=lambda x: x.inference_time)
        best_quality = max(comparison_data.values(), key=lambda x: x["avg_quality_score"])
        best_memory = min(performance_data.values(), key=lambda x: x.memory_usage)
        
        recommendations.append(f"Fastest model: {best_speed.model_name} ({best_speed.inference_time:.2f}s)")
        recommendations.append(f"Highest quality: {best_quality} (score: {comparison_data[best_quality]['avg_quality_score']:.2f})")
        recommendations.append(f"Most memory efficient: {best_memory.model_name} ({best_memory.memory_usage:.2f}MB)")
        
        # Overall recommendation
        if best_speed.model_name == best_quality and best_speed.model_name == best_memory.model_name:
            recommendations.append(f"🏆 Overall best model: {best_speed.model_name}")
        else:
            recommendations.append("📊 Choose model based on priority: speed, quality, or memory efficiency")
        
        return recommendations
    
    def save_performance_report(self, filepath: str, comparison_results: Dict[str, Any]):
        """Save performance comparison report to file."""
        report = {
            "timestamp": time.time(),
            "models_tested": list(comparison_results["model_comparisons"].keys()),
            "results": comparison_results,
            "summary": {
                "total_models": len(comparison_results["model_comparisons"]),
                "best_recommendations": comparison_results["recommendations"]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {filepath}")
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        if model_key not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_key}")
        
        config = self.SUPPORTED_MODELS[model_key]
        
        # Check if model is loaded
        is_loaded = model_key in self.loaded_models
        
        # Get performance history for this model
        performance_history = [p for p in self.performance_history if p.model_name == model_key]
        
        return {
            "name": config["name"],
            "type": config["type"],
            "task": config["task"],
            "description": config["description"],
            "is_loaded": is_loaded,
            "performance_history": performance_history,
            "cache_dir": self.cache_dir
        }
    
    def unload_model(self, model_key: str):
        """Unload a model from memory."""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
        if model_key in self.tokenizers:
            del self.tokenizers[model_key]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info(f"Unloaded model: {model_key}")
    
    def unload_all_models(self):
        """Unload all models from memory."""
        self.loaded_models.clear()
        self.tokenizers.clear()
        
        import gc
        gc.collect()
        
        logger.info("Unloaded all models")
    
    # BDD Validation Methods - Core functionality for project scope
    
    def validate_bdd_against_backend_result(self, generated_bdd: Dict[str, str], 
                                          backend_result: Dict[str, Any],
                                          original_ticket: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate generated BDD scenarios against backend execution results.
        This is the core validation function for the project scope.
        """
        validation_result = {
            "overall_score": 0.0,
            "requirement_coverage": 0.0,
            "scenario_completeness": 0.0,
            "step_accuracy": 0.0,
            "backend_alignment": 0.0,
            "validation_details": {},
            "recommendations": []
        }
        
        try:
            # 1. Validate requirement coverage
            requirement_score = self._validate_requirement_coverage(
                generated_bdd, original_ticket
            )
            validation_result["requirement_coverage"] = requirement_score
            
            # 2. Validate scenario completeness
            completeness_score = self._validate_scenario_completeness(generated_bdd)
            validation_result["scenario_completeness"] = completeness_score
            
            # 3. Validate step accuracy
            step_score = self._validate_step_accuracy(generated_bdd)
            validation_result["step_accuracy"] = step_score
            
            # 4. Validate alignment with backend result
            backend_score = self._validate_backend_alignment(
                generated_bdd, backend_result
            )
            validation_result["backend_alignment"] = backend_score
            
            # Calculate overall score
            validation_result["overall_score"] = (
                requirement_score * 0.3 +
                completeness_score * 0.25 +
                step_score * 0.25 +
                backend_score * 0.2
            )
            
            # Generate recommendations
            validation_result["recommendations"] = self._generate_validation_recommendations(
                validation_result
            )
            
            logger.info(f"BDD validation completed with overall score: {validation_result['overall_score']:.2f}")
            
        except Exception as e:
            logger.error(f"BDD validation failed: {str(e)}")
            validation_result["error"] = str(e)
        
        return validation_result
    
    def _validate_requirement_coverage(self, generated_bdd: Dict[str, str], 
                                     original_ticket: Dict[str, str]) -> float:
        """Validate how well the BDD covers the original requirements."""
        if not self._is_model_loaded("sentence-transformer"):
            self.load_model("sentence-transformer")
        
        model = self.loaded_models["sentence-transformer"]
        
        # Extract requirement elements
        requirements = [
            original_ticket.get("summary", ""),
            original_ticket.get("description", ""),
            original_ticket.get("acceptance_criteria", "")
        ]
        requirement_text = " ".join(req for req in requirements if req)
        
        # Extract BDD content
        bdd_content = generated_bdd.get("feature", "") + " " + generated_bdd.get("step_definitions", "")
        
        if not requirement_text or not bdd_content:
            return 0.0
        
        # Calculate semantic similarity
        embeddings = model.encode([requirement_text, bdd_content])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return max(0.0, min(1.0, similarity))
    
    def _validate_scenario_completeness(self, generated_bdd: Dict[str, str]) -> float:
        """Validate completeness of BDD scenarios."""
        feature_content = generated_bdd.get("feature", "")
        step_definitions = generated_bdd.get("step_definitions", "")
        
        completeness_score = 0.0
        
        # Check for essential BDD elements
        checks = {
            "has_feature": "Feature:" in feature_content,
            "has_scenario": "Scenario:" in feature_content,
            "has_given": "Given" in feature_content,
            "has_when": "When" in feature_content,
            "has_then": "Then" in feature_content,
            "has_step_definitions": len(step_definitions) > 50,
            "has_imports": "import" in step_definitions.lower(),
            "has_test_framework": any(fw in step_definitions for fw in ["@wdio", "cucumber", "pytest"])
        }
        
        completeness_score = sum(checks.values()) / len(checks)
        
        return completeness_score
    
    def _validate_step_accuracy(self, generated_bdd: Dict[str, str]) -> float:
        """Validate accuracy of step definitions."""
        step_definitions = generated_bdd.get("step_definitions", "")
        
        if not step_definitions:
            return 0.0
        
        accuracy_score = 0.0
        
        # Check for step definition quality indicators
        quality_checks = {
            "has_step_decorators": any(decorator in step_definitions for decorator in ["Given(", "When(", "Then("]),
            "has_async_await": "await" in step_definitions,
            "has_assertions": any(assertion in step_definitions for assertion in ["expect", "assert", "should"]),
            "has_selectors": any(selector in step_definitions for selector in ["$", "browser", "page"]),
            "proper_syntax": step_definitions.count("(") == step_definitions.count(")"),
            "has_implementation": not ("// TODO" in step_definitions or "pass" in step_definitions)
        }
        
        accuracy_score = sum(quality_checks.values()) / len(quality_checks)
        
        return accuracy_score
    
    def _validate_backend_alignment(self, generated_bdd: Dict[str, str], 
                                  backend_result: Dict[str, Any]) -> float:
        """Validate alignment between generated BDD and backend execution result."""
        if not backend_result:
            return 0.5  # Neutral score if no backend result
        
        alignment_score = 0.0
        
        # Check backend result status
        backend_status = backend_result.get("status", "unknown")
        backend_success = backend_status in ["completed", "success"]
        
        # Extract BDD content
        feature_content = generated_bdd.get("feature", "")
        step_definitions = generated_bdd.get("step_definitions", "")
        
        alignment_checks = {
            "backend_successful": backend_success,
            "has_ticket_reference": backend_result.get("ticket_key", "") in feature_content,
            "appropriate_complexity": len(step_definitions) > 100 if backend_success else True,
            "matches_execution_pattern": self._check_execution_pattern_match(generated_bdd, backend_result)
        }
        
        alignment_score = sum(alignment_checks.values()) / len(alignment_checks)
        
        return alignment_score
    
    def _check_execution_pattern_match(self, generated_bdd: Dict[str, str], 
                                     backend_result: Dict[str, Any]) -> bool:
        """Check if BDD pattern matches backend execution pattern."""
        feature_content = generated_bdd.get("feature", "")
        
        # Check for workflow-related patterns
        workflow_patterns = ["navigate", "click", "form", "submit", "verify"]
        has_workflow_steps = any(pattern in feature_content.lower() for pattern in workflow_patterns)
        
        # Check if backend had navigation
        backend_had_navigation = "application_data" in str(backend_result) or "navigation" in str(backend_result)
        
        # Pattern should match: if backend had navigation, BDD should have workflow steps
        if backend_had_navigation:
            return has_workflow_steps
        else:
            return True  # No specific pattern required if no navigation
    
    def _generate_validation_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if validation_result["requirement_coverage"] < 0.7:
            recommendations.append("Improve requirement coverage - BDD scenarios don't fully address the original requirements")
        
        if validation_result["scenario_completeness"] < 0.8:
            recommendations.append("Add missing BDD elements - ensure all scenarios have Given/When/Then structure")
        
        if validation_result["step_accuracy"] < 0.7:
            recommendations.append("Improve step definition quality - add proper assertions and selectors")
        
        if validation_result["backend_alignment"] < 0.6:
            recommendations.append("Better align BDD with backend execution patterns")
        
        if validation_result["overall_score"] > 0.8:
            recommendations.append("Excellent BDD quality - ready for implementation")
        elif validation_result["overall_score"] > 0.6:
            recommendations.append("Good BDD quality - minor improvements recommended")
        else:
            recommendations.append("BDD quality needs significant improvement before implementation")
        
        return recommendations
    
    def compare_bdd_with_expected(self, generated_bdd: Dict[str, str], 
                                expected_bdd: Dict[str, str]) -> Dict[str, Any]:
        """Compare generated BDD with expected BDD scenarios."""
        if not self._is_model_loaded("sentence-transformer"):
            self.load_model("sentence-transformer")
        
        model = self.loaded_models["sentence-transformer"]
        
        comparison_result = {
            "feature_similarity": 0.0,
            "step_similarity": 0.0,
            "overall_similarity": 0.0,
            "differences": [],
            "improvements": []
        }
        
        try:
            # Compare feature content
            generated_feature = generated_bdd.get("feature", "")
            expected_feature = expected_bdd.get("feature", "")
            
            if generated_feature and expected_feature:
                feature_embeddings = model.encode([generated_feature, expected_feature])
                feature_similarity = cosine_similarity([feature_embeddings[0]], [feature_embeddings[1]])[0][0]
                comparison_result["feature_similarity"] = max(0.0, min(1.0, feature_similarity))
            
            # Compare step definitions
            generated_steps = generated_bdd.get("step_definitions", "")
            expected_steps = expected_bdd.get("step_definitions", "")
            
            if generated_steps and expected_steps:
                step_embeddings = model.encode([generated_steps, expected_steps])
                step_similarity = cosine_similarity([step_embeddings[0]], [step_embeddings[1]])[0][0]
                comparison_result["step_similarity"] = max(0.0, min(1.0, step_similarity))
            
            # Calculate overall similarity
            comparison_result["overall_similarity"] = (
                comparison_result["feature_similarity"] * 0.6 +
                comparison_result["step_similarity"] * 0.4
            )
            
            # Identify differences and improvements
            comparison_result["differences"] = self._identify_bdd_differences(generated_bdd, expected_bdd)
            comparison_result["improvements"] = self._suggest_bdd_improvements(comparison_result)
            
        except Exception as e:
            logger.error(f"BDD comparison failed: {str(e)}")
            comparison_result["error"] = str(e)
        
        return comparison_result
    
    def _identify_bdd_differences(self, generated_bdd: Dict[str, str], 
                                expected_bdd: Dict[str, str]) -> List[str]:
        """Identify key differences between generated and expected BDD."""
        differences = []
        
        generated_feature = generated_bdd.get("feature", "")
        expected_feature = expected_bdd.get("feature", "")
        
        # Check scenario count
        generated_scenarios = generated_feature.count("Scenario:")
        expected_scenarios = expected_feature.count("Scenario:")
        
        if generated_scenarios != expected_scenarios:
            differences.append(f"Scenario count mismatch: generated {generated_scenarios}, expected {expected_scenarios}")
        
        # Check for missing Given/When/Then
        for step_type in ["Given", "When", "Then"]:
            generated_count = generated_feature.count(step_type)
            expected_count = expected_feature.count(step_type)
            
            if generated_count < expected_count:
                differences.append(f"Missing {step_type} steps: generated {generated_count}, expected {expected_count}")
        
        return differences
    
    def _suggest_bdd_improvements(self, comparison_result: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on comparison results."""
        improvements = []
        
        if comparison_result["feature_similarity"] < 0.7:
            improvements.append("Improve feature description to better match expected structure")
        
        if comparison_result["step_similarity"] < 0.7:
            improvements.append("Align step definitions with expected implementation patterns")
        
        if comparison_result["overall_similarity"] < 0.8:
            improvements.append("Review and enhance overall BDD structure and content")
        
        return improvements

    def validate_backend_bdd_results(self, backend_result: Dict[str, Any], 
                                   expected_result: Dict[str, Any], 
                                   jira_ticket: Dict[str, Any]) -> BDDValidationResult:
        """
        Validate BDD results from backend against expected results.
        This is the core method for testing if our backend generates correct BDD scenarios.
        """
        start_time = time.time()
        
        try:
            # Extract BDD content from backend result
            backend_feature = backend_result.get("feature", "")
            backend_steps = backend_result.get("step_definitions", "")
            backend_ticket_key = backend_result.get("ticket_key", "")
            
            # Extract expected content
            expected_feature = expected_result.get("feature", "")
            expected_steps = expected_result.get("step_definitions", "")
            
            # Validate ticket key consistency
            if backend_ticket_key != jira_ticket.get("key", ""):
                logger.warning(f"Ticket key mismatch: backend={backend_ticket_key}, expected={jira_ticket.get('key')}")
            
            # Calculate similarity scores using sentence transformer
            feature_similarity = self._calculate_text_similarity(backend_feature, expected_feature)
            steps_similarity = self._calculate_text_similarity(backend_steps, expected_steps)
            overall_similarity = (feature_similarity + steps_similarity) / 2
            
            # Validate BDD structure
            backend_scenarios = backend_feature.count("Scenario:")
            expected_scenarios = expected_feature.count("Scenario:")
            scenario_count_match = backend_scenarios == expected_scenarios
            
            # Check BDD structure elements
            structure_checks = {
                "feature_keyword": "Feature:" in backend_feature,
                "scenario_keyword": "Scenario:" in backend_feature,
                "given_steps": "Given" in backend_feature,
                "when_steps": "When" in backend_feature,
                "then_steps": "Then" in backend_feature,
                "step_imports": "import" in backend_steps.lower(),
                "step_functions": any(keyword in backend_steps for keyword in ["Given", "When", "Then"])
            }
            
            structure_match = all(structure_checks.values())
            
            # Determine overall validation result
            validation_passed = (
                overall_similarity >= 0.7 and  # 70% similarity threshold
                structure_match and
                scenario_count_match
            )
            
            # Identify differences
            differences = []
            if not scenario_count_match:
                differences.append(f"Scenario count mismatch: backend={backend_scenarios}, expected={expected_scenarios}")
            
            if feature_similarity < 0.7:
                differences.append(f"Feature content similarity too low: {feature_similarity:.2f}")
            
            if steps_similarity < 0.7:
                differences.append(f"Step definitions similarity too low: {steps_similarity:.2f}")
            
            for check_name, check_result in structure_checks.items():
                if not check_result:
                    differences.append(f"Missing BDD structure element: {check_name}")
            
            # Calculate BDD quality scores
            backend_bdd_quality = self._evaluate_bdd_quality(backend_feature, jira_ticket)
            expected_bdd_quality = self._evaluate_bdd_quality(expected_feature, jira_ticket)
            
            validation_time = time.time() - start_time
            logger.info(f"BDD validation completed in {validation_time:.2f}s - Passed: {validation_passed}")
            
            return BDDValidationResult(
                backend_bdd_quality=backend_bdd_quality,
                expected_bdd_quality=expected_bdd_quality,
                similarity_score=overall_similarity,
                structure_match=structure_match,
                scenario_count_match=scenario_count_match,
                validation_passed=validation_passed,
                differences=differences
            )
            
        except Exception as e:
            logger.error(f"BDD validation failed: {str(e)}")
            return BDDValidationResult(
                backend_bdd_quality=0.0,
                expected_bdd_quality=0.0,
                similarity_score=0.0,
                structure_match=False,
                scenario_count_match=False,
                validation_passed=False,
                differences=[f"Validation error: {str(e)}"]
            )
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using sentence transformer."""
        try:
            model, _ = self.load_model("sentence-transformer")
            
            # Generate embeddings
            embeddings1 = model.encode([text1])
            embeddings2 = model.encode([text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0
    
    def run_bdd_validation_test_suite(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a comprehensive BDD validation test suite against backend results.
        This is the main entry point for testing our backend BDD generation.
        """
        results = {
            "test_timestamp": time.time(),
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "validation_results": [],
            "summary": {}
        }
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running BDD validation test {i+1}/{len(test_cases)}")
            
            try:
                validation_result = self.validate_backend_bdd_results(
                    test_case["backend_result"],
                    test_case["expected_result"], 
                    test_case["jira_ticket"]
                )
                
                test_result = {
                    "test_id": test_case.get("test_id", f"test_{i+1}"),
                    "jira_key": test_case["jira_ticket"].get("key", "UNKNOWN"),
                    "validation_passed": validation_result.validation_passed,
                    "similarity_score": validation_result.similarity_score,
                    "backend_quality": validation_result.backend_bdd_quality,
                    "expected_quality": validation_result.expected_bdd_quality,
                    "differences": validation_result.differences
                }
                
                results["validation_results"].append(test_result)
                
                if validation_result.validation_passed:
                    results["passed_tests"] += 1
                else:
                    results["failed_tests"] += 1
                    
            except Exception as e:
                logger.error(f"Test case {i+1} failed: {str(e)}")
                results["failed_tests"] += 1
                results["validation_results"].append({
                    "test_id": f"test_{i+1}",
                    "jira_key": "ERROR",
                    "validation_passed": False,
                    "error": str(e)
                })
        
        # Calculate summary statistics
        total_tests = results["total_tests"]
        passed_tests = results["passed_tests"]
        
        results["summary"] = {
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": results["failed_tests"],
            "avg_similarity": np.mean([r.get("similarity_score", 0) for r in results["validation_results"]]),
            "avg_backend_quality": np.mean([r.get("backend_quality", 0) for r in results["validation_results"]]),
            "avg_expected_quality": np.mean([r.get("expected_quality", 0) for r in results["validation_results"]])
        }
        
        logger.info(f"BDD validation test suite completed: {passed_tests}/{total_tests} tests passed")
        return results
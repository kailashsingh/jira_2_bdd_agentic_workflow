"""
Essential Hugging Face models for BDD generation and validation.
Focused on core models needed for BDD scenario creation and validation against backend results.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelRecommendation:
    """Data class for essential model recommendations for BDD testing."""
    model_name: str
    huggingface_id: str
    use_case: str
    strengths: List[str]
    limitations: List[str]
    performance_score: float
    recommended_for: List[str]
    memory_requirements: str
    inference_speed: str

class HuggingFaceModelResearch:
    """Essential models for BDD scenario creation and validation."""
    
    # Only essential models for our BDD workflow
    RECOMMENDED_MODELS = {
        # Primary Model for Code Understanding and Requirements Analysis
        "codebert": ModelRecommendation(
            model_name="CodeBERT", 
            huggingface_id="microsoft/codebert-base",
            use_case="Primary model for understanding JIRA requirements and code context",
            strengths=[
                "Pre-trained on code-text pairs",
                "Excellent for understanding requirements to code mapping",
                "Fast inference for requirement analysis",
                "Essential for BDD structure generation"
            ],
            limitations=[
                "Encoder-only, requires additional generation model",
                "Best used for analysis rather than generation"
            ],
            performance_score=9.0,
            recommended_for=["requirement_analysis", "bdd_structure_generation", "code_context_understanding"],
            memory_requirements="~500MB",
            inference_speed="Fast"
        ),
        
        # Sentence Similarity for BDD Validation
        "sentence_transformer": ModelRecommendation(
            model_name="Sentence Transformer",
            huggingface_id="all-MiniLM-L6-v2",
            use_case="Validating generated BDD scenarios against expected results",
            strengths=[
                "Excellent for comparing generated BDD against expected outcomes",
                "Fast validation of scenario similarity",
                "Compact model perfect for testing framework",
                "Essential for automated BDD quality assessment"
            ],
            limitations=[
                "No generation capabilities",
                "Limited to validation and similarity tasks only"
            ],
            performance_score=9.5,
            recommended_for=["bdd_validation", "scenario_comparison", "quality_assessment", "backend_result_validation"],
            memory_requirements="~200MB",
            inference_speed="Very Fast"
        )
    }
    
    # Focused model selection strategies for BDD testing
    SELECTION_STRATEGIES = {
        "bdd_validation": {
            "primary": ["sentence_transformer"],
            "description": "Essential models for validating BDD scenarios against backend results",
            "use_case": "Comparing generated BDD with backend results and expected outcomes"
        },
        
        "requirement_analysis": {
            "primary": ["codebert"],
            "description": "Analyze JIRA requirements and code context for BDD generation",
            "use_case": "Understanding requirements for accurate BDD scenario creation"
        },
        
        "complete_workflow": {
            "primary": ["codebert", "sentence_transformer"],
            "description": "Complete BDD workflow from requirement analysis to validation",
            "use_case": "End-to-end BDD creation and validation pipeline"
        }
    }
    
    def __init__(self):
        """Initialize the essential model research helper."""
        self.recommendations = self.RECOMMENDED_MODELS
        self.strategies = self.SELECTION_STRATEGIES
    
    def get_model_recommendation(self, model_key: str) -> Optional[ModelRecommendation]:
        """Get detailed recommendation for a specific essential model."""
        return self.recommendations.get(model_key)
    
    def get_models_for_use_case(self, use_case: str) -> List[ModelRecommendation]:
        """Get recommended essential models for a specific use case."""
        matching_models = []
        
        for model in self.recommendations.values():
            if use_case in model.recommended_for or use_case in model.use_case.lower():
                matching_models.append(model)
        
        # Sort by performance score
        matching_models.sort(key=lambda x: x.performance_score, reverse=True)
        return matching_models
    
    def get_strategy_recommendations(self, strategy: str) -> Dict[str, Any]:
        """Get model recommendations for a specific BDD strategy."""
        if strategy not in self.strategies:
            available_strategies = list(self.strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy}. Available: {available_strategies}")
        
        strategy_info = self.strategies[strategy].copy()
        strategy_models = []
        
        for model_key in strategy_info["primary"]:
            if model_key in self.recommendations:
                strategy_models.append(self.recommendations[model_key])
        
        strategy_info["models"] = strategy_models
        return strategy_info
    
    def get_essential_model_config(self) -> Dict[str, Any]:
        """Get essential model configuration for BDD testing framework."""
        return {
            "codebert": {
                "model_id": "microsoft/codebert-base",
                "task": "feature-extraction",
                "use_for": ["requirement_analysis", "code_understanding"],
                "memory_limit": "500MB",
                "performance_priority": "balanced"
            },
            "sentence_transformer": {
                "model_id": "all-MiniLM-L6-v2", 
                "task": "sentence-similarity",
                "use_for": ["bdd_validation", "similarity_comparison"],
                "memory_limit": "200MB",
                "performance_priority": "speed"
            }
        }
    
    def validate_bdd_workflow_readiness(self) -> Dict[str, Any]:
        """Validate that essential models are ready for BDD workflow."""
        validation_result = {
            "workflow_ready": True,
            "models_validated": 0,
            "total_models": len(self.RECOMMENDED_MODELS),
            "issues": [],
            "recommendations": []
        }
        
        # Check each essential model
        for model_key, model in self.RECOMMENDED_MODELS.items():
            try:
                # Validate model configuration
                if model.performance_score >= 8.0:
                    validation_result["models_validated"] += 1
                else:
                    validation_result["issues"].append(f"Model {model_key} has low performance score: {model.performance_score}")
                    validation_result["workflow_ready"] = False
                    
            except Exception as e:
                validation_result["issues"].append(f"Model {model_key} validation error: {str(e)}")
                validation_result["workflow_ready"] = False
        
        # Add BDD-specific recommendations
        validation_result["recommendations"] = [
            "Use CodeBERT for JIRA requirement analysis and code context understanding",
            "Use Sentence Transformer for validating generated BDD against backend results",
            "Load models only when needed to optimize memory usage",
            "Cache model instances for repeated BDD validation operations",
            "Set similarity threshold to 70% for BDD validation acceptance"
        ]
        
        logger.info(f"BDD workflow validation: {validation_result['models_validated']}/{validation_result['total_models']} models ready")
        
        return validation_result

# Essential convenience functions for BDD testing
def get_essential_models_for_bdd() -> List[str]:
    """Get the essential models for BDD generation and validation."""
    return ["codebert", "sentence_transformer"]

def get_bdd_validation_model() -> str:
    """Get the primary model for BDD validation."""
    return "sentence_transformer"

def get_requirement_analysis_model() -> str:
    """Get the primary model for requirement analysis."""
    return "codebert"

def get_essential_model_config() -> Dict[str, Any]:
    """Get essential model configuration for BDD testing framework."""
    research = HuggingFaceModelResearch()
    return research.get_essential_model_config()

def validate_bdd_setup() -> Dict[str, Any]:
    """Validate BDD testing setup readiness."""
    research = HuggingFaceModelResearch()
    return research.validate_bdd_workflow_readiness()

# Essential model validation criteria for BDD testing
ESSENTIAL_MODEL_CRITERIA = {
    "performance_thresholds": {
        "minimum_score": 8.0,  # High threshold for essential models
        "required_memory": "< 1GB",  # Memory efficient for testing
        "required_speed": "Fast or Very Fast"
    },
    "bdd_specific_requirements": {
        "requirement_analysis": {
            "model": "microsoft/codebert-base",
            "task": "feature-extraction",
            "expected_performance": "> 8.0"
        },
        "bdd_validation": {
            "model": "all-MiniLM-L6-v2",
            "task": "sentence-similarity", 
            "expected_performance": "> 9.0"
        }
    },
    "validation_thresholds": {
        "similarity_acceptance": 0.70,  # 70% similarity for BDD validation
        "requirement_coverage": 0.80,   # 80% requirement coverage
        "structure_completeness": 1.0   # 100% BDD structure (Given/When/Then)
    }
}
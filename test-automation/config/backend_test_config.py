"""
Example configuration for backend functional testing
Copy this file and customize for your environment
"""

# Backend Configuration
BACKEND_CONFIG = {
    "base_url": "http://localhost:8000",
    "timeout": 30,
    "max_retries": 3,
    "endpoints": {
        "health": "/health",
        "workflow_trigger": "/workflow/trigger/tickets",
        "workflow_status": "/workflow/status/{run_id}",
        "debug_rag": "/debug/rag-search",
        "debug_jira": "/debug/jira-tickets",
        "debug_navigation": "/debug/test-navigation",
        "debug_url_validation": "/debug/validate-url"
    }
}

# Test Data Configuration
TEST_DATA_CONFIG = {
    "sample_tickets": [
        {
            "key": "FUNC-001",
            "summary": "User registration functionality",
            "description": "Implement user registration with email verification",
            "acceptance_criteria": "Given a new user, when they register, then they should receive verification email",
            "issue_type": "Story",
            "components": ["frontend", "backend"]
        },
        {
            "key": "FUNC-002", 
            "summary": "Payment processing integration",
            "description": "Integrate with payment gateway for order processing",
            "acceptance_criteria": "Given valid payment details, when order is submitted, then payment should be processed",
            "issue_type": "Story",
            "components": ["backend", "payment"]
        },
        {
            "key": "FUNC-003",
            "summary": "Search functionality optimization",
            "description": "Optimize search performance for large datasets",
            "acceptance_criteria": "Given search query, when search is performed, then results should load within 2 seconds",
            "issue_type": "Improvement",
            "components": ["backend", "database"]
        }
    ],
    "test_navigation_scenarios": [
        {
            "summary": "Login page navigation test",
            "description": "Test navigation to login page at https://example.com/login",
            "acceptance_criteria": "User should be able to navigate to login page and see login form"
        },
        {
            "summary": "Shopping cart workflow test",
            "description": "Test complete shopping cart workflow from product selection to checkout",
            "acceptance_criteria": "User should be able to add products, view cart, and proceed to checkout"
        }
    ]
}

# Agent Testing Configuration
AGENT_TEST_CONFIG = {
    "bdd_generator": {
        "test_scenarios": [
            {
                "name": "valid_user_story",
                "ticket_data": TEST_DATA_CONFIG["sample_tickets"][0],
                "expected_testable": True,
                "expected_bdd_quality": 0.8
            },
            {
                "name": "improvement_ticket",
                "ticket_data": TEST_DATA_CONFIG["sample_tickets"][2],
                "expected_testable": True,
                "expected_bdd_quality": 0.7
            }
        ],
        "performance_thresholds": {
            "max_generation_time": 10.0,  # seconds
            "min_quality_score": 0.7,
            "max_memory_usage": 1000  # MB
        }
    },
    "workflow_orchestrator": {
        "test_scenarios": [
            {
                "name": "single_ticket_workflow",
                "mode": "tickets",
                "ticket_keys": ["FUNC-001"],
                "expected_nodes": ["fetch_jira_tickets", "process_ticket", "generate_tests", "create_pr"],
                "max_execution_time": 60.0
            },
            {
                "name": "batch_ticket_workflow", 
                "mode": "tickets",
                "ticket_keys": ["FUNC-001", "FUNC-002"],
                "expected_nodes": ["fetch_jira_tickets", "process_ticket", "generate_tests", "create_pr"],
                "max_execution_time": 120.0
            }
        ]
    }
}

# Performance Testing Configuration
PERFORMANCE_CONFIG = {
    "load_tests": [
        {
            "endpoint": "/health",
            "concurrent_requests": 10,
            "total_requests": 100,
            "expected_success_rate": 0.99,
            "max_avg_response_time": 0.5
        },
        {
            "endpoint": "/debug/rag-search",
            "concurrent_requests": 5,
            "total_requests": 50,
            "expected_success_rate": 0.95,
            "max_avg_response_time": 2.0
        }
    ],
    "stress_tests": [
        {
            "endpoint": "/workflow/trigger/tickets",
            "concurrent_requests": 3,
            "total_requests": 10,
            "expected_success_rate": 0.90,
            "max_avg_response_time": 30.0
        }
    ]
}

# Validation Thresholds
VALIDATION_THRESHOLDS = {
    "api": {
        "health_check_max_time": 1.0,
        "workflow_trigger_max_time": 5.0,
        "debug_endpoint_max_time": 10.0
    },
    "agents": {
        "bdd_generation_max_time": 15.0,
        "workflow_execution_max_time": 300.0,
        "min_bdd_quality_score": 0.7
    },
    "models": {
        "similarity_threshold": 0.8,
        "performance_threshold": 2.0,
        "accuracy_threshold": 0.85
    }
}

# Mock Configuration for Isolated Testing
MOCK_CONFIG = {
    "enable_mocks": True,
    "mock_external_services": True,
    "mock_llm_responses": True,
    "mock_data": {
        "jira_tickets": TEST_DATA_CONFIG["sample_tickets"],
        "github_files": [
            {"path": "features/authentication.feature", "content": "Feature: Authentication"},
            {"path": "step_definitions/auth.steps.ts", "content": "import { Given } from '@wdio/cucumber-framework';"}
        ],
        "rag_search_results": [
            {"content": "Feature: Login functionality", "metadata": {"path": "login.feature", "similarity": 0.9}}
        ]
    }
}

# Test Suite Configuration
TEST_SUITE_CONFIG = {
    "comprehensive_backend_test": {
        "name": "comprehensive_backend_functional_test",
        "bdd_scenarios": AGENT_TEST_CONFIG["bdd_generator"]["test_scenarios"],
        "workflow_scenarios": AGENT_TEST_CONFIG["workflow_orchestrator"]["test_scenarios"],
        "tool_scenarios": [
            {"tool_name": "jira_tools", "test_name": "fetch_tickets", "expected_ticket_count": 3},
            {"tool_name": "github_tools", "test_name": "fetch_files", "expected_file_count": 2},
            {"tool_name": "rag_tools", "test_name": "search_code", "query": "authentication", "expected_results": 1},
            {"tool_name": "application_tools", "test_name": "navigate_app", "navigation_required": False}
        ]
    },
    "performance_test": {
        "name": "backend_performance_test",
        "bdd_performance": {"iterations": 5},
        "workflow_performance": {"iterations": 3}
    }
}

# Environment-Specific Overrides
ENVIRONMENT_CONFIGS = {
    "development": {
        "backend_url": "http://localhost:8000",
        "timeout": 30,
        "enable_debug": True,
        "log_level": "DEBUG"
    },
    "staging": {
        "backend_url": "https://staging-api.example.com",
        "timeout": 60,
        "enable_debug": False,
        "log_level": "INFO"
    },
    "production": {
        "backend_url": "https://api.example.com",
        "timeout": 30,
        "enable_debug": False,
        "log_level": "ERROR"
    }
}

# Export configuration
__all__ = [
    'BACKEND_CONFIG',
    'TEST_DATA_CONFIG', 
    'AGENT_TEST_CONFIG',
    'PERFORMANCE_CONFIG',
    'VALIDATION_THRESHOLDS',
    'MOCK_CONFIG',
    'TEST_SUITE_CONFIG',
    'ENVIRONMENT_CONFIGS'
]
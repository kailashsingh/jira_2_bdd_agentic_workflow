import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil
from typing import Generator, Any, Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend" / "src"))

# Import test modules
from src.models.model_manager import ModelManager
from src.utils.test_helpers import TestDataGenerator
from src.backend_integration import BackendTestClient, AgentTestRunner
from src.config.logging import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Test configuration
TEST_CONFIG = {
    "models": {
        "codebert": {
            "name": "microsoft/codebert-base",
            "task": "feature-extraction",
            "description": "Primary model for requirement analysis and BDD understanding"
        },
        "sentence-transformer": {
            "name": "all-MiniLM-L6-v2", 
            "task": "sentence-similarity",
            "description": "Essential for BDD validation and similarity comparison"
        }
    },
    "thresholds": {
        "similarity_threshold": 0.8,
        "performance_threshold": 2.0,
        "accuracy_threshold": 0.85,
        "bdd_quality_threshold": 0.7,
        "api_response_threshold": 5.0,
        "codebert_similarity_threshold": 0.6  # Specific threshold for CodeBERT BDD validation
    },
    "backend": {
        "base_url": "http://localhost:8000",
        "timeout": 30,
        "max_retries": 3
    },
    "test_data": {
        "sample_jira_tickets": [
            {
                "key": "TEST-123",
                "summary": "User login functionality",
                "description": "As a user, I want to log into the application so that I can access my account",
                "acceptance_criteria": "Given a valid user, when they enter credentials, then they should be logged in",
                "components": ["frontend"],
                "issue_type": "Story"
            },
            {
                "key": "TEST-456", 
                "summary": "Shopping cart checkout",
                "description": "As a customer, I want to checkout my cart so that I can purchase items",
                "acceptance_criteria": "Given items in cart, when checkout is initiated, then payment should be processed",
                "components": ["backend"],
                "issue_type": "Story"
            }
        ],
        "sample_bdd_scenarios": [
            """
            Feature: User Login
            
            Scenario: Successful login
                Given a user with valid credentials
                When they submit the login form
                Then they should be redirected to the dashboard
            """,
            """
            Feature: Shopping Cart
            
            Scenario: Checkout process
                Given items are in the shopping cart
                When the user clicks checkout
                Then the payment form should be displayed
            """
        ]
    },
    "thresholds": {
        "similarity_threshold": 0.8,
        "performance_threshold": 2.0,
        "accuracy_threshold": 0.85,
        "bdd_quality_threshold": 0.7,
        "api_response_threshold": 5.0,
        "codebert_similarity_threshold": 0.6  # Specific threshold for CodeBERT BDD validation
    },
}

# Test markers
def pytest_configure(config):
    """Configure test markers"""
    config.addinivalue_line("markers", "smoke: Quick smoke tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "model: Model-specific tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "backend: Backend API tests")
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "bdd: BDD generation tests")
    config.addinivalue_line("markers", "workflow: Workflow orchestration tests")

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG

@pytest.fixture
def model_manager():
    """Provide ModelManager instance"""
    return ModelManager()

@pytest.fixture  
def test_data_generator():
    """Provide TestDataGenerator instance"""
    return TestDataGenerator()

@pytest.fixture
def backend_test_client():
    """Provide BackendTestClient instance"""
    return BackendTestClient(
        base_url=TEST_CONFIG["backend"]["base_url"],
        timeout=TEST_CONFIG["backend"]["timeout"]
    )

@pytest.fixture
def agent_test_runner():
    """Provide AgentTestRunner instance"""
    return AgentTestRunner()

@pytest.fixture
async def async_backend_client(backend_test_client):
    """Provide async backend client with proper setup/teardown"""
    await backend_test_client.initialize_async_client()
    yield backend_test_client
    await backend_test_client.close_async_client()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def mock_jira_tools():
    """Mock JIRA tools for testing."""
    with patch('src.tools.jira_tools.JiraTools') as mock:
        mock_instance = Mock()
        mock_instance.get_sprint_tickets.return_value = TEST_CONFIG["test_data"]["sample_jira_tickets"]
        mock_instance.get_tickets.return_value = TEST_CONFIG["test_data"]["sample_jira_tickets"]
        mock_instance.update_ticket_comment.return_value = True
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_github_tools():
    """Mock GitHub tools for testing."""
    with patch('src.tools.github_tools.GitHubTools') as mock:
        mock_instance = Mock()
        mock_instance.get_feature_files.return_value = ["feature1.feature", "feature2.feature"]
        mock_instance.get_step_definitions.return_value = ["steps1.py", "steps2.py"]
        mock_instance.create_branch.return_value = True
        mock_instance.create_or_update_file.return_value = True
        mock_instance.create_pull_request.return_value = "https://github.com/test/repo/pull/123"
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_rag_tools():
    """Mock RAG tools for testing."""
    with patch('src.tools.rag_tools.RAGTools') as mock:
        mock_instance = Mock()
        mock_instance.search_similar_code.return_value = [
            {"content": "similar code 1", "score": 0.9},
            {"content": "similar code 2", "score": 0.8}
        ]
        mock_instance.index_codebase.return_value = True
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_application_tools():
    """Mock application tools for testing."""
    with patch('src.tools.application_tools.ApplicationTools') as mock:
        mock_instance = Mock()
        mock_instance.needs_navigation.return_value = True
        mock_instance.navigate_and_collect_data.return_value = {
            "page_title": "Test Page",
            "elements": ["button", "input", "form"],
            "forms": [{"action": "/login", "method": "POST"}]
        }
        mock_instance.navigate_and_collect_data_using_mcp = AsyncMock(
            return_value="Application data collected successfully"
        )
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_bdd_generator():
    """Mock BDD generator agent for testing."""
    with patch('src.agents.bdd_generator_agent.BDDGeneratorAgent') as mock:
        mock_instance = Mock()
        mock_instance.is_testable.return_value = True
        mock_instance.generate_bdd_scenarios.return_value = {
            "ticket_key": "TEST-123",
            "feature_file_name": "test_feature.feature",
            "steps_file_name": "test_steps.py", 
            "feature": TEST_CONFIG["test_data"]["sample_bdd_scenarios"][0],
            "step_definitions": "def test_step(): pass"
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_jira_ticket():
    """Provide a sample JIRA ticket for testing."""
    return TEST_CONFIG["test_data"]["sample_jira_tickets"][0]

@pytest.fixture
def sample_bdd_scenario():
    """Provide a sample BDD scenario for testing."""
    return TEST_CONFIG["test_data"]["sample_bdd_scenarios"][0]

@pytest.fixture
def model_manager():
    """Provide a model manager instance for testing."""
    from src.models.model_manager import ModelManager
    return ModelManager()

@pytest.fixture
def codebert_model_config():
    """Provide CodeBERT-specific configuration for testing."""
    return {
        "model_key": "codebert",
        "model_name": "microsoft/codebert-base",
        "expected_type": "encoder",
        "task": "feature-extraction",
        "similarity_threshold": TEST_CONFIG["thresholds"]["codebert_similarity_threshold"]
    }

@pytest.fixture
def bdd_validation_test_data():
    """Provide test data specifically for BDD validation with CodeBERT."""
    return {
        "jira_requirement": """
        As a user, I want to login to the application so that I can access my account.
        Acceptance Criteria:
        - User enters valid username and password
        - System authenticates user credentials
        - User is redirected to dashboard on success
        - Error message shown for invalid credentials
        """,
        "good_bdd_scenario": """
        @TEST-123 @AutoGenerated
        Feature: User Authentication
          As a user
          I want to login to the application
          So that I can access my account
          
          Scenario: Successful login with valid credentials
            Given user is on the login page
            When user enters valid username "testuser"
            And user enters valid password "testpass"
            And user clicks the login button
            Then user should be redirected to dashboard
            And user should see welcome message
            
          Scenario: Failed login with invalid credentials
            Given user is on the login page
            When user enters invalid username "invalid"
            And user enters invalid password "wrong"
            And user clicks the login button
            Then user should see error message
            And user should remain on login page
        """,
        "poor_bdd_scenario": """
        Feature: Login
          Scenario: Login
            Given login page
            When click
            Then something
        """,
        "backend_generated_bdd": """
        @TEST-123 @AutoGenerated
        Feature: User login functionality
          Scenario: User authentication
            Given user has valid credentials
            When user attempts to login
            Then user should be authenticated
        """
    }

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "JIRA_URL": "https://test-jira.com",
        "JIRA_EMAIL": "test@example.com", 
        "JIRA_API_TOKEN": "test-token",
        "GITHUB_TOKEN": "test-github-token",
        "GITHUB_REPO": "test/repo",
        "OPENAI_API_KEY": "test-openai-key",
        "ENVIRONMENT": "test"
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock:
        mock_instance = Mock()
        mock_instance.chat.completions.create.return_value = Mock(
            choices=[
                Mock(message=Mock(content="Generated BDD scenario"))
            ]
        )
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_transformers():
    """Mock transformers models for testing."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer, \
         patch('transformers.AutoModel') as mock_model:
        
        mock_tokenizer.from_pretrained.return_value = Mock(
            encode=Mock(return_value=[1, 2, 3, 4, 5]),
            decode=Mock(return_value="decoded text")
        )
        
        mock_model.from_pretrained.return_value = Mock()
        
        yield {
            "tokenizer": mock_tokenizer,
            "model": mock_model
        }

@pytest.fixture
def benchmark_suite():
    """Provide benchmark test suite configuration."""
    return {
        "test_cases": [
            {
                "name": "simple_ticket",
                "ticket": TEST_CONFIG["test_data"]["sample_jira_tickets"][0],
                "expected_scenarios": 2,
                "complexity": "low"
            },
            {
                "name": "complex_ticket",
                "ticket": {
                    **TEST_CONFIG["test_data"]["sample_jira_tickets"][1],
                    "description": "Complex multi-step workflow " * 50  # Make it complex
                },
                "expected_scenarios": 5,
                "complexity": "high"
            }
        ],
        "models_to_test": ["codebert", "roberta", "bert"],
        "metrics": ["accuracy", "speed", "memory_usage", "similarity_score"]
    }
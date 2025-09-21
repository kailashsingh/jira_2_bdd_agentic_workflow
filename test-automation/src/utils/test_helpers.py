"""
Test utilities and helpers for the BDD testing framework.
"""

import json
import time
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Generator
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass, asdict
import uuid
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

@dataclass
class TestJiraTicket:
    """Test JIRA ticket data structure."""
    key: str
    summary: str
    description: str
    acceptance_criteria: str
    components: List[str]
    status: str = "To Do"
    assignee: Optional[str] = None
    priority: str = "Medium"
    story_points: Optional[int] = None
    created: Optional[str] = None
    updated: Optional[str] = None

@dataclass
class TestBDDScenario:
    """Test BDD scenario data structure."""
    feature_name: str
    scenario_name: str
    given_steps: List[str]
    when_steps: List[str]
    then_steps: List[str]
    tags: List[str] = None

@dataclass
class TestWorkflowRun:
    """Test workflow run data structure."""
    run_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    ticket_keys: List[str] = None
    error: Optional[str] = None
    result: Optional[Dict] = None

class TestDataGenerator:
    """Generates test data for various testing scenarios."""
    
    def __init__(self):
        self.fake = Faker()
        
    def generate_jira_ticket(self, 
                           key: Optional[str] = None,
                           complexity: str = "medium",
                           components: Optional[List[str]] = None) -> TestJiraTicket:
        """Generate a test JIRA ticket."""
        if not key:
            key = f"TEST-{random.randint(100, 9999)}"
            
        if not components:
            components = [random.choice(["frontend", "backend", "api", "database"])]
            
        base_summaries = {
            "simple": [
                "Update button color",
                "Fix typo in label",
                "Add logging statement",
                "Update configuration"
            ],
            "medium": [
                "User login functionality",
                "Shopping cart checkout",
                "Email notification system",
                "File upload feature"
            ],
            "complex": [
                "Multi-step user registration workflow",
                "Advanced search with filters and pagination", 
                "Real-time chat system with notifications",
                "Payment processing with multiple gateways"
            ]
        }
        
        base_descriptions = {
            "simple": "A simple change that requires minimal testing.",
            "medium": "A moderate feature that requires standard BDD testing coverage.",
            "complex": "A complex feature that requires comprehensive testing scenarios."
        }
        
        summaries = base_summaries.get(complexity, base_summaries["medium"])
        summary = random.choice(summaries)
        
        description = f"{base_descriptions[complexity]} {self.fake.text(max_nb_chars=200)}"
        
        acceptance_criteria = self._generate_acceptance_criteria(summary, complexity)
        
        return TestJiraTicket(
            key=key,
            summary=summary,
            description=description,
            acceptance_criteria=acceptance_criteria,
            components=components,
            assignee=self.fake.name(),
            story_points=random.randint(1, 8) if complexity != "simple" else 1,
            created=self.fake.date_time_between(start_date="-30d", end_date="now").isoformat(),
            updated=self.fake.date_time_between(start_date="-7d", end_date="now").isoformat()
        )
    
    def _generate_acceptance_criteria(self, summary: str, complexity: str) -> str:
        """Generate acceptance criteria based on summary and complexity."""
        criteria_templates = {
            "simple": [
                f"Given the {summary.lower()}, when the user interacts with it, then it should work correctly."
            ],
            "medium": [
                f"Given a user wants to use {summary.lower()}",
                f"When they follow the standard process",
                f"Then they should achieve the expected outcome",
                f"And the system should handle edge cases appropriately"
            ],
            "complex": [
                f"Given a user needs {summary.lower()}",
                f"When they start the multi-step process",
                f"Then each step should be validated",
                f"And error handling should be comprehensive",
                f"And performance should meet requirements",
                f"And the system should be secure and reliable"
            ]
        }
        
        criteria = criteria_templates.get(complexity, criteria_templates["medium"])
        return "\n".join(criteria)
    
    def generate_bdd_scenario(self, 
                            ticket: TestJiraTicket,
                            scenario_type: str = "positive") -> TestBDDScenario:
        """Generate a BDD scenario from a JIRA ticket."""
        feature_name = ticket.summary
        
        if scenario_type == "positive":
            scenario_name = f"Successful {ticket.summary.lower()}"
            given_steps = [f"the user has access to {ticket.summary.lower()}"]
            when_steps = [f"they use the {ticket.summary.lower()} feature"]
            then_steps = [f"the {ticket.summary.lower()} should work correctly"]
        elif scenario_type == "negative":
            scenario_name = f"Error handling for {ticket.summary.lower()}"
            given_steps = [f"invalid input is provided for {ticket.summary.lower()}"]
            when_steps = [f"the user attempts to use {ticket.summary.lower()}"]
            then_steps = [f"an appropriate error message should be displayed"]
        else:  # edge case
            scenario_name = f"Edge case for {ticket.summary.lower()}"
            given_steps = [f"boundary conditions exist for {ticket.summary.lower()}"]
            when_steps = [f"the user tests edge scenarios"]
            then_steps = [f"the system should handle them gracefully"]
            
        return TestBDDScenario(
            feature_name=feature_name,
            scenario_name=scenario_name,
            given_steps=given_steps,
            when_steps=when_steps,
            then_steps=then_steps,
            tags=[ticket.key.lower(), scenario_type]
        )
    
    def generate_test_suite(self, num_tickets: int = 5) -> List[TestJiraTicket]:
        """Generate a complete test suite with various ticket types."""
        tickets = []
        complexities = ["simple", "medium", "complex"]
        
        for i in range(num_tickets):
            complexity = complexities[i % len(complexities)]
            ticket = self.generate_jira_ticket(complexity=complexity)
            tickets.append(ticket)
            
        return tickets
    
    def generate_workflow_run(self, 
                            tickets: List[TestJiraTicket],
                            status: str = "completed") -> TestWorkflowRun:
        """Generate a test workflow run."""
        run_id = f"test_run_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now().isoformat()
        
        completed_at = None
        error = None
        result = None
        
        if status == "completed":
            completed_at = (datetime.now() + timedelta(minutes=random.randint(1, 30))).isoformat()
            result = {
                "tickets_processed": len(tickets),
                "scenarios_generated": len(tickets) * random.randint(2, 5),
                "success_rate": random.uniform(0.8, 1.0)
            }
        elif status == "failed":
            error = random.choice([
                "Model loading failed",
                "JIRA connection timeout",
                "GitHub API rate limit exceeded",
                "Invalid ticket format"
            ])
            
        return TestWorkflowRun(
            run_id=run_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            ticket_keys=[t.key for t in tickets],
            error=error,
            result=result
        )

class MockFactory:
    """Factory for creating mock objects for testing."""
    
    @staticmethod
    def create_mock_jira_tools(tickets: List[TestJiraTicket] = None) -> Mock:
        """Create a mock JIRA tools instance."""
        if tickets is None:
            generator = TestDataGenerator()
            tickets = generator.generate_test_suite()
            
        mock = Mock()
        mock.get_sprint_tickets.return_value = [asdict(t) for t in tickets]
        mock.get_tickets.return_value = [asdict(t) for t in tickets]
        mock.update_ticket_comment.return_value = True
        return mock
    
    @staticmethod
    def create_mock_github_tools() -> Mock:
        """Create a mock GitHub tools instance."""
        mock = Mock()
        mock.get_feature_files.return_value = [
            "login.feature",
            "checkout.feature", 
            "search.feature"
        ]
        mock.get_step_definitions.return_value = [
            "login_steps.py",
            "checkout_steps.py",
            "search_steps.py"
        ]
        mock.create_branch.return_value = True
        mock.create_or_update_file.return_value = True
        mock.create_pull_request.return_value = "https://github.com/test/repo/pull/123"
        mock.set_repository.return_value = None
        return mock
    
    @staticmethod
    def create_mock_rag_tools() -> Mock:
        """Create a mock RAG tools instance."""
        mock = Mock()
        mock.search_similar_code.return_value = [
            {"content": "similar step definition 1", "score": 0.9},
            {"content": "similar feature scenario", "score": 0.8},
            {"content": "related test case", "score": 0.7}
        ]
        mock.index_codebase.return_value = True
        return mock
    
    @staticmethod
    def create_mock_application_tools() -> Mock:
        """Create a mock application tools instance."""
        mock = Mock()
        mock.needs_navigation.return_value = True
        mock.navigate_and_collect_data.return_value = {
            "page_title": "Test Application",
            "elements": ["button#login", "input[name='username']", "input[name='password']"],
            "forms": [{"action": "/login", "method": "POST"}],
            "links": [{"href": "/register", "text": "Sign Up"}]
        }
        mock.navigate_and_collect_data_using_mcp = AsyncMock(
            return_value="Detailed application data collected successfully"
        )
        return mock
    
    @staticmethod
    def create_mock_bdd_generator(generation_result: Optional[Dict] = None) -> Mock:
        """Create a mock BDD generator agent."""
        if generation_result is None:
            generation_result = {
                "ticket_key": "TEST-123",
                "feature_file_name": "test_feature.feature",
                "steps_file_name": "test_steps.py",
                "feature": """
                Feature: Test Feature
                
                Scenario: Test scenario
                    Given a test condition
                    When an action is performed
                    Then the expected result occurs
                """,
                "step_definitions": """
                from pytest_bdd import given, when, then
                
                @given('a test condition')
                def test_condition():
                    pass
                    
                @when('an action is performed')
                def perform_action():
                    pass
                    
                @then('the expected result occurs')
                def verify_result():
                    pass
                """
            }
        
        mock = Mock()
        mock.is_testable.return_value = True
        mock.generate_bdd_scenarios.return_value = generation_result
        return mock

class TestFileManager:
    """Manages test files and temporary directories."""
    
    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []
    
    def create_temp_dir(self) -> Path:
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, content: str, suffix: str = ".tmp") -> Path:
        """Create a temporary file with content."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        temp_path = Path(path)
        
        with open(fd, 'w') as f:
            f.write(content)
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def create_feature_file(self, scenarios: List[TestBDDScenario], temp_dir: Path) -> Path:
        """Create a Gherkin feature file from scenarios."""
        if not scenarios:
            raise ValueError("At least one scenario is required")
            
        feature_name = scenarios[0].feature_name
        content = f"Feature: {feature_name}\n\n"
        
        for scenario in scenarios:
            content += f"  Scenario: {scenario.scenario_name}\n"
            
            for step in scenario.given_steps:
                content += f"    Given {step}\n"
            for step in scenario.when_steps:
                content += f"    When {step}\n"
            for step in scenario.then_steps:
                content += f"    Then {step}\n"
                
            content += "\n"
        
        feature_file = temp_dir / f"{feature_name.lower().replace(' ', '_')}.feature"
        feature_file.write_text(content)
        return feature_file
    
    def create_steps_file(self, scenarios: List[TestBDDScenario], temp_dir: Path) -> Path:
        """Create a pytest-bdd steps file from scenarios."""
        if not scenarios:
            raise ValueError("At least one scenario is required")
            
        feature_name = scenarios[0].feature_name
        
        content = f"""
from pytest_bdd import scenarios, given, when, then, parsers

# Load scenarios from feature file
scenarios('../features/{feature_name.lower().replace(' ', '_')}.feature')

"""
        
        # Collect unique steps
        all_steps = set()
        for scenario in scenarios:
            all_steps.update(scenario.given_steps)
            all_steps.update(scenario.when_steps)
            all_steps.update(scenario.then_steps)
        
        # Generate step definitions
        for step in sorted(all_steps):
            step_func_name = step.lower().replace(' ', '_').replace("'", "")
            content += f"""
@given('{step}')
@when('{step}')
@then('{step}')
def {step_func_name}():
    '''Implementation for: {step}'''
    pass
"""
        
        steps_file = temp_dir / f"{feature_name.lower().replace(' ', '_')}_steps.py"
        steps_file.write_text(content)
        return steps_file
    
    def cleanup(self):
        """Clean up all temporary files and directories."""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
        
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                pass
        
        self.temp_files.clear()
        self.temp_dirs.clear()

class PerformanceTimer:
    """Context manager for measuring performance."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def __str__(self):
        return f"{self.name}: {self.duration:.3f}s"

class AsyncTestHelper:
    """Helper for async testing scenarios."""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run a coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
    
    @staticmethod
    async def simulate_async_delay(min_delay: float = 0.1, max_delay: float = 0.5):
        """Simulate async operation delay."""
        delay = random.uniform(min_delay, max_delay)
        await asyncio.sleep(delay)
    
    @staticmethod
    def create_async_mock(*args, **kwargs) -> AsyncMock:
        """Create an AsyncMock with optional return values."""
        mock = AsyncMock()
        if 'return_value' in kwargs:
            mock.return_value = kwargs['return_value']
        return mock

class ValidationHelper:
    """Helper for validating test results."""
    
    @staticmethod
    def validate_bdd_feature(content: str) -> Dict[str, Any]:
        """Validate BDD feature file content."""
        issues = []
        warnings = []
        
        # Check for required elements
        if "Feature:" not in content:
            issues.append("Missing Feature declaration")
        
        if "Scenario:" not in content:
            issues.append("Missing Scenario declarations")
        
        # Check for BDD keywords
        required_keywords = ["Given", "When", "Then"]
        for keyword in required_keywords:
            if keyword not in content:
                warnings.append(f"Missing '{keyword}' steps")
        
        # Count scenarios
        scenario_count = content.count("Scenario:")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "scenario_count": scenario_count,
            "has_all_keywords": len(warnings) == 0
        }
    
    @staticmethod
    def validate_step_definitions(content: str) -> Dict[str, Any]:
        """Validate pytest-bdd step definitions."""
        issues = []
        warnings = []
        
        # Check for imports
        required_imports = ["pytest_bdd", "given", "when", "then"]
        for imp in required_imports:
            if imp not in content:
                issues.append(f"Missing import: {imp}")
        
        # Check for step decorators
        step_decorators = ["@given", "@when", "@then"]
        decorator_count = sum(content.count(decorator) for decorator in step_decorators)
        
        if decorator_count == 0:
            issues.append("No step definitions found")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "step_count": decorator_count
        }
    
    @staticmethod
    def validate_jira_ticket(ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JIRA ticket data structure."""
        required_fields = ["key", "summary", "description"]
        missing_fields = [field for field in required_fields if field not in ticket_data]
        
        warnings = []
        if "acceptance_criteria" not in ticket_data:
            warnings.append("Missing acceptance_criteria field")
        
        if "components" not in ticket_data:
            warnings.append("Missing components field")
        
        return {
            "is_valid": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "warnings": warnings
        }

# Utility functions
def generate_test_reports_dir() -> Path:
    """Create a test reports directory."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (reports_dir / "coverage").mkdir(exist_ok=True)
    (reports_dir / "performance").mkdir(exist_ok=True)
    (reports_dir / "models").mkdir(exist_ok=True)
    
    return reports_dir

def load_test_config() -> Dict[str, Any]:
    """Load test configuration from file or environment."""
    config_path = Path("test_config.json")
    
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    
    # Default configuration
    return {
        "models": {
            "default": "codebert",
            "enabled": ["codebert", "roberta-base", "t5-small"]
        },
        "thresholds": {
            "similarity": 0.8,
            "performance": 2.0,
            "accuracy": 0.85
        },
        "test_data": {
            "max_tickets": 10,
            "default_complexity": "medium"
        }
    }

def setup_test_logging():
    """Setup logging for tests."""
    import logging
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("test_automation.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
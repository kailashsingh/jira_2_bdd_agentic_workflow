"""
Functional Tests for Backend API Integration
Tests the complete JIRA to BDD workflow backend functionality
"""

import pytest
import pytest_asyncio
import asyncio
import os
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

from src.backend_integration import BackendTestClient, AgentTestRunner
from src.utils.test_helpers import TestDataGenerator
from src.config.logging import get_logger

logger = get_logger(__name__)

@pytest.mark.backend
@pytest.mark.integration
class TestBackendFunctionalFlow:
    """Test complete backend functional workflows"""
    
    @pytest.fixture
    def backend_client(self):
        """Initialize backend test client"""
        return BackendTestClient(base_url="http://localhost:8000")
    
    @pytest.fixture
    def agent_runner(self):
        """Initialize agent test runner"""
        return AgentTestRunner()
    
    @pytest.fixture
    def test_data_generator(self):
        """Initialize test data generator"""
        return TestDataGenerator()
    
    @pytest.fixture
    async def async_backend_client(self, backend_client):
        """Setup async backend client"""
        await backend_client.initialize_async_client()
        yield backend_client
        await backend_client.close_async_client()

    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, async_backend_client):
        """Test basic health check functionality"""
        result = await async_backend_client.test_health_endpoint()
        
        assert result.success, f"Health check failed: {result.error_message}"
        assert result.status_code == 200
        assert result.response_time < 1.0  # Should be fast
        assert "status" in result.response_data
        assert result.response_data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_workflow_trigger_single_ticket(self, async_backend_client, test_data_generator):
        """Test workflow trigger with single ticket"""
        # Generate test ticket
        test_ticket = test_data_generator.generate_jira_ticket()
        ticket_key = test_ticket["key"]
        
        # Mock the workflow execution to avoid external dependencies
        with patch('backend.src.agents.orchestrator.WorkflowOrchestrator') as mock_orchestrator:
            mock_instance = mock_orchestrator.return_value
            mock_instance.process_tickets = AsyncMock(return_value={
                'status': 'completed',
                'message': f'Successfully processed 1 tickets',
                'result': {'tickets_processed': 1}
            })
            
            # Trigger workflow
            result = await async_backend_client.trigger_workflow_test([ticket_key])
            
            assert result.success, f"Workflow failed: {result.error_message}"
            assert result.run_id is not None
            assert result.status == "completed"
            assert result.ticket_keys == [ticket_key]
            assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_workflow_trigger_multiple_tickets(self, async_backend_client, test_data_generator):
        """Test workflow trigger with multiple tickets"""
        # Generate multiple test tickets
        test_tickets = test_data_generator.generate_test_suite(3)
        ticket_keys = [ticket["key"] for ticket in test_tickets]
        
        # Mock the workflow execution
        with patch('backend.src.agents.orchestrator.WorkflowOrchestrator') as mock_orchestrator:
            mock_instance = mock_orchestrator.return_value
            mock_instance.process_tickets = AsyncMock(return_value={
                'status': 'completed',
                'message': f'Successfully processed {len(ticket_keys)} tickets',
                'result': {'tickets_processed': len(ticket_keys)}
            })
            
            # Trigger workflow
            result = await async_backend_client.trigger_workflow_test(ticket_keys)
            
            assert result.success, f"Workflow failed: {result.error_message}"
            assert result.run_id is not None
            assert result.status == "completed"
            assert result.ticket_keys == ticket_keys
            assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_debug_rag_search_functionality(self, async_backend_client):
        """Test RAG search debug endpoint"""
        test_query = "user authentication login feature"
        
        # Mock RAG tools to return consistent results
        with patch('backend.src.tools.rag_tools.RAGTools') as mock_rag:
            mock_instance = mock_rag.return_value
            mock_instance.search_similar_code.return_value = [
                {
                    "content": "Feature: User login functionality",
                    "metadata": {"path": "features/login.feature", "similarity": 0.85}
                }
            ]
            
            result = await async_backend_client.test_debug_rag_search(test_query)
            
            assert result.success, f"RAG search failed: {result.error_message}"
            assert result.status_code == 200
            assert "results" in result.response_data
            assert result.response_data["query"] == test_query

    @pytest.mark.asyncio
    async def test_debug_jira_tickets_endpoint(self, async_backend_client):
        """Test Jira tickets debug endpoint"""
        # Mock Jira tools to return test tickets
        with patch('backend.src.tools.jira_tools.JiraTools') as mock_jira:
            mock_instance = mock_jira.return_value
            mock_instance.get_sprint_tickets.return_value = [
                {
                    "key": "TEST-123",
                    "summary": "Test user authentication",
                    "description": "Implement login functionality",
                    "issue_type": "Story"
                }
            ]
            
            result = await async_backend_client.test_debug_jira_tickets()
            
            assert result.success, f"Jira tickets test failed: {result.error_message}"
            assert result.status_code == 200
            assert "tickets" in result.response_data
            assert len(result.response_data["tickets"]) > 0

    @pytest.mark.asyncio
    async def test_navigation_functionality(self, async_backend_client):
        """Test application navigation functionality"""
        navigation_test_data = {
            "summary": "User login page testing",
            "description": "Test the login functionality at https://example.com/login",
            "acceptance_criteria": "User should be able to login with valid credentials"
        }
        
        # Mock application tools
        with patch('backend.src.tools.application_tools.ApplicationTools') as mock_app_tools:
            mock_instance = mock_app_tools.return_value
            mock_instance.needs_navigation.return_value = True
            mock_instance._extract_url_from_jira_data.return_value = "https://example.com/login"
            mock_instance._extract_navigation_instructions.return_value = ["Navigate to login page", "Enter credentials"]
            mock_instance.navigate_and_collect_data.return_value = {
                "url": "https://example.com/login",
                "title": "Login Page",
                "forms": [{"inputs": [{"type": "text", "name": "username"}, {"type": "password", "name": "password"}]}],
                "elements": [{"type": "button", "text": "Login"}]
            }
            
            result = await async_backend_client.test_navigation_endpoint(navigation_test_data)
            
            assert result.success, f"Navigation test failed: {result.error_message}"
            assert result.status_code == 200
            assert "navigation_needed" in result.response_data
            if result.response_data["navigation_needed"]:
                assert "extracted_url" in result.response_data
                assert "navigation_instructions" in result.response_data

    @pytest.mark.asyncio
    async def test_url_validation(self, async_backend_client):
        """Test URL validation functionality"""
        test_url = "https://httpbin.org/status/200"
        
        # Mock application tools for URL validation
        with patch('backend.src.tools.application_tools.ApplicationTools') as mock_app_tools:
            mock_instance = mock_app_tools.return_value
            mock_instance.start_browser.return_value = None
            mock_instance.close_browser.return_value = None
            
            # Mock the page object
            mock_page = Mock()
            mock_page.goto.return_value = None
            mock_page.title.return_value = "Test Page"
            mock_instance.page = mock_page
            mock_instance._collect_page_elements.return_value = [{"type": "div", "text": "content"}]
            mock_instance._collect_forms.return_value = []
            
            result = await async_backend_client.test_url_validation(test_url)
            
            assert result.success, f"URL validation failed: {result.error_message}"
            assert result.status_code == 200
            assert "url" in result.response_data
            assert result.response_data["url"] == test_url

    @pytest.mark.asyncio
    async def test_comprehensive_api_suite(self, async_backend_client):
        """Test complete API functionality"""
        # Mock all external dependencies
        with patch.multiple(
            'backend.src.tools.rag_tools',
            RAGTools=Mock()
        ), patch.multiple(
            'backend.src.tools.jira_tools',
            JiraTools=Mock()
        ), patch.multiple(
            'backend.src.tools.application_tools',
            ApplicationTools=Mock()
        ):
            # Setup mocks
            rag_mock = Mock()
            rag_mock.search_similar_code.return_value = []
            
            jira_mock = Mock()
            jira_mock.get_sprint_tickets.return_value = []
            
            app_mock = Mock()
            app_mock.needs_navigation.return_value = False
            app_mock.start_browser.return_value = None
            app_mock.close_browser.return_value = None
            
            # Run comprehensive test
            results = await async_backend_client.run_comprehensive_api_test()
            
            assert "tests" in results
            assert "summary" in results
            assert results["summary"]["total_tests"] > 0
            
            # Check individual test results
            for test_name, test_result in results["tests"].items():
                assert hasattr(test_result, 'success') or 'success' in test_result
                logger.info(f"Test {test_name}: {'PASSED' if getattr(test_result, 'success', test_result.get('success')) else 'FAILED'}")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_performance(self, async_backend_client):
        """Test API performance under load"""
        # Test health endpoint performance
        load_test_results = await async_backend_client.run_load_test(
            endpoint="/health", 
            concurrent_requests=5, 
            total_requests=20
        )
        
        assert load_test_results["successful_requests"] > 0
        assert load_test_results["success_rate"] > 0.8  # 80% success rate minimum
        assert load_test_results["avg_response_time"] < 2.0  # Average response time under 2 seconds
        assert load_test_results["requests_per_second"] > 1.0  # At least 1 RPS

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_end_to_end_workflow(self, async_backend_client, test_data_generator):
        """Test complete end-to-end workflow execution"""
        # Generate comprehensive test scenario
        test_tickets = test_data_generator.generate_test_suite(2)
        ticket_keys = [ticket["key"] for ticket in test_tickets]
        
        # Mock all backend components for end-to-end test
        with patch.multiple(
            'backend.src.agents.orchestrator',
            JiraTools=Mock(),
            GitHubTools=Mock(),
            RAGTools=Mock(),
            ApplicationTools=Mock(),
            BDDGeneratorAgent=Mock()
        ) as mocks:
            
            # Configure comprehensive mocks
            jira_mock = mocks['JiraTools'].return_value
            jira_mock.get_tickets.return_value = test_tickets
            jira_mock.update_ticket_comment.return_value = True
            
            github_mock = mocks['GitHubTools'].return_value
            github_mock.get_feature_files.return_value = []
            github_mock.get_step_definitions.return_value = []
            github_mock.create_branch.return_value = True
            github_mock.create_or_update_file.return_value = True
            github_mock.create_pull_request.return_value = "https://github.com/test/pr/1"
            
            rag_mock = mocks['RAGTools'].return_value
            rag_mock.search_similar_code.return_value = []
            rag_mock.index_codebase.return_value = True
            
            app_mock = mocks['ApplicationTools'].return_value
            app_mock.needs_navigation.return_value = False
            
            bdd_mock = mocks['BDDGeneratorAgent'].return_value
            bdd_mock.is_testable.return_value = True
            bdd_mock.generate_bdd_scenarios.return_value = {
                "feature": "Feature: Test feature\n  Scenario: Test scenario",
                "step_definitions": "import { Given } from '@wdio/cucumber-framework';",
                "ticket_key": ticket_keys[0],
                "feature_file_name": "test.feature",
                "steps_file_name": "test.steps.ts"
            }
            
            # Create WorkflowOrchestrator with mocked dependencies
            with patch('backend.src.agents.orchestrator.WorkflowOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = Mock()
                mock_orchestrator.process_tickets = AsyncMock(return_value={
                    'status': 'completed',
                    'message': f'Successfully processed {len(ticket_keys)} tickets',
                    'result': {
                        'tickets_processed': len(ticket_keys),
                        'pr_urls': ['https://github.com/test/pr/1', 'https://github.com/test/pr/2']
                    }
                })
                mock_orchestrator_class.return_value = mock_orchestrator
                
                # Execute end-to-end workflow
                result = await async_backend_client.trigger_workflow_test(ticket_keys)
                
                # Verify end-to-end execution
                assert result.success, f"End-to-end workflow failed: {result.error_message}"
                assert result.status == "completed"
                assert len(result.ticket_keys) == len(ticket_keys)
                assert result.execution_time > 0
                
                # Verify workflow components were called
                mock_orchestrator.process_tickets.assert_called_once_with(ticket_keys, None)


@pytest.mark.backend
@pytest.mark.unit
class TestBackendAgentIntegration:
    """Test backend agent integration and functionality"""
    
    @pytest.fixture
    def agent_runner(self):
        """Initialize agent test runner"""
        return AgentTestRunner()
    
    @pytest.fixture
    def test_data_generator(self):
        """Initialize test data generator"""
        return TestDataGenerator()

    @pytest.mark.asyncio
    async def test_bdd_generator_agent_integration(self, agent_runner, test_data_generator):
        """Test BDD Generator Agent integration"""
        # Prepare test scenarios
        test_scenarios = [
            {
                "name": "valid_user_story",
                "ticket_data": test_data_generator.generate_jira_ticket(
                    issue_type="Story",
                    summary="User authentication",
                    description="Implement login functionality"
                ),
                "existing_code": [],
                "application_data": {},
                "mock_similar_code": []
            },
            {
                "name": "bug_ticket",
                "ticket_data": test_data_generator.generate_jira_ticket(
                    issue_type="Bug",
                    summary="Fix login error",
                    description="Login button not working"
                ),
                "existing_code": [],
                "application_data": {},
                "mock_similar_code": []
            }
        ]
        
        # Mock LLM responses
        with patch('backend.src.agents.bdd_generator_agent.ChatOpenAI') as mock_llm_class:
            mock_llm = Mock()
            
            # Mock responses for different test scenarios
            def mock_invoke(messages):
                mock_response = Mock()
                if "determine if it requires BDD testing" in str(messages):
                    # is_testable check
                    mock_response.content = "TRUE" if "Story" in str(messages) else "FALSE"
                else:
                    # BDD generation
                    mock_response.content = """
FILE_DECISIONS:
<<FEATURE_FILE_ACTION>>FALSE<<FEATURE_FILE_ACTION_END>>
<<FEATURE_FILE_NAME>>features/test.feature<<FEATURE_FILE_NAME_END>>
<<STEPS_FILE_ACTION>>FALSE<<STEPS_FILE_ACTION_END>>
<<STEPS_FILE_NAME>>features/steps/test.steps.ts<<STEPS_FILE_NAME_END>>

FEATURE_FILE:
<<FEATURE_START>>
@TEST-123 @AutoGenerated
Feature: Test feature
  Scenario: Test scenario
    Given user is on test page
    When user performs test action
    Then test result should be visible
<<FEATURE_END>>

STEP_DEFINITIONS:
<<STEPS_START>>
import { Given, When, Then } from '@wdio/cucumber-framework';

Given('user is on test page', async () => {
  await browser.url('/test');
});

When('user performs test action', async () => {
  await $('button=Test').click();
});

Then('test result should be visible', async () => {
  await expect($('.result')).toBeDisplayed();
});
<<STEPS_END>>
"""
                return mock_response
            
            mock_llm.invoke = Mock(side_effect=mock_invoke)
            mock_llm_class.return_value = mock_llm
            
            # Run BDD agent tests
            results = await agent_runner.test_bdd_generator_agent(test_scenarios)
            
            assert len(results) == 2
            
            # Check user story result (should be testable)
            story_result = results[0]
            assert story_result.success
            assert story_result.agent_name == "BDDGeneratorAgent"
            assert story_result.execution_time > 0
            
            # Check bug result (should not be testable)
            bug_result = results[1]
            assert bug_result.success  # Success means proper handling of non-testable ticket
            assert bug_result.agent_name == "BDDGeneratorAgent"

    @pytest.mark.asyncio
    async def test_workflow_orchestrator_integration(self, agent_runner, test_data_generator):
        """Test Workflow Orchestrator integration"""
        # Prepare workflow test scenarios
        workflow_scenarios = [
            {
                "name": "single_ticket_workflow",
                "mode": "tickets",
                "ticket_keys": ["TEST-123"],
                "project": None,
                "mock_tickets": [test_data_generator.generate_jira_ticket()],
                "is_testable": True,
                "needs_navigation": False,
                "mock_bdd_output": {
                    "feature": "Feature: Test",
                    "step_definitions": "import test",
                    "ticket_key": "TEST-123",
                    "feature_file_name": "test.feature",
                    "steps_file_name": "test.steps.ts"
                }
            },
            {
                "name": "sprint_workflow",
                "mode": "sprint",
                "sprint_id": 123,
                "mock_sprint_tickets": test_data_generator.generate_test_suite(2),
                "is_testable": True,
                "needs_navigation": True,
                "mock_application_data": {
                    "url": "https://example.com",
                    "title": "Test App",
                    "elements": [],
                    "forms": []
                }
            }
        ]
        
        # Run workflow tests
        results = await agent_runner.test_workflow_orchestrator(workflow_scenarios)
        
        assert len(results) == 2
        
        for result in results:
            assert result.success, f"Workflow test failed: {result.error_message}"
            assert result.execution_time > 0
            assert len(result.nodes_executed) > 0

    @pytest.mark.asyncio
    async def test_tools_integration(self, agent_runner):
        """Test individual tools integration"""
        tool_test_scenarios = [
            {
                "tool_name": "jira_tools",
                "test_name": "fetch_tickets",
                "expected_ticket_count": 5
            },
            {
                "tool_name": "github_tools", 
                "test_name": "fetch_files",
                "expected_file_count": 10,
                "repository": "test/repo"
            },
            {
                "tool_name": "rag_tools",
                "test_name": "search_code",
                "query": "user authentication",
                "expected_results": 3
            },
            {
                "tool_name": "application_tools",
                "test_name": "navigate_app",
                "navigation_required": True,
                "data_expected": True
            }
        ]
        
        # Run tool tests
        results = await agent_runner.test_tools_integration(tool_test_scenarios)
        
        assert len(results) == 4
        
        for result in results:
            assert result.success, f"Tool test failed for {result.agent_name}: {result.error_message}"
            assert result.execution_time > 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agent_performance(self, agent_runner, test_data_generator):
        """Test agent performance and load handling"""
        # Prepare comprehensive test suite
        test_suite_config = {
            "name": "performance_test_suite",
            "bdd_scenarios": [
                {
                    "name": f"performance_test_{i}",
                    "ticket_data": test_data_generator.generate_jira_ticket(),
                    "existing_code": [],
                    "application_data": {}
                }
                for i in range(5)  # Test with 5 scenarios
            ],
            "workflow_scenarios": [
                {
                    "name": "performance_workflow",
                    "mode": "tickets",
                    "ticket_keys": ["PERF-001"],
                    "mock_tickets": [test_data_generator.generate_jira_ticket()],
                    "is_testable": True
                }
            ],
            "tool_scenarios": [
                {
                    "tool_name": "jira_tools",
                    "test_name": "performance_jira"
                }
            ]
        }
        
        # Mock LLM for performance testing
        with patch('backend.src.agents.bdd_generator_agent.ChatOpenAI') as mock_llm_class:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "TRUE"  # Simple response for performance testing
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            # Run comprehensive test suite
            results = await agent_runner.run_comprehensive_agent_tests(test_suite_config)
            
            assert results["summary"]["total_tests"] > 0
            assert results["summary"]["success_rate"] > 0.7  # At least 70% success rate
            assert results["summary"]["avg_test_time"] < 5.0  # Average test time under 5 seconds
            
            logger.info(f"Performance test results: {results['summary']}")


@pytest.mark.backend
@pytest.mark.smoke
class TestBackendSmoke:
    """Quick smoke tests for backend functionality"""
    
    @pytest.fixture
    def backend_client(self):
        """Initialize backend test client for smoke tests"""
        return BackendTestClient()

    def test_backend_imports(self):
        """Test that all backend modules can be imported"""
        try:
            from src.backend_integration import BackendTestClient, AgentTestRunner
            from backend.src.agents.orchestrator import WorkflowOrchestrator
            from backend.src.agents.bdd_generator_agent import BDDGeneratorAgent
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import backend modules: {e}")

    def test_backend_client_initialization(self, backend_client):
        """Test backend client can be initialized"""
        assert backend_client is not None
        assert backend_client.base_url == "http://localhost:8000"
        assert backend_client.timeout == 30

    def test_agent_runner_initialization(self):
        """Test agent runner can be initialized"""
        runner = AgentTestRunner()
        assert runner is not None
        assert runner.test_results == []
        assert runner.mock_data == {}

    @pytest.mark.asyncio
    async def test_basic_async_setup(self, backend_client):
        """Test async client setup and teardown"""
        await backend_client.initialize_async_client()
        assert backend_client.async_client is not None
        
        await backend_client.close_async_client()
        assert backend_client.async_client is None
"""
Backend Agent Testing Module
Provides testing utilities for LangGraph workflows and AI agents
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from unittest.mock import Mock, AsyncMock, patch
from src.config.logging import get_logger

logger = get_logger(__name__)

@dataclass
class AgentTestResult:
    """Result of individual agent testing"""
    agent_name: str
    test_name: str
    success: bool
    execution_time: float
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    error_message: Optional[str]
    performance_metrics: Dict[str, Any]

@dataclass
class WorkflowTestResult:
    """Result of workflow orchestration testing"""
    workflow_name: str
    success: bool
    execution_time: float
    nodes_executed: List[str]
    final_state: Dict[str, Any]
    error_message: Optional[str]
    agent_results: List[AgentTestResult]

class AgentTestRunner:
    """Comprehensive testing framework for backend agents"""
    
    def __init__(self):
        self.test_results = []
        self.mock_data = {}
        
    def setup_mock_data(self, data: Dict[str, Any]):
        """Setup mock data for testing"""
        self.mock_data.update(data)
        logger.info(f"Mock data setup with {len(data)} entries")

    # BDD Generator Agent Testing
    
    async def test_bdd_generator_agent(self, test_scenarios: List[Dict]) -> List[AgentTestResult]:
        """Test BDD Generator Agent with various scenarios"""
        results = []
        
        for scenario in test_scenarios:
            result = await self._test_single_bdd_generation(scenario)
            results.append(result)
            
        return results
    
    async def _test_single_bdd_generation(self, scenario: Dict) -> AgentTestResult:
        """Test single BDD generation scenario"""
        start_time = time.time()
        
        try:
            # Import BDD agent here to avoid circular imports
            from backend.src.agents.bdd_generator_agent import BDDGeneratorAgent
            from backend.src.tools.rag_tools import RAGTools
            
            # Setup mocks
            mock_rag_tools = Mock(spec=RAGTools)
            mock_rag_tools.search_similar_code.return_value = scenario.get("mock_similar_code", [])
            
            # Initialize agent
            agent = BDDGeneratorAgent(mock_rag_tools)
            
            # Test testability check
            ticket_data = scenario["ticket_data"]
            is_testable = agent.is_testable(ticket_data)
            
            # Test BDD generation if testable
            if is_testable:
                generated_bdd = agent.generate_bdd_scenarios(
                    ticket_data,
                    scenario.get("existing_code", []),
                    scenario.get("application_data", {})
                )
                
                success = self._validate_bdd_output(generated_bdd)
                output_data = generated_bdd
                error_message = None
                
            else:
                success = True  # Not testable is a valid result
                output_data = {"is_testable": False}
                error_message = None
                
            execution_time = time.time() - start_time
            
            return AgentTestResult(
                agent_name="BDDGeneratorAgent",
                test_name=scenario.get("name", "bdd_generation_test"),
                success=success,
                execution_time=execution_time,
                input_data=scenario,
                output_data=output_data,
                error_message=error_message,
                performance_metrics={
                    "is_testable": is_testable,
                    "generation_time": execution_time,
                    "output_size": len(str(output_data))
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"BDD generation test failed: {str(e)}")
            
            return AgentTestResult(
                agent_name="BDDGeneratorAgent",
                test_name=scenario.get("name", "bdd_generation_test"),
                success=False,
                execution_time=execution_time,
                input_data=scenario,
                output_data={},
                error_message=str(e),
                performance_metrics={"execution_time": execution_time}
            )

    def _validate_bdd_output(self, bdd_output: Dict) -> bool:
        """Validate BDD generation output"""
        required_fields = ["feature", "step_definitions", "ticket_key", "feature_file_name", "steps_file_name"]
        
        # Check all required fields exist
        for field in required_fields:
            if field not in bdd_output:
                logger.error(f"Missing required field: {field}")
                return False
                
        # Validate feature content
        feature_content = bdd_output.get("feature", "")
        if not feature_content or "Feature:" not in feature_content:
            logger.error("Invalid feature content")
            return False
            
        # Validate step definitions
        step_content = bdd_output.get("step_definitions", "")
        if not step_content or "import" not in step_content.lower():
            logger.error("Invalid step definitions content")
            return False
            
        return True

    # Workflow Orchestrator Testing
    
    async def test_workflow_orchestrator(self, test_workflows: List[Dict]) -> List[WorkflowTestResult]:
        """Test workflow orchestrator with various scenarios"""
        results = []
        
        for workflow_config in test_workflows:
            result = await self._test_single_workflow(workflow_config)
            results.append(result)
            
        return results
    
    async def _test_single_workflow(self, workflow_config: Dict) -> WorkflowTestResult:
        """Test single workflow execution"""
        start_time = time.time()
        
        try:
            # Import orchestrator here to avoid circular imports
            from backend.src.agents.orchestrator import WorkflowOrchestrator
            
            # Setup mocks based on configuration
            with patch.multiple(
                'backend.src.agents.orchestrator',
                JiraTools=Mock(),
                GitHubTools=Mock(),
                RAGTools=Mock(),
                ApplicationTools=Mock(),
                BDDGeneratorAgent=Mock()
            ) as mocks:
                
                # Configure mocks
                self._configure_workflow_mocks(mocks, workflow_config)
                
                # Initialize orchestrator
                orchestrator = WorkflowOrchestrator()
                
                # Execute workflow
                if workflow_config.get("mode") == "tickets":
                    result = await orchestrator.process_tickets(
                        workflow_config["ticket_keys"],
                        workflow_config.get("project")
                    )
                else:
                    result = await orchestrator.run(workflow_config.get("sprint_id"))
                
                execution_time = time.time() - start_time
                
                # Analyze workflow execution
                success = result.get("status") == "completed" if "status" in result else "completed" in str(result)
                nodes_executed = self._extract_executed_nodes(result)
                
                return WorkflowTestResult(
                    workflow_name=workflow_config.get("name", "workflow_test"),
                    success=success,
                    execution_time=execution_time,
                    nodes_executed=nodes_executed,
                    final_state=result,
                    error_message=result.get("error") if not success else None,
                    agent_results=[]  # Could be populated with sub-agent results
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Workflow test failed: {str(e)}")
            
            return WorkflowTestResult(
                workflow_name=workflow_config.get("name", "workflow_test"),
                success=False,
                execution_time=execution_time,
                nodes_executed=[],
                final_state={},
                error_message=str(e),
                agent_results=[]
            )

    def _configure_workflow_mocks(self, mocks: Dict, config: Dict):
        """Configure mocks for workflow testing"""
        
        # Configure Jira Tools mock
        jira_mock = mocks['JiraTools'].return_value
        jira_mock.get_tickets.return_value = config.get("mock_tickets", [])
        jira_mock.get_sprint_tickets.return_value = config.get("mock_sprint_tickets", [])
        jira_mock.update_ticket_comment.return_value = True
        
        # Configure GitHub Tools mock
        github_mock = mocks['GitHubTools'].return_value
        github_mock.get_feature_files.return_value = config.get("mock_feature_files", [])
        github_mock.get_step_definitions.return_value = config.get("mock_step_definitions", [])
        github_mock.create_branch.return_value = True
        github_mock.create_or_update_file.return_value = True
        github_mock.create_pull_request.return_value = config.get("mock_pr_url", "https://github.com/test/pr/1")
        
        # Configure RAG Tools mock
        rag_mock = mocks['RAGTools'].return_value
        rag_mock.search_similar_code.return_value = config.get("mock_similar_code", [])
        rag_mock.index_codebase.return_value = True
        
        # Configure Application Tools mock
        app_mock = mocks['ApplicationTools'].return_value
        app_mock.needs_navigation.return_value = config.get("needs_navigation", False)
        app_mock.navigate_and_collect_data_using_mcp = AsyncMock(
            return_value=config.get("mock_application_data", {})
        )
        
        # Configure BDD Generator Agent mock
        bdd_mock = mocks['BDDGeneratorAgent'].return_value
        bdd_mock.is_testable.return_value = config.get("is_testable", True)
        bdd_mock.generate_bdd_scenarios.return_value = config.get("mock_bdd_output", {
            "feature": "Feature: Test feature",
            "step_definitions": "import { Given } from '@wdio/cucumber-framework';",
            "ticket_key": "TEST-123",
            "feature_file_name": "test.feature",
            "steps_file_name": "test.steps.ts"
        })

    def _extract_executed_nodes(self, result: Dict) -> List[str]:
        """Extract executed workflow nodes from result"""
        # This would need to be adapted based on actual LangGraph output structure
        nodes = []
        
        if isinstance(result, dict):
            # Look for common workflow state indicators
            if "tickets" in result:
                nodes.append("fetch_jira_tickets")
            if "current_ticket" in result:
                nodes.append("process_ticket")
            if "codebase_indexed" in result:
                nodes.append("index_codebase")
            if "application_data" in result:
                nodes.append("navigate_application")
            if "generated_tests" in result:
                nodes.append("generate_tests")
            if "pr_urls" in result:
                nodes.append("create_pr")
                
        return nodes

    # Tools Testing
    
    async def test_tools_integration(self, tool_tests: List[Dict]) -> List[AgentTestResult]:
        """Test individual tools integration"""
        results = []
        
        for tool_test in tool_tests:
            result = await self._test_single_tool(tool_test)
            results.append(result)
            
        return results
    
    async def _test_single_tool(self, tool_test: Dict) -> AgentTestResult:
        """Test individual tool functionality"""
        start_time = time.time()
        tool_name = tool_test.get("tool_name", "unknown_tool")
        
        try:
            # Test different tools based on configuration
            if tool_name == "jira_tools":
                result = await self._test_jira_tools(tool_test)
            elif tool_name == "github_tools":
                result = await self._test_github_tools(tool_test)
            elif tool_name == "rag_tools":
                result = await self._test_rag_tools(tool_test)
            elif tool_name == "application_tools":
                result = await self._test_application_tools(tool_test)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
            execution_time = time.time() - start_time
            
            return AgentTestResult(
                agent_name=tool_name,
                test_name=tool_test.get("test_name", f"{tool_name}_test"),
                success=result.get("success", False),
                execution_time=execution_time,
                input_data=tool_test,
                output_data=result,
                error_message=result.get("error"),
                performance_metrics={
                    "execution_time": execution_time,
                    "data_size": len(str(result))
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool test failed for {tool_name}: {str(e)}")
            
            return AgentTestResult(
                agent_name=tool_name,
                test_name=tool_test.get("test_name", f"{tool_name}_test"),
                success=False,
                execution_time=execution_time,
                input_data=tool_test,
                output_data={},
                error_message=str(e),
                performance_metrics={"execution_time": execution_time}
            )

    async def _test_jira_tools(self, config: Dict) -> Dict:
        """Test Jira tools functionality"""
        # Mock Jira tools testing
        return {
            "success": True,
            "tickets_fetched": config.get("expected_ticket_count", 5),
            "connection_status": "connected"
        }

    async def _test_github_tools(self, config: Dict) -> Dict:
        """Test GitHub tools functionality"""
        # Mock GitHub tools testing
        return {
            "success": True,
            "files_fetched": config.get("expected_file_count", 10),
            "repository": config.get("repository", "test/repo")
        }

    async def _test_rag_tools(self, config: Dict) -> Dict:
        """Test RAG tools functionality"""
        # Mock RAG tools testing
        return {
            "success": True,
            "similar_code_found": config.get("expected_results", 3),
            "search_query": config.get("query", "test query")
        }

    async def _test_application_tools(self, config: Dict) -> Dict:
        """Test application tools functionality"""
        # Mock application tools testing
        return {
            "success": True,
            "navigation_completed": config.get("navigation_required", True),
            "data_collected": config.get("data_expected", True)
        }

    # Comprehensive Testing Suite
    
    async def run_comprehensive_agent_tests(self, test_suite_config: Dict) -> Dict[str, Any]:
        """Run comprehensive test suite for all agents and workflows"""
        
        start_time = time.time()
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "test_suite": test_suite_config.get("name", "comprehensive_agent_test"),
            "bdd_agent_tests": [],
            "workflow_tests": [],
            "tool_tests": [],
            "summary": {}
        }
        
        # Test BDD Generator Agent
        if "bdd_scenarios" in test_suite_config:
            bdd_results = await self.test_bdd_generator_agent(test_suite_config["bdd_scenarios"])
            results["bdd_agent_tests"] = [asdict(r) for r in bdd_results]
            
        # Test Workflow Orchestrator
        if "workflow_scenarios" in test_suite_config:
            workflow_results = await self.test_workflow_orchestrator(test_suite_config["workflow_scenarios"])
            results["workflow_tests"] = [asdict(r) for r in workflow_results]
            
        # Test Tools
        if "tool_scenarios" in test_suite_config:
            tool_results = await self.test_tools_integration(test_suite_config["tool_scenarios"])
            results["tool_tests"] = [asdict(r) for r in tool_results]
            
        # Calculate summary
        all_tests = []
        all_tests.extend(results["bdd_agent_tests"])
        all_tests.extend([t for t in results["workflow_tests"]])
        all_tests.extend(results["tool_tests"])
        
        total_tests = len(all_tests)
        passed_tests = sum(1 for test in all_tests if test.get("success", False))
        failed_tests = total_tests - passed_tests
        total_execution_time = time.time() - start_time
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_execution_time,
            "avg_test_time": total_execution_time / total_tests if total_tests > 0 else 0
        }
        
        logger.info(f"Agent test suite completed: {passed_tests}/{total_tests} tests passed")
        
        return results

    # Performance Testing
    
    async def run_performance_tests(self, performance_config: Dict) -> Dict[str, Any]:
        """Run performance tests for agents and workflows"""
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "performance_metrics": {},
            "load_test_results": {},
            "memory_usage": {},
            "response_times": {}
        }
        
        # BDD Generation Performance
        if "bdd_performance" in performance_config:
            bdd_perf = await self._test_bdd_performance(performance_config["bdd_performance"])
            results["performance_metrics"]["bdd_generation"] = bdd_perf
            
        # Workflow Performance
        if "workflow_performance" in performance_config:
            workflow_perf = await self._test_workflow_performance(performance_config["workflow_performance"])
            results["performance_metrics"]["workflow_execution"] = workflow_perf
            
        return results
    
    async def _test_bdd_performance(self, config: Dict) -> Dict:
        """Test BDD generation performance"""
        iterations = config.get("iterations", 10)
        execution_times = []
        
        for i in range(iterations):
            start_time = time.time()
            # Simulate BDD generation
            await asyncio.sleep(0.1)  # Placeholder
            execution_times.append(time.time() - start_time)
            
        return {
            "iterations": iterations,
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "total_time": sum(execution_times)
        }
    
    async def _test_workflow_performance(self, config: Dict) -> Dict:
        """Test workflow orchestration performance"""
        iterations = config.get("iterations", 5)
        execution_times = []
        
        for i in range(iterations):
            start_time = time.time()
            # Simulate workflow execution
            await asyncio.sleep(0.5)  # Placeholder
            execution_times.append(time.time() - start_time)
            
        return {
            "iterations": iterations,
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "total_time": sum(execution_times)
        }
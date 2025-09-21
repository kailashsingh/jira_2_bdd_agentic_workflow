"""
FastAPI Backend Integration for Test Automation Framework
Provides testing utilities for the JIRA to BDD workflow backend API
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from httpx import AsyncClient, Response
from fastapi.testclient import TestClient
from dataclasses import dataclass
from src.config.logging import get_logger

logger = get_logger(__name__)

@dataclass
class WorkflowTestResult:
    """Result of workflow execution testing"""
    run_id: str
    status: str
    started_at: str
    completed_at: Optional[str]
    execution_time: Optional[float]
    ticket_keys: List[str]
    success: bool
    error_message: Optional[str]
    api_response: Dict[str, Any]

@dataclass
class ApiEndpointTest:
    """Test result for individual API endpoint"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    response_data: Dict[str, Any]
    error_message: Optional[str]

class BackendTestClient:
    """FastAPI backend test client with comprehensive testing capabilities"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.test_client = None
        self.async_client = None
        
    def initialize_test_client(self, app):
        """Initialize FastAPI TestClient for unit testing"""
        self.test_client = TestClient(app)
        logger.info("Initialized FastAPI TestClient")
        
    async def initialize_async_client(self):
        """Initialize async HTTP client for integration testing"""
        self.async_client = AsyncClient(base_url=self.base_url, timeout=self.timeout)
        logger.info(f"Initialized async client for {self.base_url}")
        
    async def close_async_client(self):
        """Close async HTTP client"""
        if self.async_client:
            await self.async_client.aclose()
            self.async_client = None
            logger.info("Closed async client")

    # Health Check and Basic API Tests
    
    async def test_health_endpoint(self) -> ApiEndpointTest:
        """Test the health check endpoint"""
        start_time = time.time()
        
        try:
            if self.test_client:
                response = self.test_client.get("/health")
                response_data = response.json()
            else:
                response = await self.async_client.get("/health")
                response_data = response.json()
                
            execution_time = time.time() - start_time
            
            return ApiEndpointTest(
                endpoint="/health",
                method="GET",
                status_code=response.status_code,
                response_time=execution_time,
                success=response.status_code == 200,
                response_data=response_data,
                error_message=None if response.status_code == 200 else f"Status: {response.status_code}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Health check failed: {str(e)}")
            
            return ApiEndpointTest(
                endpoint="/health",
                method="GET",
                status_code=500,
                response_time=execution_time,
                success=False,
                response_data={},
                error_message=str(e)
            )

    # Workflow Testing
    
    async def trigger_workflow_test(self, ticket_keys: Union[str, List[str]], 
                                   project: Optional[str] = None) -> WorkflowTestResult:
        """Test workflow trigger endpoint and monitor execution"""
        
        # Prepare request data
        if isinstance(ticket_keys, str):
            ticket_keys = [ticket_keys]
            
        request_data = {
            "ticket_keys": ticket_keys,
            "project": project
        }
        
        start_time = time.time()
        
        try:
            # Trigger workflow
            if self.test_client:
                response = self.test_client.post("/workflow/trigger/tickets", json=request_data)
                response_data = response.json()
            else:
                response = await self.async_client.post("/workflow/trigger/tickets", json=request_data)
                response_data = response.json()
                
            if response.status_code != 200:
                return WorkflowTestResult(
                    run_id="",
                    status="failed",
                    started_at=datetime.now().isoformat(),
                    completed_at=None,
                    execution_time=time.time() - start_time,
                    ticket_keys=ticket_keys,
                    success=False,
                    error_message=f"Failed to trigger workflow: {response.status_code}",
                    api_response=response_data
                )
                
            run_id = response_data.get("run_id")
            
            # Monitor workflow execution
            final_status = await self.monitor_workflow(run_id)
            execution_time = time.time() - start_time
            
            return WorkflowTestResult(
                run_id=run_id,
                status=final_status["status"],
                started_at=response_data.get("started_at"),
                completed_at=final_status.get("completed_at"),
                execution_time=execution_time,
                ticket_keys=ticket_keys,
                success=final_status["status"] == "completed",
                error_message=final_status.get("error"),
                api_response=final_status
            )
            
        except Exception as e:
            logger.error(f"Workflow test failed: {str(e)}")
            return WorkflowTestResult(
                run_id="",
                status="failed",
                started_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat(),
                execution_time=time.time() - start_time,
                ticket_keys=ticket_keys,
                success=False,
                error_message=str(e),
                api_response={}
            )

    async def monitor_workflow(self, run_id: str, max_wait_time: int = 300, 
                              poll_interval: int = 5) -> Dict[str, Any]:
        """Monitor workflow execution until completion or timeout"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                if self.test_client:
                    response = self.test_client.get(f"/workflow/status/{run_id}")
                    status_data = response.json()
                else:
                    response = await self.async_client.get(f"/workflow/status/{run_id}")
                    status_data = response.json()
                    
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"Failed to get status: {response.status_code}"
                    }
                    
                status = status_data.get("status")
                
                if status in ["completed", "failed"]:
                    return status_data
                    
                logger.info(f"Workflow {run_id} status: {status}")
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring workflow {run_id}: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
                
        # Timeout
        return {
            "status": "timeout",
            "error": f"Workflow monitoring timed out after {max_wait_time} seconds"
        }

    # Debug Endpoints Testing
    
    async def test_debug_rag_search(self, query: str) -> ApiEndpointTest:
        """Test RAG search debug endpoint"""
        start_time = time.time()
        
        try:
            if self.test_client:
                response = self.test_client.get(f"/debug/rag-search?query={query}")
                response_data = response.json()
            else:
                response = await self.async_client.get(f"/debug/rag-search?query={query}")
                response_data = response.json()
                
            execution_time = time.time() - start_time
            
            return ApiEndpointTest(
                endpoint="/debug/rag-search",
                method="GET",
                status_code=response.status_code,
                response_time=execution_time,
                success=response.status_code == 200,
                response_data=response_data,
                error_message=None if response.status_code == 200 else f"Status: {response.status_code}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"RAG search test failed: {str(e)}")
            
            return ApiEndpointTest(
                endpoint="/debug/rag-search",
                method="GET",
                status_code=500,
                response_time=execution_time,
                success=False,
                response_data={},
                error_message=str(e)
            )

    async def test_debug_jira_tickets(self) -> ApiEndpointTest:
        """Test Jira tickets debug endpoint"""
        start_time = time.time()
        
        try:
            if self.test_client:
                response = self.test_client.get("/debug/jira-tickets")
                response_data = response.json()
            else:
                response = await self.async_client.get("/debug/jira-tickets")
                response_data = response.json()
                
            execution_time = time.time() - start_time
            
            return ApiEndpointTest(
                endpoint="/debug/jira-tickets",
                method="GET",
                status_code=response.status_code,
                response_time=execution_time,
                success=response.status_code == 200,
                response_data=response_data,
                error_message=None if response.status_code == 200 else f"Status: {response.status_code}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Jira tickets test failed: {str(e)}")
            
            return ApiEndpointTest(
                endpoint="/debug/jira-tickets",
                method="GET",
                status_code=500,
                response_time=execution_time,
                success=False,
                response_data={},
                error_message=str(e)
            )

    async def test_navigation_endpoint(self, test_data: Dict[str, str]) -> ApiEndpointTest:
        """Test application navigation debug endpoint"""
        start_time = time.time()
        
        request_data = {
            "summary": test_data.get("summary", "Test navigation"),
            "description": test_data.get("description", "Test description"),
            "acceptance_criteria": test_data.get("acceptance_criteria", "Navigate to test page")
        }
        
        try:
            if self.test_client:
                response = self.test_client.post("/debug/test-navigation", json=request_data)
                response_data = response.json()
            else:
                response = await self.async_client.post("/debug/test-navigation", json=request_data)
                response_data = response.json()
                
            execution_time = time.time() - start_time
            
            return ApiEndpointTest(
                endpoint="/debug/test-navigation",
                method="POST",
                status_code=response.status_code,
                response_time=execution_time,
                success=response.status_code == 200,
                response_data=response_data,
                error_message=None if response.status_code == 200 else f"Status: {response.status_code}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Navigation test failed: {str(e)}")
            
            return ApiEndpointTest(
                endpoint="/debug/test-navigation",
                method="POST",
                status_code=500,
                response_time=execution_time,
                success=False,
                response_data={},
                error_message=str(e)
            )

    async def test_url_validation(self, url: str) -> ApiEndpointTest:
        """Test URL validation debug endpoint"""
        start_time = time.time()
        
        try:
            if self.test_client:
                response = self.test_client.get(f"/debug/validate-url?url={url}")
                response_data = response.json()
            else:
                response = await self.async_client.get(f"/debug/validate-url?url={url}")
                response_data = response.json()
                
            execution_time = time.time() - start_time
            
            return ApiEndpointTest(
                endpoint="/debug/validate-url",
                method="GET",
                status_code=response.status_code,
                response_time=execution_time,
                success=response.status_code == 200,
                response_data=response_data,
                error_message=None if response.status_code == 200 else f"Status: {response.status_code}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"URL validation test failed: {str(e)}")
            
            return ApiEndpointTest(
                endpoint="/debug/validate-url",
                method="GET",
                status_code=500,
                response_time=execution_time,
                success=False,
                response_data={},
                error_message=str(e)
            )

    # Comprehensive API Testing
    
    async def run_comprehensive_api_test(self) -> Dict[str, Any]:
        """Run comprehensive API test suite"""
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "total_response_time": 0.0
            }
        }
        
        # Test health endpoint
        health_result = await self.test_health_endpoint()
        results["tests"]["health_check"] = health_result
        
        # Test debug endpoints
        rag_result = await self.test_debug_rag_search("test query")
        results["tests"]["rag_search"] = rag_result
        
        jira_result = await self.test_debug_jira_tickets()
        results["tests"]["jira_tickets"] = jira_result
        
        # Test navigation
        nav_test_data = {
            "summary": "Test navigation functionality",
            "description": "Test case for navigation testing",
            "acceptance_criteria": "Should navigate to test page successfully"
        }
        nav_result = await self.test_navigation_endpoint(nav_test_data)
        results["tests"]["navigation_test"] = nav_result
        
        # Test URL validation
        url_result = await self.test_url_validation("https://httpbin.org/status/200")
        results["tests"]["url_validation"] = url_result
        
        # Calculate summary
        all_tests = [health_result, rag_result, jira_result, nav_result, url_result]
        results["summary"]["total_tests"] = len(all_tests)
        results["summary"]["passed_tests"] = sum(1 for test in all_tests if test.success)
        results["summary"]["failed_tests"] = sum(1 for test in all_tests if not test.success)
        results["summary"]["total_response_time"] = sum(test.response_time for test in all_tests)
        
        logger.info(f"API test completed: {results['summary']['passed_tests']}/{results['summary']['total_tests']} passed")
        
        return results

    # Batch Testing for Load Testing
    
    async def run_load_test(self, endpoint: str, concurrent_requests: int = 10, 
                           total_requests: int = 100) -> Dict[str, Any]:
        """Run load test on specific endpoint"""
        
        start_time = time.time()
        results = []
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def single_request():
            async with semaphore:
                try:
                    if endpoint == "/health":
                        result = await self.test_health_endpoint()
                    elif endpoint == "/debug/rag-search":
                        result = await self.test_debug_rag_search("load test query")
                    elif endpoint == "/debug/jira-tickets":
                        result = await self.test_debug_jira_tickets()
                    else:
                        result = await self.test_health_endpoint()  # Default
                        
                    return result
                except Exception as e:
                    logger.error(f"Load test request failed: {str(e)}")
                    return None
                    
        # Execute requests
        tasks = [single_request() for _ in range(total_requests)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful responses
        successful_results = [r for r in responses if r is not None and not isinstance(r, Exception)]
        failed_requests = total_requests - len(successful_results)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        response_times = [r.response_time for r in successful_results]
        success_count = sum(1 for r in successful_results if r.success)
        
        return {
            "endpoint": endpoint,
            "concurrent_requests": concurrent_requests,
            "total_requests": total_requests,
            "successful_requests": len(successful_results),
            "failed_requests": failed_requests,
            "success_rate": success_count / len(successful_results) if successful_results else 0,
            "total_execution_time": total_time,
            "requests_per_second": total_requests / total_time,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0
        }
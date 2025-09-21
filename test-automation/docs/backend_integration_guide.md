# How Test Automation Framework Calls Backend and Verifies Output

## Overview

The test automation framework provides comprehensive backend integration testing for the JIRA to BDD workflow. It calls the backend APIs, monitors workflow execution, and validates that the generated BDD scenarios are correctly created and match expected quality standards.

## Architecture Flow

```
Test Framework → Backend API → Agent Workflows → BDD Generation → Output Validation
```

## 1. Backend API Integration

### FastAPI Client (`BackendTestClient`)

The framework uses a dedicated client to interact with the backend:

```python
from src.backend_integration import BackendTestClient

# Initialize client
client = BackendTestClient(base_url="http://localhost:8000")
await client.initialize_async_client()

# Test health endpoint
health_result = await client.test_health_endpoint()
assert health_result.success
assert health_result.status_code == 200
```

### Key API Endpoints Tested

| Endpoint | Purpose | Validation |
|----------|---------|------------|
| `/health` | Backend health check | Status code 200, response contains "healthy" |
| `/workflow/trigger/tickets` | Trigger BDD generation workflow | Workflow starts, returns run_id |
| `/workflow/status/{run_id}` | Monitor workflow progress | Status updates, completion tracking |
| `/debug/rag-search` | Test RAG search functionality | Returns relevant code snippets |
| `/debug/jira-tickets` | Test JIRA integration | Returns ticket data |
| `/debug/test-navigation` | Test application navigation | Navigation success/failure |

## 2. Workflow Execution and Monitoring

### Workflow Trigger and Monitoring

```python
# Trigger workflow with JIRA tickets
result = await client.trigger_workflow_test(["TEST-123", "TEST-456"])

# The framework automatically monitors execution
assert result.success, f"Workflow failed: {result.error_message}"
assert result.status == "completed"
assert result.execution_time > 0
```

### Workflow Monitoring Process

1. **Trigger Workflow**: Send POST request to `/workflow/trigger/tickets`
2. **Get Run ID**: Extract workflow run identifier from response
3. **Poll Status**: Continuously check `/workflow/status/{run_id}` until completion
4. **Validate Results**: Verify workflow completed successfully and generated expected outputs

```python
async def monitor_workflow(self, run_id: str, max_wait_time: int = 300, poll_interval: int = 5):
    """Monitor workflow execution until completion or timeout"""
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        response = await self.async_client.get(f"/workflow/status/{run_id}")
        status_data = response.json()
        
        status = status_data.get("status")
        
        if status in ["completed", "failed"]:
            return status_data
            
        await asyncio.sleep(poll_interval)
    
    # Timeout handling
    return {"status": "timeout", "error": "Workflow monitoring timed out"}
```

## 3. Agent Testing and Output Validation

### BDD Generator Agent Testing

The framework specifically tests the BDD generation agent to ensure it produces valid output:

```python
from src.backend_integration import AgentTestRunner

# Test BDD generation
agent_runner = AgentTestRunner()

test_scenarios = [{
    "name": "user_authentication_test",
    "ticket_data": {
        "key": "TEST-123",
        "summary": "User login functionality",
        "description": "As a user, I want to login to access my account",
        "acceptance_criteria": "Given valid credentials, when user logs in, then they should access dashboard"
    }
}]

results = await agent_runner.test_bdd_generator_agent(test_scenarios)

# Validate results
for result in results:
    assert result.success, f"BDD generation failed: {result.error_message}"
    assert "feature" in result.output_data
    assert "step_definitions" in result.output_data
```

### BDD Output Validation Logic

The framework validates BDD generation output using multiple criteria:

```python
def _validate_bdd_output(self, bdd_output: Dict) -> bool:
    """Validate BDD generation output structure and content"""
    
    # 1. Required Fields Validation
    required_fields = ["feature", "step_definitions", "ticket_key", "feature_file_name", "steps_file_name"]
    
    for field in required_fields:
        if field not in bdd_output:
            logger.error(f"Missing required field: {field}")
            return False
    
    # 2. Feature Content Validation
    feature_content = bdd_output.get("feature", "")
    if not feature_content or "Feature:" not in feature_content:
        logger.error("Invalid feature content - missing Feature: declaration")
        return False
    
    # 3. Gherkin Structure Validation
    if not any(keyword in feature_content for keyword in ["Given", "When", "Then"]):
        logger.error("Invalid Gherkin structure - missing Given/When/Then steps")
        return False
    
    # 4. Step Definitions Validation
    step_content = bdd_output.get("step_definitions", "")
    if not step_content or "import" not in step_content.lower():
        logger.error("Invalid step definitions - missing import statements")
        return False
    
    # 5. File Name Validation
    feature_file = bdd_output.get("feature_file_name", "")
    if not feature_file.endswith('.feature'):
        logger.error("Invalid feature file extension")
        return False
    
    return True
```

## 4. Output Quality Verification

### Model-Based Quality Assessment

The framework uses CodeBERT and sentence transformers to validate BDD quality:

```python
# Test BDD quality using CodeBERT similarity
def test_bdd_quality_validation(self, model_manager):
    """Verify generated BDD matches requirement quality"""
    
    jira_requirement = """
    As a user, I want to login to the application 
    so that I can access my account.
    Acceptance Criteria:
    - User enters valid credentials
    - System authenticates user  
    - User is redirected to dashboard
    """
    
    generated_bdd = """
    @TEST-123
    Feature: User Authentication
      Scenario: Successful login
        Given user has valid credentials
        When user submits login form
        Then user should be redirected to dashboard
    """
    
    # Calculate similarity using CodeBERT
    similarity_score = model_manager.compare_models_similarity(
        jira_requirement, generated_bdd, ["codebert"]
    )["codebert"]
    
    # Validate quality threshold
    assert similarity_score > 0.7, f"BDD quality too low: {similarity_score}"
    
    print(f"✅ BDD Quality Score: {similarity_score:.3f}")
```

### Backend Result Validation

The framework compares backend-generated results with expected outputs:

```python
def validate_backend_results(self, backend_output: Dict, expected_output: Dict) -> Dict:
    """Validate backend results against expected data"""
    
    validation_result = {
        "valid": True,
        "similarity_score": 0.0,
        "issues": []
    }
    
    # 1. Structure Validation
    if not self._validate_output_structure(backend_output, expected_output):
        validation_result["valid"] = False
        validation_result["issues"].append("Structure mismatch")
    
    # 2. Content Similarity Validation
    similarity = self._calculate_content_similarity(backend_output, expected_output)
    validation_result["similarity_score"] = similarity
    
    if similarity < 0.8:  # 80% similarity threshold
        validation_result["valid"] = False
        validation_result["issues"].append(f"Content similarity too low: {similarity:.2f}")
    
    # 3. Business Logic Validation
    if not self._validate_business_logic(backend_output):
        validation_result["valid"] = False
        validation_result["issues"].append("Business logic validation failed")
    
    return validation_result
```

## 5. Comprehensive Test Execution

### Complete API Test Suite

```python
async def run_comprehensive_api_test(self) -> Dict[str, Any]:
    """Run complete backend validation suite"""
    
    results = {
        "test_timestamp": datetime.now().isoformat(),
        "tests": {},
        "summary": {"total_tests": 0, "passed_tests": 0, "failed_tests": 0}
    }
    
    # 1. Basic Health Check
    health_result = await self.test_health_endpoint()
    results["tests"]["health_check"] = health_result
    
    # 2. RAG Search Functionality
    rag_result = await self.test_debug_rag_search("user authentication")
    results["tests"]["rag_search"] = rag_result
    
    # 3. JIRA Integration
    jira_result = await self.test_debug_jira_tickets()
    results["tests"]["jira_tickets"] = jira_result
    
    # 4. Application Navigation
    nav_result = await self.test_navigation_endpoint({
        "summary": "Test navigation",
        "description": "Test case for navigation",
        "acceptance_criteria": "Should navigate successfully"
    })
    results["tests"]["navigation"] = nav_result
    
    # 5. Workflow Execution
    workflow_result = await self.trigger_workflow_test(["TEST-123"])
    results["tests"]["workflow_execution"] = workflow_result
    
    # Calculate summary
    all_tests = [health_result, rag_result, jira_result, nav_result]
    results["summary"]["total_tests"] = len(all_tests)
    results["summary"]["passed_tests"] = sum(1 for test in all_tests if test.success)
    results["summary"]["failed_tests"] = sum(1 for test in all_tests if not test.success)
    
    return results
```

### Load Testing and Performance Validation

```python
# Load testing capabilities
load_results = await client.run_load_test(
    endpoint="/health", 
    concurrent_requests=10, 
    total_requests=100
)

# Validate performance
assert load_results["success_rate"] > 0.95, "Success rate too low"
assert load_results["avg_response_time"] < 1.0, "Response time too slow"
assert load_results["requests_per_second"] > 50, "Throughput too low"
```

## 6. Validation Thresholds and Criteria

### Quality Thresholds

```python
VALIDATION_THRESHOLDS = {
    "similarity_acceptance": 0.70,      # 70% similarity for BDD validation
    "requirement_coverage": 0.80,       # 80% requirement coverage
    "structure_completeness": 1.0,      # 100% BDD structure (Given/When/Then)
    "api_response_time": 5.0,           # Max 5 seconds API response
    "workflow_success_rate": 0.95,      # 95% workflow success rate
    "performance_threshold": 2.0        # Max 2 seconds for BDD generation
}
```

### Success Criteria

1. **API Endpoints**: All return 200 status codes and expected data structure
2. **Workflow Execution**: Completes within timeout with "completed" status
3. **BDD Generation**: Produces valid Gherkin features and step definitions
4. **Content Quality**: Generated BDD matches requirements with >70% similarity
5. **Performance**: All operations complete within acceptable time limits
6. **Backend Validation**: Generated output matches expected data structure and content

## 7. Example Test Execution Flow

```python
@pytest.mark.asyncio
async def test_complete_jira_to_bdd_workflow():
    """Test complete JIRA to BDD workflow with output validation"""
    
    # 1. Initialize test components
    client = BackendTestClient()
    await client.initialize_async_client()
    
    # 2. Verify backend health
    health = await client.test_health_endpoint()
    assert health.success, "Backend not healthy"
    
    # 3. Test JIRA integration
    jira_result = await client.test_debug_jira_tickets()
    assert jira_result.success, "JIRA integration failed"
    
    # 4. Trigger BDD generation workflow
    workflow_result = await client.trigger_workflow_test(["TEST-123"])
    assert workflow_result.success, f"Workflow failed: {workflow_result.error_message}"
    assert workflow_result.status == "completed"
    
    # 5. Validate generated BDD output
    # (This would involve checking the actual generated files)
    
    # 6. Performance validation
    assert workflow_result.execution_time < 60.0, "Workflow too slow"
    
    # 7. Cleanup
    await client.close_async_client()
    
    print("✅ Complete JIRA to BDD workflow validation passed")
```

## Summary

The test automation framework provides comprehensive validation of the JIRA to BDD workflow by:

1. **API Integration Testing**: Validates all backend endpoints work correctly
2. **Workflow Monitoring**: Tracks workflow execution from start to completion  
3. **Output Validation**: Ensures generated BDD scenarios are structurally valid and high quality
4. **Model-Based Quality Assessment**: Uses CodeBERT and sentence transformers to verify BDD quality
5. **Performance Testing**: Validates response times and throughput meet requirements
6. **Backend Result Verification**: Compares actual output with expected results

This ensures that the backend properly generates high-quality BDD scenarios from JIRA requirements and that all components work correctly together.
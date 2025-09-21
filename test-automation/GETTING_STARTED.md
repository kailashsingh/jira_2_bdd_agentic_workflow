# Getting Started with Test Automation Framework

## Prerequisites

- Python 3.9+
- Git
- 2GB+ available RAM
- Internet connection for model downloads

## Quick Setup

### 1. Navigate to Test Automation Directory

```powershell
cd test-automation
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Set Environment Variables (Optional for Basic Tests)

```powershell
# For integration tests only
$env:JIRA_URL = "https://your-jira-instance.com"
$env:JIRA_EMAIL = "your-email@company.com"
$env:JIRA_API_TOKEN = "your-api-token"
$env:GITHUB_TOKEN = "your-github-token"
$env:GITHUB_REPO = "owner/repository"
$env:OPENAI_API_KEY = "your-openai-key"
```

### 4. Run Basic Tests

```powershell
# Smoke tests (fast, basic validation)
pytest -m smoke -v

# Model tests (medium speed)
pytest -m model -v

# All tests except slow ones
pytest -m "not slow" -v
```

## Your First Backend Integration Test

```python
# test_my_first_backend.py
import pytest
from src.backend_integration import BackendTestClient

@pytest.mark.asyncio
async def test_backend_health():
    # Initialize client
    client = BackendTestClient()
    await client.initialize_async_client()
    
    try:
        # Test health endpoint
        result = await client.test_health_endpoint()
        
        # Verify results
        assert result.success, f"Health check failed: {result.error_message}"
        assert result.status_code == 200
        assert "status" in result.response_data
        
        print(f"✅ Backend health check passed in {result.response_time:.2f}s")
        
    finally:
        await client.close_async_client()
```

Run it:
```powershell
pytest test_my_first_backend.py -v
```

## Your First Workflow Test

```python
# test_workflow_integration.py
import pytest
from src.backend_integration import AgentTestRunner
from src.utils.test_helpers import TestDataGenerator

@pytest.mark.asyncio
async def test_bdd_generation():
    # Initialize components
    runner = AgentTestRunner()
    data_generator = TestDataGenerator()
    
    # Create test scenario
    test_scenario = {
        "name": "user_authentication_test",
        "ticket_data": data_generator.generate_jira_ticket(
            summary="User login functionality",
            description="Implement secure user authentication",
            issue_type="Story"
        )
    }
    
    # Test BDD generation
    results = await runner.test_bdd_generator_agent([test_scenario])
    
    # Verify results
    assert len(results) == 1
    result = results[0]
    assert result.success, f"BDD generation failed: {result.error_message}"
    
    print(f"✅ BDD generation test passed in {result.execution_time:.2f}s")
```

## Your First Model Comparison

```python
# test_my_first_comparison.py
from src.models.model_manager import ModelManager
from src.utils.test_helpers import TestDataGenerator

def test_compare_models():
    # Initialize
    model_manager = ModelManager()
    data_generator = TestDataGenerator()
    
    # Generate test data
    test_tickets = data_generator.generate_test_suite(3)
    
    # Compare models
    models = ["codebert", "sentence-transformer"]
    results = model_manager.compare_models(test_tickets, models)
    
    # Print results
    for model, stats in results["model_comparisons"].items():
        print(f"{model}: Quality={stats['avg_quality_score']:.2f}")
    
    assert len(results["model_comparisons"]) == 2
```

Run it:
```powershell
pytest test_my_first_comparison.py -v
```

## Choosing Your First Model

### For Beginners (Fast and Reliable)
```python
model_name = "sentence-transformer"  # Fastest, good for similarity
```

### For Code-Heavy Projects
```python
model_name = "codebert"  # Best for code understanding
```

### For High-Quality Output
```python
model_name = "graphcodebert"  # Highest quality (slower)
```

## Common Commands

```powershell
# Run specific test file
pytest tests/models/test_model_comparison.py -v

# Run backend tests
pytest tests/backend/ -v

# Run backend integration tests
pytest -m "backend and integration" -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run performance benchmarks
pytest -m performance -v

# Run parallel tests (faster)
pytest -n auto

# Run and stop on first failure
pytest --maxfail=1

# Run specific backend test class
pytest tests/backend/test_functional_integration.py::TestBackendFunctionalFlow -v
```

## Troubleshooting

### Issue: Model Download Fails
```powershell
# Test internet connection and model availability
python -c "from transformers import AutoModel; print(AutoModel.from_pretrained('microsoft/codebert-base', local_files_only=False))"
```

### Issue: Out of Memory
```powershell
# Run with smaller test set
pytest -k "not large" -v
```

### Issue: Tests Too Slow
```powershell
# Skip slow tests
pytest -m "not slow and not performance" -v
```

### Issue: Import Errors
```powershell
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## Next Steps

1. **Explore the full README.md** for comprehensive documentation
2. **Run benchmarks** to find the best model for your use case
3. **Add custom test data** specific to your JIRA tickets
4. **Integrate with your CI/CD** pipeline
5. **Optimize for your deployment** environment

## Getting Help

- Check `README.md` for detailed documentation
- Look at existing test files in `tests/` for examples
- Review error messages and stack traces
- Enable debug logging: `pytest -v -s --log-cli-level=DEBUG`
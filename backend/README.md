# Jira to BDD Agent Backend

An intelligent agentic workflow system that automatically generates BDD (Behavior-Driven Development) test scenarios from Jira tickets and creates pull requests with the generated tests.

## Overview

This backend service provides a FastAPI-based REST API that orchestrates an automated workflow to:
1. Fetch Jira tickets from sprints
2. Analyze tickets to determine if they require BDD testing
3. Generate BDD scenarios and step definitions using AI
4. Create GitHub pull requests with the generated tests
5. Update Jira tickets with progress information

## Architecture

The system uses LangGraph for workflow orchestration and integrates with:
- **Jira API** - For fetching tickets and updating comments
- **GitHub API** - For creating branches, files, and pull requests
- **OpenAI API** - For AI-powered BDD generation
- **ChromaDB** - For RAG (Retrieval-Augmented Generation) with existing codebase
- **Hugging Face API** - For embeddings (optional, falls back to local)

## Available APIs

### 1. Trigger Workflow
**POST** `/workflow/trigger`

Triggers the BDD generation workflow for a specific sprint or all active sprints.

**Request Body:**
```json
{
  "sprint_id": 123,  // Optional: specific sprint ID
  "jira_keys": ["PROJ-123", "PROJ-456"]  // Optional: specific ticket keys
}
```

**Response:**
```json
{
  "status": "running",
  "run_id": "run_20241201_143022",
  "started_at": "2024-12-01T14:30:22.123456",
  "message": "Workflow triggered successfully"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/workflow/trigger" \
  -H "Content-Type: application/json" \
  -d '{"sprint_id": 123}'
```

### 2. Get Workflow Status
**GET** `/workflow/status/{run_id}`

Retrieves the current status of a workflow run.

**Response:**
```json
{
  "status": "completed",
  "started_at": "2024-12-01T14:30:22.123456",
  "completed_at": "2024-12-01T14:35:45.789012",
  "sprint_id": 123,
  "result": {
    "completed": true,
    "pr_urls": ["https://github.com/org/repo/pull/123"]
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/workflow/status/run_20241201_143022"
```

### 3. Get All Workflow Runs
**GET** `/workflow/runs`

Retrieves all workflow runs and their statuses.

**Response:**
```json
[
  {
    "status": "completed",
    "started_at": "2024-12-01T14:30:22.123456",
    "completed_at": "2024-12-01T14:35:45.789012",
    "sprint_id": 123
  }
]
```

**Example:**
```bash
curl "http://localhost:8000/workflow/runs"
```

### 4. Health Check
**GET** `/health`

Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "Jira to BDD Agent"
}
```

**Example:**
```bash
curl "http://localhost:8000/health"
```

### 5. Debug RAG Search
**GET** `/debug/rag-search?query={query}`

Debug endpoint to test RAG search functionality.

**Parameters:**
- `query` (string): Search query for similar code

**Response:**
```json
{
  "query": "user login authentication",
  "results": [
    {
      "content": "Feature: User Authentication\nGiven a user is on the login page...",
      "metadata": {
        "type": "feature",
        "path": "src/features/login.feature",
        "name": "login.feature"
      }
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:8000/debug/rag-search?query=user%20login"
```

### 6. Debug Jira Tickets
**GET** `/debug/jira-tickets`

Debug endpoint to test Jira connection and fetch tickets.

**Response:**
```json
{
  "tickets": [
    {
      "key": "PROJ-123",
      "summary": "Implement user login functionality",
      "description": "As a user, I want to be able to log in...",
      "acceptance_criteria": "Given valid credentials, user should be able to log in",
      "issue_type": "Story",
      "status": "To Do"
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:8000/debug/jira-tickets"
```

## Workflow Flow Diagram

```mermaid
graph TD
    A[Start Workflow] --> B[Fetch Jira Tickets]
    B --> C[Index Existing Codebase]
    C --> D[Process Current Ticket]
    D --> E{Is Ticket Testable?}
    E -->|No| F[Move to Next Ticket]
    E -->|Yes| G[Search Similar Code via RAG]
    G --> H[Generate BDD Scenarios]
    H --> I[Create GitHub Branch]
    I --> J[Create Feature File]
    J --> K[Create Step Definitions]
    K --> L[Create Pull Request]
    L --> M[Update Jira Comment]
    M --> F
    F --> N{More Tickets?}
    N -->|Yes| D
    N -->|No| O[Workflow Complete]
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style E fill:#fff3e0
    style N fill:#fff3e0
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Access to Jira instance with API token
- GitHub repository with appropriate permissions
- OpenAI API key
- (Optional) Hugging Face API key for embeddings

### 1. Environment Configuration

Create a `.env` file in the backend directory with the following variables:

```env
# Jira Configuration
JIRA_URL=https://your-company.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_API_TOKEN=your-jira-api-token

# GitHub Configuration
GITHUB_TOKEN=your-github-personal-access-token
GITHUB_REPO=your-org/your-repo
GITHUB_BRANCH_PREFIX=feature/auto-bdd

# LLM Configuration
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACE_API_KEY=your-huggingface-api-key  # Optional
MODEL_NAME=gpt-4

# RAG Configuration
VECTOR_DB_PATH=./chroma_db
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Required API Keys and Tokens

#### Jira API Token
1. Go to [Atlassian Account Settings](https://id.atlassian.com/manage-profile/security/api-tokens)
2. Click "Create API token"
3. Give it a label (e.g., "BDD Agent")
4. Copy the generated token

#### GitHub Personal Access Token
1. Go to GitHub Settings > Developer settings > Personal access tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo`, `workflow` (if using GitHub Actions)
4. Copy the generated token

#### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the generated key

#### Hugging Face API Key (Optional)
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Select "Read" access
4. Copy the generated token

### 4. Repository Structure

Ensure your GitHub repository has the following structure:
```
your-repo/
├── src/
│   ├── features/          # Gherkin feature files
│   └── step-definitions/  # TypeScript step definitions
└── ...
```

### 5. Run the Application

#### Development Mode
```bash
cd backend
python main.py
```

#### Production Mode with Uvicorn
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 6. Verify Installation

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

Test Jira connection:
```bash
curl http://localhost:8000/debug/jira-tickets
```

## Usage Examples

### Basic Workflow Trigger
```bash
# Trigger workflow for a specific sprint
curl -X POST "http://localhost:8000/workflow/trigger" \
  -H "Content-Type: application/json" \
  -d '{"sprint_id": 123}'

# Trigger workflow for all active sprints
curl -X POST "http://localhost:8000/workflow/trigger" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Monitor Workflow Progress
```bash
# Get status of a specific run
curl "http://localhost:8000/workflow/status/run_20241201_143022"

# Get all runs
curl "http://localhost:8000/workflow/runs"
```

## Configuration Options

### Model Configuration
- `MODEL_NAME`: OpenAI model to use (default: "gpt-4")
- `HUGGINGFACE_API_KEY`: Optional, for embeddings (falls back to local)

### GitHub Configuration
- `GITHUB_BRANCH_PREFIX`: Prefix for auto-generated branches (default: "feature/auto-bdd")

### RAG Configuration
- `VECTOR_DB_PATH`: Path to ChromaDB storage (default: "./chroma_db")

## Troubleshooting

### Common Issues

1. **Jira Connection Failed**
   - Verify JIRA_URL, JIRA_EMAIL, and JIRA_API_TOKEN
   - Check if the API token has proper permissions

2. **GitHub Connection Failed**
   - Verify GITHUB_TOKEN and GITHUB_REPO
   - Ensure the token has repo access

3. **OpenAI API Errors**
   - Verify OPENAI_API_KEY is valid
   - Check API quota and billing

4. **Empty RAG Search Results**
   - Ensure the repository has existing feature files
   - Check if the codebase indexing completed successfully

### Logs

Logs are written to `backend/logs/workflow.log` with DEBUG level by default. Check this file for detailed error information.

## API Documentation

Once the server is running, you can access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Security Considerations

- Store all API keys and tokens securely
- Use environment variables or secure secret management
- Regularly rotate API tokens
- Limit GitHub token permissions to minimum required
- Consider using Jira project-specific permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]
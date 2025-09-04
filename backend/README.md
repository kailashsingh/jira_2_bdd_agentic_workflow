# Jira to BDD Agent

An intelligent agentic workflow system that automatically generates BDD (Behavior-Driven Development) test scenarios and step definitions from Jira tickets.

## Features

- 🎯 **Automatic Test Generation**: Converts Jira user stories into BDD test scenarios
- 🤖 **LLM-Powered Intelligence**: Uses GPT-4 for intelligent test generation
- 📚 **RAG Integration**: Learns from existing codebase to maintain consistency
- 🔄 **Git Integration**: Automatically creates branches and pull requests
- 📊 **Web Dashboard**: Monitor and trigger workflows through intuitive UI
- 🔗 **Full Traceability**: Links Jira tickets to generated tests and PRs

## Prerequisites

- Python 3.8+
- Jira Cloud account or self-hosted Jira instance
- GitHub account with repository access
- OpenAI API key

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jira-bdd-agent.git
cd jira-bdd-agent

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Configuration

Edit `.env` file with your credentials:

```env
# Jira Configuration
JIRA_URL=https://your-instance.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-jira-api-token

# GitHub Configuration
GITHUB_TOKEN=ghp_your_github_token
GITHUB_REPO=username/bdd-test-repo

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key
```

### 3. Run the Application

```bash
# Start the API server
python main.py

# API will be available at http://localhost:8000
```

### 4. Access the Dashboard

Open `dashboard.html` in your browser or serve it with:

```bash
python -m http.server 8080
# Navigate to http://localhost:8080/dashboard.html
```

## API Endpoints

### Trigger Workflow
```bash
POST /workflow/trigger
{
    "sprint_id": 5  # Optional, uses active sprint if not provided
}
```

### Check Status
```bash
GET /workflow/status/{run_id}
```

### Health Check
```bash
GET /health
```

## Project Structure

```
jira-bdd-agent/
├── src/
│   ├── agents/           # LLM agents for different tasks
│   ├── tools/            # Integration tools (Jira, GitHub, RAG)
│   ├── config/           # Configuration and settings
│   └── prompts/          # LLM prompt templates
├── sample-bdd-repo/      # Sample BDD test repository
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
└── dashboard.html        # Web UI dashboard
```

## How It Works

1. **Trigger**: Workflow initiated via API or webhook when sprint starts
2. **Fetch**: Agent pulls Jira tickets from the sprint
3. **Analyze**: LLM determines if tickets require testing
4. **Index**: RAG system indexes existing BDD codebase
5. **Generate**: LLM creates BDD scenarios and step definitions
6. **Commit**: Agent creates branch and commits new tests
7. **PR**: Automatically raises pull request for review
8. **Update**: Jira ticket updated with PR link

## Sample Jira Tickets

Create these in your Jira for testing:

**DEMO-1**: User Password Reset
- Type: Story
- Description: Users should be able to reset their password via email

**DEMO-2**: Product Filtering  
- Type: Story
- Description: Customers can filter products by category and price

**DEMO-3**: Database Migration
- Type: Task
- Description: Migrate user table schema (non-testable)

## Troubleshooting

### Jira Connection Issues
- Verify API token is valid
- Check user has project access
- Ensure JIRA_URL includes https://

### GitHub Rate Limiting
- Use personal access token
- Implement request caching
- Add retry logic with backoff

### LLM Token Limits
- Chunk large contexts
- Use GPT-3.5 for cost optimization
- Implement context summarization

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Support

For issues or questions, please open a GitHub issue or contact the maintainers.

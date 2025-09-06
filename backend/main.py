from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from src.agents.orchestrator import WorkflowOrchestrator
import asyncio
from datetime import datetime
from src.config.logging import setup_logging, get_logger

# Set up logging with DEBUG level to see all logs
setup_logging(log_level="DEBUG")
logger = get_logger(__name__)

app = FastAPI(title="Jira to BDD Agent API")              

class WorkflowRequest(BaseModel):
    sprint_id: Optional[int] = None
    jira_keys: Optional[List[str]] = None

class WorkflowResponse(BaseModel):
    status: str
    run_id: str
    started_at: str
    message: str

# Store for workflow runs (in production, use a database)
workflow_runs = {}

@app.post("/workflow/trigger", response_model=WorkflowResponse)
async def trigger_workflow(request: WorkflowRequest):
    """Trigger the BDD generation workflow"""
    logger.info(f"Received workflow trigger request with sprint_id={request.sprint_id}")
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start workflow in background
    logger.info(f"Starting workflow {run_id} in background")
    asyncio.create_task(run_workflow_async(run_id, request.sprint_id))
    
    workflow_runs[run_id] = {
        'status': 'running',
        'started_at': datetime.now().isoformat(),
        'sprint_id': request.sprint_id
    }
    logger.info(f"Created workflow run entry for {run_id}")
    
    return WorkflowResponse(
        status="running",
        run_id=run_id,
        started_at=workflow_runs[run_id]['started_at'],
        message="Workflow triggered successfully"
    )

async def run_workflow_async(run_id: str, sprint_id: Optional[int]):
    """Run the workflow asynchronously"""
    logger.info(f"Starting workflow execution for run_id={run_id}")
    try:
        logger.debug("Initializing WorkflowOrchestrator")
        orchestrator = WorkflowOrchestrator()
        
        logger.info(f"Running orchestrator for sprint_id={sprint_id}")
        result = await asyncio.to_thread(orchestrator.run, sprint_id)
        
        logger.info(f"Workflow {run_id} completed successfully")
        workflow_runs[run_id]['status'] = 'completed'
        workflow_runs[run_id]['result'] = result
        workflow_runs[run_id]['completed_at'] = datetime.now().isoformat()
    except Exception as e:
        logger.error(f"Workflow {run_id} failed with error: {str(e)}", exc_info=True)
        workflow_runs[run_id]['status'] = 'failed'
        workflow_runs[run_id]['error'] = str(e)

@app.get("/workflow/status/{run_id}")
async def get_workflow_status(run_id: str):
    """Get the status of a workflow run"""
    if run_id not in workflow_runs:
        raise HTTPException(status_code=404, detail="Run ID not found")
    
    return workflow_runs[run_id]

@app.get("/workflow/runs")
async def get_all_runs():
    """Get all workflow runs"""
    return list(workflow_runs.values())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Jira to BDD Agent"}

@app.get("/debug/rag-search")
async def debug_rag_search(query: str):
    """Debug endpoint to test RAG search"""
    from src.tools.rag_tools import RAGTools
    rag_tools = RAGTools()
    results = rag_tools.search_similar_code(query)
    return {"query": query, "results": results}

@app.get("/debug/jira-tickets")
async def debug_jira_tickets():
    """Debug endpoint to test Jira connection"""
    logger.info("Debug endpoint: Fetching Jira tickets")
    from src.tools.jira_tools import JiraTools
    jira_tools = JiraTools()
    tickets = jira_tools.get_sprint_tickets()
    logger.info(f"Debug endpoint: Found {len(tickets)} tickets")
    return {"tickets": tickets}

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
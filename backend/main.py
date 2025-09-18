from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union
import uvicorn
from src.agents.orchestrator import WorkflowOrchestrator
import asyncio
from datetime import datetime
from src.config.logging import setup_logging, get_logger

# Set up logging with DEBUG level to see all logs
setup_logging(log_level="INFO")
logger = get_logger(__name__)

app = FastAPI(title="Jira to BDD Agent API")              

class WorkflowRequest(BaseModel):
    sprint_id: Optional[int] = None
    jira_keys: Optional[List[str]] = None

class TicketRequest(BaseModel):
    ticket_keys: Union[str, List[str]]  # Can be single string or list of strings
    project: Optional[str] = None

class WorkflowResponse(BaseModel):
    status: str
    run_id: str
    started_at: str
    message: str

class NavigationTestRequest(BaseModel):
    summary: str
    description: str
    acceptance_criteria: str

class NavigationTestResponse(BaseModel):
    navigation_needed: bool
    extracted_url: Optional[str] = None
    navigation_instructions: List[str] = []
    page_data: Optional[dict] = None

class UrlValidationResponse(BaseModel):
    url: str
    accessible: bool
    title: Optional[str] = None
    status_code: Optional[int] = None
    elements_count: Optional[int] = None
    forms_count: Optional[int] = None

# Store for workflow runs (in production, use a database)
workflow_runs = {}

@app.post("/workflow/trigger/tickets", response_model=WorkflowResponse)
async def trigger_tickets_workflow(request: TicketRequest):
    """Trigger the BDD generation workflow for one or more tickets"""
    # Convert single ticket to list for unified processing
    ticket_keys = [request.ticket_keys] if isinstance(request.ticket_keys, str) else request.ticket_keys
    
    logger.info(f"Received workflow trigger request for tickets: {ticket_keys}")
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start workflow in background
    logger.info(f"Starting tickets workflow {run_id} in background")
    asyncio.create_task(run_tickets_workflow_async(run_id, ticket_keys, request.project))
    
    workflow_runs[run_id] = {
        'status': 'running',
        'started_at': datetime.now().isoformat(),
        'ticket_keys': ticket_keys,
        'project': request.project
    }
    logger.info(f"Created workflow run entry for {run_id}")
    
    return WorkflowResponse(
        status="running",
        run_id=run_id,
        started_at=workflow_runs[run_id]['started_at'],
        message=f"Workflow triggered successfully for tickets: {ticket_keys}"
    )

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

async def run_tickets_workflow_async(run_id: str, ticket_keys: List[str], project: Optional[str] = None):
    """Run the workflow for one or more tickets asynchronously"""
    logger.info(f"Starting workflow execution for run_id={run_id}, tickets={ticket_keys}")
    
    try:
        logger.debug("Initializing WorkflowOrchestrator")
        orchestrator = WorkflowOrchestrator()
        
        logger.info(f"Processing tickets in batch: {ticket_keys}")
        result = await orchestrator.process_tickets(ticket_keys, project)
        
        # Update workflow status based on result
        if result['status'] == 'error':
            logger.error(f"Workflow failed: {result['message']}")
            workflow_runs[run_id].update({
                'status': 'failed',
                'error': result['message'],
                'completed_at': datetime.now().isoformat()
            })
        else:
            # Workflow completed successfully
            workflow_runs[run_id].update({
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'result': result.get('result', {})
            })
        
    except Exception as e:
        logger.error(f"Workflow {run_id} failed with error: {str(e)}", exc_info=True)
        workflow_runs[run_id]['status'] = 'failed'
        workflow_runs[run_id]['error'] = str(e)

async def run_workflow_async(run_id: str, sprint_id: Optional[int]):
    """Run the workflow asynchronously"""
    logger.info(f"Starting workflow execution for run_id={run_id}")
    try:
        logger.debug("Initializing WorkflowOrchestrator")
        orchestrator = WorkflowOrchestrator()
        
        logger.info(f"Running orchestrator for sprint_id={sprint_id}")
        # result = await asyncio.to_thread(orchestrator.run, sprint_id)
        result = await orchestrator.run(sprint_id)
        
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

@app.post("/debug/test-navigation", response_model=NavigationTestResponse)
async def test_navigation(request: NavigationTestRequest):
    """Test application navigation functionality with sample Jira ticket data"""
    logger.info("Debug endpoint: Testing application navigation")
    
    try:
        from src.tools.application_tools import ApplicationTools
        
        # Create mock Jira data from request
        jira_data = {
            'summary': request.summary,
            'description': request.description,
            'acceptance_criteria': request.acceptance_criteria,
            'key': 'TEST-NAV'
        }
        
        app_tools = ApplicationTools()
        
        # Check if navigation is needed
        navigation_needed = app_tools.needs_navigation(jira_data)
        
        if not navigation_needed:
            return NavigationTestResponse(
                navigation_needed=False,
                navigation_instructions=[]
            )
        
        # Extract URL and instructions
        extracted_url = app_tools._extract_url_from_jira_data(jira_data)
        navigation_instructions = app_tools._extract_navigation_instructions(jira_data)
        
        # Navigate and collect data
        page_data = app_tools.navigate_and_collect_data(jira_data)
        
        return NavigationTestResponse(
            navigation_needed=True,
            extracted_url=extracted_url,
            navigation_instructions=navigation_instructions,
            page_data=page_data
        )
        
    except Exception as e:
        logger.error(f"Navigation test failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Navigation test failed: {str(e)}")

@app.get("/debug/validate-url", response_model=UrlValidationResponse)
async def validate_url(url: str):
    """Validate if a URL is accessible and extract basic page information"""
    logger.info(f"Debug endpoint: Validating URL {url}")
    
    try:
        from src.tools.application_tools import ApplicationTools
        
        app_tools = ApplicationTools()
        app_tools.start_browser()
        
        try:
            # Navigate to the URL
            app_tools.page.goto(url, wait_until='networkidle', timeout=10000)
            
            # Get basic page information
            title = app_tools.page.title()
            elements = app_tools._collect_page_elements()
            forms = app_tools._collect_forms()
            
            return UrlValidationResponse(
                url=url,
                accessible=True,
                title=title,
                status_code=200,
                elements_count=len(elements),
                forms_count=len(forms)
            )
            
        except Exception as e:
            logger.warning(f"Failed to access URL {url}: {str(e)}")
            return UrlValidationResponse(
                url=url,
                accessible=False,
                status_code=500
            )
        finally:
            app_tools.close_browser()
            
    except Exception as e:
        logger.error(f"URL validation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"URL validation failed: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
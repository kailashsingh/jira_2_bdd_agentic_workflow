from langgraph.graph import Graph, END
from typing import Dict, Any, Optional
import asyncio
from src.tools.jira_tools import JiraTools
from src.tools.github_tools import GitHubTools
from src.tools.rag_tools import RAGTools
from src.tools.application_tools import ApplicationTools
from src.agents.bdd_generator_agent import BDDGeneratorAgent
from src.config.settings import settings
from src.config.logging import get_logger
import json

logger = get_logger(__name__)

class WorkflowOrchestrator:
    def __init__(self):
        self.jira_tools = JiraTools()
        self.github_tools = GitHubTools()
        self.rag_tools = RAGTools()
        self.application_tools = ApplicationTools()
        self.bdd_agent = BDDGeneratorAgent(self.rag_tools)
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow"""
        workflow = Graph()
        
        # Define nodes
        workflow.add_node("fetch_jira_tickets", self.fetch_jira_tickets)
        workflow.add_node("process_ticket", self.process_ticket)
        workflow.add_node("index_codebase", self.index_codebase)
        workflow.add_node("navigate_application", self.navigate_application)
        workflow.add_node("generate_tests", self.generate_tests)
        workflow.add_node("create_pr", self.create_pr)
        workflow.add_node("next_ticket", self.next_ticket)
        
        # Define edges
        workflow.add_edge("fetch_jira_tickets", "process_ticket")
        workflow.add_conditional_edges(
            "process_ticket",
            self.decide_next_node_post_process_ticket,
            {
                "index_codebase": "index_codebase",
                "navigate_application": "navigate_application",
                "generate_tests": "generate_tests",
                "next_ticket": "next_ticket",
                END: END
            }
        )
        workflow.add_conditional_edges(
            "index_codebase",
            self.decide_next_node_post_index_codebase,
            {
                "navigate_application": "navigate_application",
                "generate_tests": "generate_tests",
                "next_ticket": "next_ticket",
                END: END
            }
        )
        workflow.add_edge("navigate_application", "generate_tests")
        workflow.add_edge("generate_tests", "create_pr")
        workflow.add_edge("create_pr", "next_ticket")
        workflow.add_edge("next_ticket", "process_ticket")
        
        # Set entry point
        workflow.set_entry_point("fetch_jira_tickets")
        
        return workflow.compile()
    
    def fetch_jira_tickets(self, state: Dict) -> Dict:
        """Fetch tickets from Jira"""
        sprint_id = state.get('sprint_id')

        logger.info(f"Fetching tickets for sprint_id: {sprint_id if sprint_id else 'active sprint'}")

        tickets = self.jira_tools.get_sprint_tickets(sprint_id)
        return {**state, 'tickets': tickets, 'current_ticket_index': 0}
    
    def index_codebase(self, state: Dict) -> Dict:
        """Index the existing BDD codebase"""
        features = self.github_tools.get_feature_files()
        step_defs = self.github_tools.get_step_definitions()

        logger.info(f'Indexing the \'bdd_codebase\' for RAG')

        self.rag_tools.index_codebase(features, step_defs, self.bdd_agent)
        return {**state, 'codebase_indexed': True}
    
    def process_ticket(self, state: Dict) -> Dict:
        """Process the current ticket"""
        tickets = state['tickets']
        current_index = state['current_ticket_index']
        
        if current_index >= len(tickets):
            return {**state, 'completed': True}
        
        current_ticket = tickets[current_index]
        logger.info(f'Processing ticket: {current_ticket["key"]}')
        is_testable = self.bdd_agent.is_testable(current_ticket)
        logger.info(f'Ticket {current_ticket["key"]} is testable: {is_testable}')
        return {
            **state,
            'current_ticket': current_ticket,
            'is_testable': is_testable
        }
    
    def decide_next_node_post_process_ticket(self, state: Dict) -> str:
        """Decide the next node after processing a ticket"""

        if state.get('completed', False):
            return END
        
        if state.get('is_testable', False):
            # Check if codebase has been indexed yet
            if not state.get('codebase_indexed', False):
                return "index_codebase"
            return "navigate_application" if self.should_navigate_application(state) else "generate_tests"
        return "next_ticket"

    def decide_next_node_post_index_codebase(self, state: Dict) -> str:
        """Decide the next node after indexing codebase"""
        
        if state.get('completed', False):
            return END
        
        if state.get('is_testable', False):
            return "navigate_application" if self.should_navigate_application(state) else "generate_tests"
        return "next_ticket"

    def should_navigate_application(self, state: Dict) -> bool:
        """Conditional edge to determine if application navigation is needed"""
        
        current_ticket = state.get('current_ticket')
        if not current_ticket:
            return False
        
        return self.application_tools.needs_navigation(current_ticket)

    async def navigate_application(self, state: Dict) -> Dict:
        """Navigate to application and collect data"""
        ticket = state['current_ticket']
        logger.info(f'Navigating application for ticket: {ticket["key"]}')
        
        # Get existing BDD data for context
        query = f"{ticket['summary']} {ticket['description'][:200]}"
        existing_bdd_data = self.rag_tools.search_similar_code(query)
        
        # Navigate and collect application data
        application_data = await self.application_tools.navigate_and_collect_data_using_mcp(
            ticket, 
            self.bdd_agent
        )

        logger.info(f'Collected application data for {ticket["key"]}: {len(application_data.split(" "))} words')
        return {**state, 'application_data': application_data}
    
    def generate_tests(self, state: Dict) -> Dict:
        """Generate BDD tests for the current ticket"""
        ticket = state['current_ticket']
        application_data = state.get('application_data', '')
        
        logger.info(f'Generating BDD tests for ticket: {ticket["key"]}')

        # Search for similar code
        query = f"{ticket['summary']} {ticket['description'][:200]}"
        similar_code = self.rag_tools.search_similar_code(query)
        
        # Generate BDD scenarios and step definitions with application data
        generated = self.bdd_agent.generate_bdd_scenarios(ticket, similar_code, application_data)
        
        return {**state, 'generated_tests': generated}
    
    def create_pr(self, state: Dict) -> Dict:
        """Create a PR with the generated tests"""

        logger.info(f'Creating PR for ticket: {state["current_ticket"]["key"]}')

        generated = state['generated_tests']
        ticket_key = generated['ticket_key']
        
        # Create a new branch
        branch_name = f"{settings.github_branch_prefix}/{ticket_key.lower()}"
        self.github_tools.create_branch(branch_name)
        
        # Create/update files
        feature_path = f"{generated['feature_file_name']}"
        step_def_path = f"{generated['steps_file_name']}"
        
        self.github_tools.create_or_update_file(
            feature_path,
            generated['feature'],
            branch_name,
            f"feat({ticket_key}): Add BDD scenarios"
        )
        
        self.github_tools.create_or_update_file(
            step_def_path,
            generated['step_definitions'],
            branch_name,
            f"feat({ticket_key}): Add step definitions"
        )
        
        # Create PR
        pr_url = self.github_tools.create_pull_request(
            branch_name,
            f"[{ticket_key}] Auto-generated BDD tests",
            f"This PR contains auto-generated BDD tests for {ticket_key}\n\n"
            f"Generated from: {state['current_ticket']['summary']}"
        )
        
        logger.info(f'Created PR for ticket {ticket_key}: {pr_url}')

        # Update Jira ticket
        comment = f"BDD tests have been auto-generated for this ticket.\n"
        comment += f"Pull Request: {pr_url}"
        self.jira_tools.update_ticket_comment(ticket_key, comment)
        
        # Move to next ticket
        return {
            **state,
            'pr_urls': state.get('pr_urls', []) + [pr_url]
        }
    
    def next_ticket(self, state: Dict) -> Dict:
        """Move to the next ticket"""

        return {
            **state,
            'current_ticket_index': state['current_ticket_index'] + 1
        }
    
    async def run(self, sprint_id: Optional[int] = None) -> Dict:
        """Execute the workflow"""
        initial_state = {'sprint_id': sprint_id}
        result = await self.workflow.ainvoke(initial_state, config={"recurssionLimit": 10})
        logger.debug(f"Workflow run result: {result}")
        return result
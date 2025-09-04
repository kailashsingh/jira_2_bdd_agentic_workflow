from langgraph.graph import Graph, END
from typing import Dict, Any, Optional
from src.tools.jira_tools import JiraTools
from src.tools.github_tools import GitHubTools
from src.tools.rag_tools import RAGTools
from src.agents.bdd_generator_agent import BDDGeneratorAgent
from src.config.settings import settings
import json

class WorkflowOrchestrator:
    def __init__(self):
        self.jira_tools = JiraTools()
        self.github_tools = GitHubTools()
        self.rag_tools = RAGTools()
        self.bdd_agent = BDDGeneratorAgent(self.rag_tools)
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow"""
        workflow = Graph()
        
        # Define nodes
        workflow.add_node("fetch_jira_tickets", self.fetch_jira_tickets)
        workflow.add_node("index_codebase", self.index_codebase)
        workflow.add_node("process_ticket", self.process_ticket)
        workflow.add_node("generate_tests", self.generate_tests)
        workflow.add_node("create_pr", self.create_pr)
        
        # Define edges
        workflow.add_edge("fetch_jira_tickets", "index_codebase")
        workflow.add_edge("index_codebase", "process_ticket")
        workflow.add_conditional_edges(
            "process_ticket",
            self.should_generate_tests,
            {
                True: "generate_tests",
                False: END
            }
        )
        workflow.add_edge("generate_tests", "create_pr")
        workflow.add_edge("create_pr", "process_ticket")
        
        # Set entry point
        workflow.set_entry_point("fetch_jira_tickets")
        
        return workflow.compile()
    
    def fetch_jira_tickets(self, state: Dict) -> Dict:
        """Fetch tickets from Jira"""
        sprint_id = state.get('sprint_id')
        tickets = self.jira_tools.get_sprint_tickets(sprint_id)
        return {**state, 'tickets': tickets, 'current_ticket_index': 0}
    
    def index_codebase(self, state: Dict) -> Dict:
        """Index the existing BDD codebase"""
        features = self.github_tools.get_feature_files()
        step_defs = self.github_tools.get_step_definitions()
        self.rag_tools.index_codebase(features, step_defs)
        return {**state, 'codebase_indexed': True}
    
    def process_ticket(self, state: Dict) -> Dict:
        """Process the current ticket"""
        tickets = state['tickets']
        current_index = state['current_ticket_index']
        
        if current_index >= len(tickets):
            return {**state, 'completed': True}
        
        current_ticket = tickets[current_index]
        is_testable = self.bdd_agent.is_testable(current_ticket)
        
        return {
            **state,
            'current_ticket': current_ticket,
            'is_testable': is_testable
        }
    
    def should_generate_tests(self, state: Dict) -> bool:
        """Conditional edge to determine if tests should be generated"""
        return state.get('is_testable', False) and not state.get('completed', False)
    
    def generate_tests(self, state: Dict) -> Dict:
        """Generate BDD tests for the current ticket"""
        ticket = state['current_ticket']
        
        # Search for similar code
        query = f"{ticket['summary']} {ticket['description'][:200]}"
        similar_code = self.rag_tools.search_similar_code(query)
        
        # Generate BDD scenarios and step definitions
        generated = self.bdd_agent.generate_bdd_scenarios(ticket, similar_code)
        
        return {**state, 'generated_tests': generated}
    
    def create_pr(self, state: Dict) -> Dict:
        """Create a PR with the generated tests"""
        generated = state['generated_tests']
        ticket_key = generated['ticket_key']
        
        # Create a new branch
        branch_name = f"{settings.github_branch_prefix}/{ticket_key.lower()}"
        self.github_tools.create_branch(branch_name)
        
        # Create/update files
        feature_path = f"features/{ticket_key.lower()}.feature"
        step_def_path = f"step-definitions/{ticket_key.lower()}.steps.ts"
        
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
        
        # Update Jira ticket
        comment = f"BDD tests have been auto-generated for this ticket.\n"
        comment += f"Pull Request: {pr_url}"
        self.jira_tools.update_ticket_comment(ticket_key, comment)
        
        # Move to next ticket
        return {
            **state,
            'current_ticket_index': state['current_ticket_index'] + 1,
            'pr_urls': state.get('pr_urls', []) + [pr_url]
        }
    
    def run(self, sprint_id: Optional[int] = None) -> Dict:
        """Execute the workflow"""
        initial_state = {'sprint_id': sprint_id}
        result = self.workflow.invoke(initial_state)
        return result
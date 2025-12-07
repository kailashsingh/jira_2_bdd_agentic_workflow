from langgraph.graph import Graph, END
from typing import Dict, Any, Optional
from src.tools.rag_tools import RAGTools
from src.agents.bdd_generator_agent import BDDGeneratorAgent
from src.config.logging import get_logger

logger = get_logger(__name__)

class ExecutorCriticOrchestrator:
    """Orchestrator for the executor-critic workflow"""
    
    def __init__(self):
        self.rag_tools = RAGTools()
        self.bdd_agent = BDDGeneratorAgent(self.rag_tools)
        self.executor_critic_workflow = self._create_executor_critic_workflow()
    
    def _create_executor_critic_workflow(self) -> Graph:
        """Create the executor-critic workflow graph"""
        workflow = Graph()
        
        # Define nodes
        workflow.add_node("executor", self.generate_tests)
        workflow.add_node("critic", self.critic)
        
        # Define edges: executor -> critic
        workflow.add_edge("executor", "critic")
        
        # Conditional edge from critic: either go back to executor or end
        workflow.add_conditional_edges(
            "critic",
            self.decide_after_critic,
            {
                "executor": "executor",
                END: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("executor")
        
        return workflow.compile()
    
    def generate_tests(self, state: Dict) -> Dict:
        """Generate BDD tests for the current ticket (executor node)"""
        ticket = state['current_ticket']
        application_data = state.get('application_data', '')
        critic_feedback = state.get('critic_feedback', '')
        needs_revision = state.get('needs_revision', False)
        
        if needs_revision and critic_feedback:
            logger.info(f'Regenerating BDD tests for ticket: {ticket["key"]} based on critic feedback')
        else:
            logger.info(f'Generating BDD tests for ticket: {ticket["key"]}')
        
        # Search for similar code
        query = f"{ticket['summary']} {ticket['description'][:200]}"
        #similar_code = self.rag_tools.search_similar_code(query)
        similar_code = []
        
        # Generate BDD scenarios and step definitions with application data and critic feedback
        generated = self.bdd_agent.generate_bdd_scenarios(
            ticket, 
            similar_code, 
            application_data,
            critic_feedback if needs_revision else None
        )
        
        return {**state, 'generated_tests': generated}
    
    def critic(self, state: Dict) -> Dict:
        """Critic node that reviews and evaluates the generated tests"""
        generated_tests = state.get('generated_tests')
        ticket = state.get('current_ticket')
        
        if not generated_tests:
            logger.warning("No generated tests found for critic to review")
            return {**state, 'critic_feedback': 'No tests to review', 'needs_revision': False}
        
        logger.info(f'Critic reviewing generated tests for ticket: {ticket["key"] if ticket else "unknown"}')
        
        # Use the BDD agent to critique the generated tests
        feedback = self.bdd_agent.critique_tests(
            ticket,
            generated_tests,
            state.get('application_data', '')
        )
        
        return {
            **state,
            'critic_feedback': feedback.get('feedback', ''),
            'needs_revision': feedback.get('needs_revision', False),
            'revision_count': state.get('revision_count', 0) + (1 if feedback.get('needs_revision', False) else 0)
        }
    
    def decide_after_critic(self, state: Dict) -> str:
        """Decide the next node after critic review"""
        needs_revision = state.get('needs_revision', False)
        revision_count = state.get('revision_count', 0)
        max_revisions = 3  # Prevent infinite loops
        
        if needs_revision and revision_count < max_revisions:
            logger.info(f"Tests need revision (attempt {revision_count + 1}/{max_revisions}), sending back to executor")
            return "executor"
        else:
            if revision_count >= max_revisions:
                logger.warning(f"Maximum revisions ({max_revisions}) reached, ending workflow")
            else:
                logger.info("Tests approved by critic, ending workflow")
            return END
    
    async def trigger_executor_critic_workflow(self, state: Dict) -> Dict:
        """Trigger the executor-critic workflow"""
        result = await self.executor_critic_workflow.ainvoke(state, config={"recurssionLimit": 10})
        logger.debug(f"Executor-critic workflow result: {result}")
        return result


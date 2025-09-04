from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, List
from src.config.settings import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

class BDDGeneratorAgent:
    def __init__(self, rag_tools):
        logger.info(f'OpenAI Model Used: {settings.model_name}')
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )
        self.rag_tools = rag_tools
    
    def is_testable(self, ticket: Dict) -> bool:
        """Determine if a ticket requires testing"""
        prompt = ChatPromptTemplate.from_template("""
        Analyze this Jira ticket and determine if it requires BDD testing.
        
        Ticket Details:
        Type: {issue_type}
        Summary: {summary}
        Description: {description}
        Acceptance Criteria: {acceptance_criteria}
        
        Return TRUE if this is a user story or feature that requires testing.
        Return FALSE if this is a bug fix, documentation, or infrastructure task.
        
        Answer with only TRUE or FALSE.
        """)
        
        response = self.llm.invoke(
            prompt.format_messages(**ticket)
        )
        
        return "TRUE" in response.content.upper()
    
    def generate_bdd_scenarios(self, ticket: Dict, existing_code: List[Dict]) -> Dict:
        """Generate BDD scenarios and step definitions"""
        
        # Format existing code context
        code_context = "\n\n".join([
            f"File: {code['metadata']['path']}\n{code['content'][:500]}..."
            for code in existing_code[:3]
        ])
        
        prompt = ChatPromptTemplate.from_template("""
        You are a BDD test automation expert. Generate BDD scenarios and step definitions
        for the following Jira ticket.
        
        Jira Ticket:
        Key: {key}
        Summary: {summary}
        Description: {description}
        Acceptance Criteria: {acceptance_criteria}
        
        Existing Code Context:
        {code_context}
        
        Generate:
        1. A Gherkin feature file with comprehensive scenarios
        2. TypeScript step definitions using WebDriverIO
        
        Follow these guidelines:
        - Use existing step definitions where possible
        - Follow the coding patterns from the existing codebase
        - Include both happy path and edge cases
        - Use clear, business-readable language
        
        Format your response as:
        FEATURE_FILE:
        [Gherkin content here]
        
        STEP_DEFINITIONS:
        [TypeScript code here]
        """)
        
        response = self.llm.invoke(
            prompt.format_messages(
                **ticket,
                code_context=code_context
            )
        )
        
        # Parse the response
        content = response.content
        feature_start = content.find("FEATURE_FILE:") + len("FEATURE_FILE:")
        step_def_start = content.find("STEP_DEFINITIONS:")
        
        feature_content = content[feature_start:step_def_start].strip()
        step_def_content = content[step_def_start + len("STEP_DEFINITIONS:"):].strip()
        
        return {
            'feature': feature_content,
            'step_definitions': step_def_content,
            'ticket_key': ticket['key']
        }
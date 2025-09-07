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
    
    def generate_bdd_scenarios(self, ticket: Dict, existing_code: List[Dict], application_data: Dict = None) -> Dict:
        """Generate BDD scenarios and step definitions"""
        
        # Format existing code context
        code_context = "\n\n".join([
            f"File: {code['metadata']['path']}\n{code['content'][:500]}..."
            for code in existing_code[:3]
        ])
        
        # Format application data context
        app_context = ""
        if application_data:
            app_context = f"""
Application Data Collected:
URL: {application_data.get('url', 'N/A')}
Title: {application_data.get('title', 'N/A')}
Elements Found: {len(application_data.get('elements', []))}
Forms Found: {len(application_data.get('forms', []))}
Navigation Flow: {', '.join(application_data.get('navigation_flow', []))}

Key Elements:
{chr(10).join([f"- {elem.get('type', 'unknown')}: {elem.get('text', elem.get('placeholder', 'N/A'))}" for elem in application_data.get('elements', [])[:10]])}

Forms:
{chr(10).join([f"- Form with {len(form.get('inputs', []))} inputs" for form in application_data.get('forms', [])[:3]])}
"""
        
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
        
        {app_context}
        
        Generate:
        1. A Gherkin feature file with comprehensive scenarios
        2. TypeScript step definitions using WebDriverIO
        
        Follow these guidelines:
        - Use existing step definitions where possible
        - Follow the coding patterns from the existing codebase
        - Include both happy path and edge cases
        - Use clear, business-readable language
        - If application data is provided, use specific element selectors and navigation flows
        - Include realistic test data based on the actual application elements found
        
        Format your response as:
        FEATURE_FILE:
        [Gherkin content here]
        
        STEP_DEFINITIONS:
        [TypeScript code here]
        """)
        
        response = self.llm.invoke(
            prompt.format_messages(
                **ticket,
                code_context=code_context,
                app_context=app_context
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
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from typing import Dict, List

import re
from src.config.settings import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

class BDDGeneratorAgent:
    def __init__(self, rag_tools):
        logger.info(f'Anthropic Model Used: {settings.model_name}')
        # self.llm = ChatOpenAI(
        #     model=settings.model_name,
        #     temperature=0.2,
        #     api_key=settings.openai_api_key,
        # )
        self.llm = ChatAnthropic(
            model_name=settings.model_name,
            temperature=0.2,
            api_key=settings.anthropic_api_key
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
    
    def generate_description_of_file(self, filePath: str, fileContent: str) -> str:
        """Generate the description of the file"""

        prompt = ChatPromptTemplate.from_template("""
        You are a BDD test automation expert. You will be provided with either a feature file or a step-definition file.
        Your task is to generate a concise, descriptive summary of the file so that it can be indexed and later matched against Jira tickets.

        Instructions:
            1. Detect the file type: Identify whether the input is a feature file (.feature) or a step-definition file (code implementing steps).

            2. If Feature File:
                - Extract the overall purpose of the feature.
                - Summarize the functionality being tested.
                - List the scenarios in plain language (short bullet points).
                - Mention the approximate path it belongs to (e.g., features/<feature_name>.feature).

            3. If Step-Definition File:
                - Identify which feature or functionality it supports.
                - Summarize the test steps implemented.
                - Mention the programming language and framework (e.g., JavaScript + Cucumber + Playwright).
                - Suggest the approximate path (e.g., step_definitions/<feature_name>Steps.js).

            4. Write the description in a structured format so it can later be cross-referenced with Jira tickets.
                                                
        File:
            File Path: {filePath}
            File Content:
            {fileContent}
        """)

        response = self.llm.invoke(
            prompt.format_messages(
                filePath = filePath,
                fileContent = fileContent
            )
        )

        content = response.content

        logger.debug(f"Description generated for file {filePath}: {content}")

        return content
    
    def generate_bdd_scenarios(self, ticket: Dict, existing_code: List[Dict], application_data: Dict = None) -> Dict:
        """Generate BDD scenarios and step definitions"""
        
        # Format existing code context
        code_context = "\n\n".join([
            f"File: {code['metadata']['path']}\n{code['content']}..."
            for code in existing_code[:3]
        ])
        
        # Format application data context
        # app_context = ""
        # if application_data:
        #     app_context = f"Application Data Collected:{app_context}"
#             app_context = f"""
# Application Data Collected:
# URL: {application_data.get('url', 'N/A')}
# Title: {application_data.get('title', 'N/A')}
# Elements Found: {len(application_data.get('elements', []))}
# Forms Found: {len(application_data.get('forms', []))}
# Navigation Flow: {', '.join(application_data.get('navigation_flow', []))}

# Key Elements:
# {chr(10).join([f"- {elem.get('type', 'unknown')}: {elem.get('text', elem.get('placeholder', 'N/A'))}" for elem in application_data.get('elements', [])[:10]])}

# Forms:
# {chr(10).join([f"- Form with {len(form.get('inputs', []))} inputs" for form in application_data.get('forms', [])[:3]])}
# """
        
        prompt = ChatPromptTemplate.from_template("""
        You are a BDD test automation expert. Generate BDD scenarios and step definitions
        for the following Jira ticket.
        
        Instructions
        1. Decide file actions
            For each Jira ticket, decide whether to update existing files or create new files.
            Use the following markers for programmatic extraction:
                <<FEATURE_FILE_ACTION>>[TRUE for Update | FALSE for CreateNew]<<FEATURE_FILE_ACTION_END>>
                <<FEATURE_FILE_NAME>>[file path to update or the feature file]<<FEATURE_FILE_NAME_END>>
                <<STEPS_FILE_ACTION>>[TRUE for Update | FALSE for CreateNew]<<STEPS_FILE_ACTION_END>>
                <<STEPS_FILE_NAME>>[file path to update or write the steps file]<<STEPS_FILE_NAME_END>>

        2. Generate Output:
            A Gherkin feature file with comprehensive scenarios
            TypeScript step definitions using WebDriverIO                                                  

        3. Follow these guidelines:
            - Use existing step definitions where possible
            - Follow the coding patterns from the existing codebase
            - Include both happy path and edge cases
            - Use clear, business-readable language
            - If application data is provided, use specific element selectors and navigation flows
            - Include realistic test data based on the actual application elements found
            - Ensure the feature file and step definitions are well-structured and maintainable
            - Add a tag with the key from the Jira Ticket Context and @AutoGenerated to the feature file
            - Do NOT use Markdown code blocks (```); only use the given markers
            - Use the exact markers <<FEATURE_START>> ... <<FEATURE_END>> and <<STEPS_START>> ... <<STEPS_END>>.

        4. Derive file paths / names from code_context
            - Look at existing feature / step files & folder names in code_context
            - Suggest a feature file name/path appropriate to domain / module from that context
            - If updating, use existing file paths; if creating, use consistent folder structure in code_context

        5. Format your response as:

            FILE_DECISIONS:
            <<FEATURE_FILE_ACTION>> true/false <<FEATURE_FILE_ACTION_END>>
            <<FEATURE_FILE_NAME>> path/to/feature/file.feature <<FEATURE_FILE_NAME_END>>
            <<STEPS_FILE_ACTION>> true/false <<STEPS_FILE_ACTION_END>>
            <<STEPS_FILE_NAME>> path/to/steps/file.steps.ts <<STEPS_FILE_NAME_END>>                                           

            FEATURE_FILE:
            <<FEATURE_START>>                                          
            [Gherkin content here]
            <<FEATURE_END>>                                          
            
            STEP_DEFINITIONS:
            <<STEPS_START>>                                          
            [TypeScript code here]
            <<STEPS_END>>

        ### Few-Shot Examples

        **Example 1**

        FILE_DECISIONS:
        <<FEATURE_FILE_ACTION>>TRUE<<FEATURE_FILE_ACTION_END>>
        <<FEATURE_FILE_NAME>>features/login.feature<<FEATURE_FILE_NAME_END>>
        <<STEPS_FILE_ACTION>>TRUE<<STEPS_FILE_ACTION_END>>
        <<STEPS_FILE_NAME>>features/steps/login.steps.ts<<STEPS_FILE_NAME_END>>                                          

        FEATURE_FILE:
        <<FEATURE_START>>
        @XYZ-456 @AutoGenerated                                          
        Feature: User login
        Scenario: Successful login
            Given user is on login page
            When user enters valid credentials
            Then user is redirected to homepage
        <<FEATURE_END>>

        STEP_DEFINITIONS:
        <<STEPS_START>>
        import {{ Given, When, Then }} from '@wdio/cucumber-framework';
        Given('user is on login page', async () => {{
        await browser.url('/login');
        }});
        When('user enters valid credentials', async () => {{
        await $('#username').setValue('test');
        await $('#password').setValue('pass');
        await $('button=Login').click();
        }});
        Then('user is redirected to homepage', async () => {{
        await expect(browser).toHaveUrl('/home');
        }});
        <<STEPS_END>>

        **Example 2**

        FILE_DECISIONS:
        <<FEATURE_FILE_ACTION>>FALSE<<FEATURE_FILE_ACTION_END>>
        <<FEATURE_FILE_NAME>>src/features/search.feature<<FEATURE_FILE_NAME_END>>
        <<STEPS_FILE_ACTION>>FALSE<<STEPS_FILE_ACTION_END>>
        <<STEPS_FILE_NAME>>src/step-definitions/search.steps.ts<<STEPS_FILE_NAME_END>>

        FEATURE_FILE:
        <<FEATURE_START>>
        @ABC-123 @AutoGenerated
        Feature: Search product
        Scenario: Find existing item
            Given user is on homepage
            When user searches for "laptop"
            Then search results with laptops should appear
        <<FEATURE_END>>

        STEP_DEFINITIONS:
        <<STEPS_START>>
        import {{ Given, When, Then }} from '@wdio/cucumber-framework';
        Given('user is on homepage', async () => {{
        await browser.url('/');
        }});
        When('user searches for {{string}}', async (term) => {{
        await $('#search').setValue(term);
        await $('button=Search').click();
        }});
        Then('search results with laptops should appear', async () => {{
        await expect($('.result-item')).toBeExisting();
        }});
        <<STEPS_END>>                                          
                                                                                                                     
        Jira Ticket Context:
        Key: {key}
        Summary: {summary}
        Description: {description}
        Acceptance Criteria: {acceptance_criteria}
        
        Existing Code Context:
        {code_context}
        
        Application Data Collected:                                          
        {app_context}                                          
        """)
        
        logger.debug(f"Formatted prompt for BDD generation: {prompt.format_messages(**ticket, code_context=code_context, app_context=application_data)}")

        response = self.llm.invoke(
            prompt.format_messages(
                **ticket,
                code_context=code_context,
                app_context=application_data
            )
        )
        
        logger.info(f'BDD Generation LLM Response: {response.content[:10]}...')

        # Parse the response
        content = response.content
        logger.debug(f'Full LLM Response Content: {content}')

        # Extract feature and step definitions using regex
        feature_match = re.search(r"<<FEATURE_START>>(.*?)<<FEATURE_END>>", content, re.S)
        steps_match = re.search(r"<<STEPS_START>>(.*?)<<STEPS_END>>", content, re.S)

        # Extract File Decisions markers
        feature_action_match = re.search(r"<<FEATURE_FILE_ACTION>>(.*?)<<FEATURE_FILE_ACTION_END>>", content, re.S)
        feature_filename_match = re.search(r"<<FEATURE_FILE_NAME>>(.*?)<<FEATURE_FILE_NAME_END>>", content, re.S)
        steps_action_match = re.search(r"<<STEPS_FILE_ACTION>>(.*?)<<STEPS_FILE_ACTION_END>>", content, re.S)
        steps_filename_match = re.search(r"<<STEPS_FILE_NAME>>(.*?)<<STEPS_FILE_NAME_END>>", content, re.S)


        logger.info(f'Feature Match: {feature_match is not None}, Steps Match: {steps_match is not None}')
        logger.info(f'Feature File Action Match: {feature_action_match is not None}, Feature File Name Match: {feature_filename_match is not None}')
        logger.info(f'Steps File Action Match: {steps_action_match is not None}, Steps File Name Match: {steps_filename_match is not None}')


        feature_content = ""
        step_def_content = ""
        feature_file_action = None  # Will be "TRUE" or "FALSE"
        feature_file_name = None
        steps_file_action = None    # Will be "TRUE" or "FALSE"
        steps_file_name = None

        if feature_match and steps_match:
            feature_content = feature_match.group(1).strip()
            step_def_content = steps_match.group(1).strip()
            feature_file_action = feature_action_match.group(1).strip().upper() == 'TRUE'
            feature_file_name = feature_filename_match.group(1).strip()
            steps_file_action = steps_action_match.group(1).strip().upper() == 'TRUE'
            steps_file_name = steps_filename_match.group(1).strip()


            logger.info(f'Extracted Feature Content: {feature_content[:10]}...')
            logger.info(f'Extracted Step Definitions Content: {step_def_content[:10]}...')
            logger.info(f'Feature File Action: {feature_file_action}, Feature File Name: {feature_file_name}')
            logger.info(f'Steps File Action: {steps_file_action}, Steps File Name: {steps_file_name}')
        else:
            logger.error("Failed to extract feature or step definition content")

        return {
            'update_feature_file': feature_file_action,
            'feature_file_name': feature_file_name,
            'feature': feature_content,
            'update_steps_file': steps_file_action,
            'steps_file_name': steps_file_name,
            'step_definitions': step_def_content,
            'ticket_key': ticket['key']
        }
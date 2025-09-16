from playwright.sync_api import sync_playwright, Page, Browser
from typing import Dict, List, Optional, Any
import re
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from src.agents.bdd_generator_agent import BDDGeneratorAgent
from src.config.logging import get_logger

logger = get_logger(__name__)

class ApplicationTools:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
    
    def _extract_url_from_jira_data(self, jira_data: Dict) -> Optional[str]:
        """Extract application URL from Jira ticket data"""
        text_content = f"{jira_data.get('summary', '')} {jira_data.get('description', '')} {jira_data.get('acceptance_criteria', '')}"
        
        logger.debug(f'Extracting URL from text content: {text_content}...')  

        # Look for URLs in the text
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+(?<!/)',  # URLs starting with http(s)
            r'www\.[^\s<>"{}|\\^`\[\]]+(?<!/)',      # URLs starting with www
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"{}|\\^`\[\]/]+)?(?<!/)'  # Domain names with optional path
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                if not match.startswith(('http://', 'https://')):
                    match = 'https://' + match
                # Basic validation - check if it looks like a web URL
                if '.' in match and any(domain in match.lower() for domain in ['.com', '.org', '.net', '.io', '.app', 'localhost']):
                    logger.info(f"Extracted URL from Jira data: {match}")
                    return match
        
        logger.info("No application URL found in Jira data")
        return None
    
    def _extract_navigation_instructions(self, jira_data: Dict) -> List[str]:
        """Extract navigation instructions from Jira ticket data"""
        instructions = []
        text_content = f"{jira_data.get('summary', '')} {jira_data.get('description', '')} {jira_data.get('acceptance_criteria', '')}"
        
        # Look for navigation-related keywords and instructions
        navigation_keywords = [
            'navigate to', 'go to', 'visit', 'open', 'click on', 'select',
            'login', 'sign in', 'enter', 'fill', 'submit', 'search for',
            'browse to', 'access', 'view', 'check', 'verify'
        ]
        
        sentences = re.split(r'[.!?]+', text_content)
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in navigation_keywords):
                instructions.append(sentence)
        
        logger.info(f"Extracted {len(instructions)} navigation instructions")
        return instructions
    
    def needs_navigation(self, jira_data: Dict) -> bool:
        """Check if navigation is needed based on Jira data"""
        url = self._extract_url_from_jira_data(jira_data)
        instructions = self._extract_navigation_instructions(jira_data)
        
        needs_nav = bool(url and instructions)
        logger.info(f"Navigation needed: {needs_nav} (URL: {bool(url)}, Instructions: {len(instructions)})")
        return needs_nav
    
    def start_browser(self):
        """Start Playwright browser"""
        if not self.playwright:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()
            logger.info("Browser started successfully")
    
    def close_browser(self):
        """Close Playwright browser"""
        if self.browser:
            self.browser.close()
            self.browser = None
            self.page = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
        logger.info("Browser closed")
    
    async def navigate_and_collect_data_using_mcp(self, jira_data: Dict, bdd_agent: BDDGeneratorAgent) -> str:
        from pydantic import BaseModel, Field, create_model
        from typing import Any, Dict as DictType, List as ListType, Optional, Union
        from langchain_core.tools import StructuredTool

        def convert_json_schema_to_pydantic(schema: DictType[str, Any], model_name: str = "DynamicModel") -> BaseModel:
            """Convert JSON schema to Pydantic BaseModel"""
            try:
                if not isinstance(schema, dict) or schema.get('type') != 'object':
                    # Create a simple model with no fields for non-object schemas
                    return create_model(model_name, __base__=BaseModel)
                
                properties = schema.get('properties', {})
                required_fields = set(schema.get('required', []))
                
                fields = {}
                for field_name, field_def in properties.items():
                    field_type = Any  # Default type
                    default_value = ...  # Required by default
                    
                    # Convert JSON schema types to Python types
                    if field_def.get('type') == 'string':
                        field_type = str
                    elif field_def.get('type') == 'number':
                        field_type = Union[int, float]
                    elif field_def.get('type') == 'integer':
                        field_type = int
                    elif field_def.get('type') == 'boolean':
                        field_type = bool
                    elif field_def.get('type') == 'array':
                        field_type = ListType[Any]
                    elif field_def.get('type') == 'object':
                        field_type = DictType[str, Any]
                    
                    # Handle enum fields
                    if 'enum' in field_def:
                        # For enum, keep as string but could add validation
                        field_type = str
                    
                    # Set default if not required
                    if field_name not in required_fields:
                        default_value = None
                        field_type = Optional[field_type]
                    
                    # Create field with description
                    field_description = field_def.get('description', '')
                    fields[field_name] = (field_type, Field(default=default_value, description=field_description))
                
                return create_model(model_name, **fields, __base__=BaseModel)
                
            except Exception as e:
                logger.warning(f"Error converting schema to Pydantic for {model_name}: {e}")
                # Return empty model as fallback
                return create_model(model_name, __base__=BaseModel)

        def fix_mcp_tool_for_google_ai(tool: StructuredTool) -> StructuredTool:
            """Fix MCP tool to work with Google Generative AI"""
            try:
                if hasattr(tool, 'args_schema') and isinstance(tool.args_schema, dict):
                    # Convert dict schema to Pydantic model
                    pydantic_model = convert_json_schema_to_pydantic(
                        tool.args_schema, 
                        f"{tool.name.title().replace('_', '')}Args"
                    )
                    
                    # Create new tool with fixed args_schema
                    return StructuredTool(
                        name=tool.name,
                        description=tool.description,
                        func=tool.func,
                        coroutine=tool.coroutine,
                        args_schema=pydantic_model,
                        metadata=getattr(tool, 'metadata', {}),
                        response_format=getattr(tool, 'response_format', 'content')
                    )
                return tool
            except Exception as e:
                logger.warning(f"Error fixing tool {tool.name}: {e}")
                return tool

        # Configure your MCP server (path to your math_server.py)
        server_params = StdioServerParameters(
            command="npx",
            args=["@playwright/mcp"],  # Replace with actual path
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Load the raw MCP tools
                raw_tools = await load_mcp_tools(session)
                logger.info(f'Loaded {len(raw_tools)} MCP tools')

                fixed_tools = []
                for tool in raw_tools:
                    try:
                        fixed_tool = fix_mcp_tool_for_google_ai(tool)
                        fixed_tools.append(fixed_tool)
                        logger.debug(f"Fixed tool: {tool.name}")
                    except Exception as e:
                        logger.warning(f"Failed to fix tool {tool.name}: {e}")
                        continue

                logger.info(f"Successfully fixed {len(fixed_tools)} tools")

                # llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
                agent = create_react_agent(model=bdd_agent.llm, tools=fixed_tools)

                text_content = f"Summary:\n{jira_data.get('summary', '')}\n\nDescription:\n{jira_data.get('description', '')}\n\nAcceptance Criteria:\n{jira_data.get('acceptance_criteria', '')}"
                prompt = f"""
You are a Playwright-MCP browser agent. Execute the following user scenario using the Playwright-MCP tools (e.g., browser_navigate, browser_fill, browser_click, browser_wait_for, browser_screenshot, browser_evaluate, playwright_get_visible_text), and **only output a single, valid JSON object** containing the collected data.

Important:
- **Do execute** the scenario via MCP tools.
- **Do NOT output any descriptive text, code, or comments**.
- **Only** return one JSON object with this exact structure:

{{
"url": "...",
"title": "...",
"tools called": ["..."],
"elements": [
{{"type":"...","text":"...","selector":"..."}},
...
],
"forms": [
{{"action":"...","method":"...","inputs":[{{"type":"...","name":"...","placeholder":"...","required":...}}, ...]}}
],
"navigation_flow": [
"..."
]
}}

### Scenario to execute:
{text_content}

### Capturing Requirements:
- After each significant step (navigation, search input, search results, add to cart, cart update), capture a screenshot via `playwright_screenshot`, and save its identifier in `screenshots`.
- Append a descriptive entry to `navigation_flow` for each action taken, e.g., "Navigated to homepage", "Searched for 'wireless mouse'", "Clicked Add to Cart", etc.
- After completing the flow:
- Extract the **actual current page URL** using an MCP method such as `playwright_evaluate("return window.location.href")`.
- Extract the **page title** via `playwright_evaluate("return document.title")`.
- Extract visible **buttons, links, inputs** into the `elements` array, including type, text (or value), and selector.
- Extract any **forms**, including action, method, and contained inputs with their attributes.
- **DO NOT** include any explanation—just return the filled JSON object.
    """
                
                logger.debug(f'Navigation prompt: {prompt}...')   

                response = await agent.ainvoke({"messages":prompt}) 
                message = response['messages'][-1].content
                logger.debug(f'Navigation response: {message}...')

                return message

    def navigate_and_collect_data(self, jira_data: Dict, existing_bdd_data: Optional[str] = None) -> Dict[str, Any]:
        """Navigate to application and collect data for BDD generation"""
        try:
            self.start_browser()
            
            url = self._extract_url_from_jira_data(jira_data)
            if not url:
                logger.warning("No URL found for navigation")
                return {}
            
            # Navigate to the application
            logger.info(f"Navigating to: {url}")
            self.page.goto(url, wait_until='networkidle')
            
            # Collect basic page information
            page_data = {
                'url': url,
                'title': self.page.title(),
                'screenshots': [],
                'elements': [],
                'forms': [],
                'navigation_flow': []
            }
            
            # Take initial screenshot
            screenshot_path = f"screenshot_{jira_data.get('key', 'unknown')}_initial.png"
            self.page.screenshot(path=screenshot_path)
            page_data['screenshots'].append(screenshot_path)
            
            # Extract navigation instructions
            instructions = self._extract_navigation_instructions(jira_data)
            
            logger.debug(f'Navigation instructions: {instructions}')

            # Execute navigation instructions
            for instruction in instructions[:5]:  # Limit to first 5 instructions
                try:
                    self._execute_navigation_instruction(instruction, page_data)
                except Exception as e:
                    logger.warning(f"Failed to execute instruction '{instruction}': {e}")
                    continue
            
            # Collect page elements and forms
            page_data['elements'] = self._collect_page_elements()
            page_data['forms'] = self._collect_forms()
            
            logger.info(f"Collected data from {len(page_data['elements'])} elements and {len(page_data['forms'])} forms")
            return page_data
            
        except Exception as e:
            logger.error(f"Error during navigation and data collection: {e}")
            return {}
        finally:
            self.close_browser()
    
    def _execute_navigation_instruction(self, instruction: str, page_data: Dict):
        """Execute a single navigation instruction"""
        instruction_lower = instruction.lower()
        
        # Take screenshot before action
        screenshot_path = f"screenshot_{instruction[:20].replace(' ', '_')}.png"
        self.page.screenshot(path=screenshot_path)
        page_data['screenshots'].append(screenshot_path)
        
        # Simple instruction execution based on common patterns
        if 'click' in instruction_lower:
            # Try to find clickable elements
            clickable_selectors = [
                'button', 'a', '[role="button"]', 'input[type="submit"]',
                '[onclick]', '.btn', '.button'
            ]
            
            for selector in clickable_selectors:
                elements = self.page.query_selector_all(selector)
                if elements:
                    # Click the first visible element
                    for element in elements:
                        if element.is_visible():
                            element.click()
                            page_data['navigation_flow'].append(f"Clicked {selector}")
                            self.page.wait_for_timeout(1000)  # Wait for page to load
                            return
        
        elif 'enter' in instruction_lower or 'fill' in instruction_lower:
            # Try to find input fields
            inputs = self.page.query_selector_all('input[type="text"], input[type="email"], input[type="password"], textarea')
            if inputs:
                # Fill the first visible input
                for input_field in inputs:
                    if input_field.is_visible():
                        input_field.fill("test_data")
                        page_data['navigation_flow'].append(f"Filled input field")
                        return
        
        elif 'search' in instruction_lower:
            # Look for search inputs
            search_inputs = self.page.query_selector_all('input[type="search"], input[placeholder*="search" i]')
            if search_inputs:
                for search_input in search_inputs:
                    if search_input.is_visible():
                        search_input.fill("test search")
                        page_data['navigation_flow'].append(f"Performed search")
                        return
        
        # If no specific action found, just wait
        self.page.wait_for_timeout(1000)
        page_data['navigation_flow'].append(f"Executed: {instruction}")
    
    def _collect_page_elements(self) -> List[Dict]:
        """Collect relevant page elements for BDD generation"""
        elements = []
        
        # Collect buttons
        buttons = self.page.query_selector_all('button, input[type="submit"], input[type="button"]')
        for button in buttons:
            if button.is_visible():
                elements.append({
                    'type': 'button',
                    'text': button.inner_text() or button.get_attribute('value') or '',
                    'selector': self._get_element_selector(button)
                })
        
        # Collect links
        links = self.page.query_selector_all('a')
        for link in links:
            if link.is_visible():
                elements.append({
                    'type': 'link',
                    'text': link.inner_text(),
                    'href': link.get_attribute('href'),
                    'selector': self._get_element_selector(link)
                })
        
        # Collect form inputs
        inputs = self.page.query_selector_all('input, textarea, select')
        for input_field in inputs:
            if input_field.is_visible():
                elements.append({
                    'type': 'input',
                    'input_type': input_field.get_attribute('type') or 'text',
                    'placeholder': input_field.get_attribute('placeholder') or '',
                    'name': input_field.get_attribute('name') or '',
                    'selector': self._get_element_selector(input_field)
                })
        
        return elements
    
    def _collect_forms(self) -> List[Dict]:
        """Collect form information"""
        forms = []
        form_elements = self.page.query_selector_all('form')
        
        for form in form_elements:
            form_data = {
                'action': form.get_attribute('action') or '',
                'method': form.get_attribute('method') or 'get',
                'inputs': []
            }
            
            # Collect form inputs
            inputs = form.query_selector_all('input, textarea, select')
            for input_field in inputs:
                form_data['inputs'].append({
                    'type': input_field.get_attribute('type') or 'text',
                    'name': input_field.get_attribute('name') or '',
                    'placeholder': input_field.get_attribute('placeholder') or '',
                    'required': input_field.has_attribute('required')
                })
            
            forms.append(form_data)
        
        return forms
    
    def _get_element_selector(self, element) -> str:
        """Generate a CSS selector for an element"""
        try:
            # Try to get a unique identifier
            element_id = element.get_attribute('id')
            if element_id:
                return f"#{element_id}"
            
            # Try to get a unique class
            classes = element.get_attribute('class')
            if classes:
                class_list = classes.split()
                for cls in class_list:
                    if cls and not cls.startswith('ng-'):
                        return f".{cls}"
            
            # Fallback to tag name
            tag_name = element.evaluate('el => el.tagName.toLowerCase()')
            return tag_name
            
        except Exception:
            return 'unknown'

from playwright.sync_api import sync_playwright, Page, Browser
from typing import Dict, List, Optional, Any
import re
import json
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
        
        # Look for URLs in the text
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[^\s<>"{}|\\^`\[\]]+',
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"{}|\\^`\[\]]*)?'
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

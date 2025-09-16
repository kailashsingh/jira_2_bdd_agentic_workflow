from jira import JIRA
from typing import List, Dict, Optional
from src.config.settings import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

class JiraTools:
    def __init__(self):
        self.jira = JIRA(
            server=settings.jira_url,
            basic_auth=(settings.jira_email, settings.jira_api_token)
        )
    
    def get_sprint_tickets(self, sprint_id: Optional[int] = None) -> List[Dict]:
        """Fetch all tickets from a sprint or active sprint"""

        jql = f"project = \"{settings.jira_project}\" and " if settings.jira_project else ""
        jql += f"sprint = {sprint_id} " if sprint_id else "sprint in openSprints() "
        jql += "and status in ('To Do', 'In Progress')"
        logger.debug(f"Using JQL query: {jql}")
        
        issues = self.jira.search_issues(jql, maxResults=100)
        logger.info(f"Found {len(issues)} tickets in sprint")
        
        tickets = []
        for issue in issues:
            ticket = {
                'key': issue.key,
                'summary': issue.fields.summary,
                'description': issue.fields.description or '',
                'acceptance_criteria': self._extract_acceptance_criteria(issue),
                'issue_type': issue.fields.issuetype.name,
                'status': issue.fields.status.name,
                'components': [comp.name.lower() for comp in issue.fields.components] if hasattr(issue.fields, 'components') else []
            }
            tickets.append(ticket)
            
            # Log each ticket's details
            logger.debug(f"Ticket {ticket['key']}: {ticket['summary']} ({ticket['status']})")
            logger.debug(f"Ticket details for {ticket['key']}:")
            logger.debug(f"  Type: {ticket['issue_type']}")
            logger.debug(f"  Description: {ticket['description'][:100]}...")
            logger.debug(f"  Acceptance Criteria: {ticket['acceptance_criteria'][:100]}...")
            
        return tickets
    
    def _extract_acceptance_criteria(self, issue):
        """Extract acceptance criteria from custom field or description"""
        # This is a placeholder - adjust based on your Jira configuration
        criteria = getattr(issue.fields, 'customfield_10106', '') or ''
        logger.debug(f"Extracted acceptance criteria for {issue.key}")
        return criteria
    
    def update_ticket_comment(self, ticket_key: str, comment: str):
        """Add a comment to a Jira ticket"""
        logger.info(f"Adding comment to ticket {ticket_key}")
        logger.debug(f"Comment content for {ticket_key}: {comment[:100]}...")
        self.jira.add_comment(ticket_key, comment)
        logger.info(f"Successfully added comment to {ticket_key}")
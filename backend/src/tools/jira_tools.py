from jira import JIRA
from typing import List, Dict, Optional, Union
from src.config.settings import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

class JiraTools:
    def __init__(self):
        self.jira = JIRA(
            server=settings.jira_url,
            basic_auth=(settings.jira_email, settings.jira_api_token)
        )
        self.allowed_statuses = set(settings.allowed_ticket_statuses)
    
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
    
    def _process_issue(self, issue) -> Dict:
        """Convert a Jira issue to our internal ticket format"""
        status = issue.fields.status.name
        
        # Check if ticket is in an allowed status
        if status not in self.allowed_statuses:
            raise ValueError(
                f"Ticket {issue.key} is in status '{status}'. "
                f"Only tickets in {', '.join(self.allowed_statuses)} status can be processed."
            )
            
        ticket = {
            'key': issue.key,
            'summary': issue.fields.summary,
            'description': issue.fields.description or '',
            'acceptance_criteria': self._extract_acceptance_criteria(issue),
            'issue_type': issue.fields.issuetype.name,
            'status': status,
            'components': [comp.name.lower() for comp in issue.fields.components] if hasattr(issue.fields, 'components') else []
        }
        return ticket

    def get_tickets(self, ticket_keys: List[str], project: Optional[str] = None) -> List[Dict]:
        """Fetch one or more tickets by their keys"""
            
        project_key = project or settings.jira_project
        jql = f'project = "{project_key}" AND key in ({",".join(ticket_keys)})'
        logger.debug(f"Using JQL query: {jql}")
        
        issues = self.jira.search_issues(jql, maxResults=len(ticket_keys))
        if not issues:
            raise ValueError(f"No tickets found in project {project_key} for keys: {ticket_keys}")
        
        # Check if all requested tickets were found
        found_keys = {issue.key for issue in issues}
        missing_keys = set(ticket_keys) - found_keys
        if missing_keys:
            raise ValueError(f"Some tickets were not found: {missing_keys}")
        
        tickets = []
        for issue in issues:
            try:
                ticket = self._process_issue(issue)
                tickets.append(ticket)
            except ValueError as e:
                logger.warning(str(e))
                # Re-raise with additional context
                raise ValueError(f"Invalid ticket state: {str(e)}")
                
        logger.info(f"Retrieved {len(tickets)} tickets")
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
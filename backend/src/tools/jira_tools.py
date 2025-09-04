from jira import JIRA
from typing import List, Dict, Optional
from src.config.settings import settings

class JiraTools:
    def __init__(self):
        self.jira = JIRA(
            server=settings.jira_url,
            basic_auth=(settings.jira_email, settings.jira_api_token)
        )
    
    def get_sprint_tickets(self, sprint_id: Optional[int] = None) -> List[Dict]:
        """Fetch all tickets from a sprint or active sprint"""
        jql = f"sprint = {sprint_id}" if sprint_id else "sprint in openSprints()"
        issues = self.jira.search_issues(jql, maxResults=100)
        
        tickets = []
        for issue in issues:
            tickets.append({
                'key': issue.key,
                'summary': issue.fields.summary,
                'description': issue.fields.description or '',
                'acceptance_criteria': self._extract_acceptance_criteria(issue),
                'issue_type': issue.fields.issuetype.name,
                'status': issue.fields.status.name
            })
        return tickets
    
    def _extract_acceptance_criteria(self, issue):
        """Extract acceptance criteria from custom field or description"""
        # This is a placeholder - adjust based on your Jira configuration
        return getattr(issue.fields, 'customfield_10001', '') or ''
    
    def update_ticket_comment(self, ticket_key: str, comment: str):
        """Add a comment to a Jira ticket"""
        self.jira.add_comment(ticket_key, comment)
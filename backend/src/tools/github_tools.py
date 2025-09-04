from github import Github
from typing import List, Optional, Dict
import base64
from src.config.settings import settings

class GitHubTools:
    def __init__(self):
        self.github = Github(settings.github_token)
        self.repo = self.github.get_repo(settings.github_repo)
    
    def get_feature_files(self) -> List[Dict]:
        """Fetch all feature files from the repository"""
        features = []
        contents = self.repo.get_contents("features")
        
        for content_file in contents:
            if content_file.path.endswith('.feature'):
                features.append({
                    'path': content_file.path,
                    'content': base64.b64decode(content_file.content).decode('utf-8'),
                    'name': content_file.name
                })
        return features
    
    def get_step_definitions(self) -> List[Dict]:
        """Fetch all step definition files"""
        step_defs = []
        contents = self.repo.get_contents("step-definitions")
        
        for content_file in contents:
            if content_file.path.endswith('.ts'):
                step_defs.append({
                    'path': content_file.path,
                    'content': base64.b64decode(content_file.content).decode('utf-8'),
                    'name': content_file.name
                })
        return step_defs
    
    def create_branch(self, branch_name: str) -> str:
        """Create a new branch from main"""
        main_branch = self.repo.get_branch("main")
        ref = f"refs/heads/{branch_name}"
        self.repo.create_git_ref(ref=ref, sha=main_branch.commit.sha)
        return branch_name
    
    def create_or_update_file(self, file_path: str, content: str, 
                              branch: str, message: str):
        """Create or update a file in the repository"""
        try:
            # Try to get existing file
            file = self.repo.get_contents(file_path, ref=branch)
            self.repo.update_file(
                file_path, message, content, file.sha, branch=branch
            )
        except:
            # File doesn't exist, create it
            self.repo.create_file(
                file_path, message, content, branch=branch
            )
    
    def create_pull_request(self, branch: str, title: str, body: str):
        """Create a pull request"""
        pr = self.repo.create_pull(
            title=title,
            body=body,
            head=branch,
            base="main"
        )
        return pr.html_url
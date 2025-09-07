from github import Github
from github.GithubException import GithubException
from typing import List, Optional, Dict
import base64
from src.config.settings import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

class GitHubTools:
    def __init__(self):
        self.github = Github(settings.github_token)
        self.repo = self.github.get_repo(settings.github_repo)
    
    def get_feature_files(self) -> List[Dict]:
        """Fetch all feature files from the repository"""

        logger.info(f"Fetching feature files from {settings.github_repo}")
        features = []
        contents = self.repo.get_contents("src/features")
        
        for content_file in contents:
            if content_file.path.endswith('.feature'):
                feature = {
                    'path': content_file.path,
                    'content': base64.b64decode(content_file.content).decode('utf-8'),
                    'name': content_file.name
                }
                logger.debug(f'Feature file: {feature}')
                features.append(feature)

        return features
    
    def get_step_definitions(self) -> List[Dict]:
        """Fetch all step definition files"""

        logger.info(f"Fetching step definitions from {settings.github_repo}")
        step_defs = []
        contents = self.repo.get_contents("src/step-definitions")
        
        for content_file in contents:
            if content_file.path.endswith('.ts'):
                step_def = {
                    'path': content_file.path,
                    'content': base64.b64decode(content_file.content).decode('utf-8'),
                    'name': content_file.name
                }
                logger.debug(f'Step definition: {step_def}')
                step_defs.append(step_def)
                
        return step_defs
    
    def create_branch(self, branch_name: str) -> str:
        """Create a new branch from the repo's default branch.

        If the branch already exists, return it without error.
        """
        default_branch_name = self.repo.default_branch
        base_branch = self.repo.get_branch(default_branch_name)
        ref = f"refs/heads/{branch_name}"

        # If branch already exists, just return
        try:
            self.repo.get_git_ref(f"heads/{branch_name}")
            logger.info(f"Branch already exists: {branch_name}")
            return branch_name
        except GithubException as e:
            if e.status != 404:
                logger.error(f"Failed checking for existing branch {branch_name}: {e}")
                raise

        try:
            self.repo.create_git_ref(ref=ref, sha=base_branch.commit.sha)
            logger.info(f"Created branch '{branch_name}' from '{default_branch_name}'")
            return branch_name
        except GithubException as e:
            # 422 often indicates the ref exists or update failed due to race; re-check and return if present
            if e.status == 422:
                try:
                    self.repo.get_git_ref(f"heads/{branch_name}")
                    logger.info(f"Branch appeared concurrently: {branch_name}")
                    return branch_name
                except GithubException:
                    pass
            logger.error(f"Failed creating branch {branch_name}: {e}")
            raise
    
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
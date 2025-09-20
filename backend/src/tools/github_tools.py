from github import Github
from github.GithubException import GithubException
from typing import List, Optional, Dict
import base64
from src.config.settings import settings
from src.config.logging import get_logger
from functools import lru_cache
import time
from github.Repository import Repository

logger = get_logger(__name__)

class GitHubTools:
    def __init__(self):
        self.github = Github(settings.github_token)
        self._repos_cache = {}
        self._cache_timestamps = {}
        self.current_repo_name = None
        self.cache_ttl = settings.cache_ttl

    def set_repository(self, repo_name: str) -> None:
        """Set the current repository based on repo name"""
        if self.current_repo_name != repo_name:
            logger.info(f"Switching to repository: {repo_name}")
            self.current_repo_name = settings.github_repo_owner + "/" + repo_name

    def _get_repo(self) -> Repository:
        """Get the current repository with caching"""
        if not self.current_repo_name:
            raise ValueError("Repository name not set. Call set_repository() first.")

        current_time = time.time()
        
        # Check if cache is enabled and if we have a cached repo
        if settings.cache_enabled and self.current_repo_name in self._repos_cache:
            cache_time = self._cache_timestamps.get(self.current_repo_name, 0)
            
            # Check if cache is still valid
            if current_time - cache_time < self.cache_ttl:
                logger.debug(f"Using cached repository: {self.current_repo_name}")
                return self._repos_cache[self.current_repo_name]
        
        # Cache miss or disabled - fetch from GitHub
        logger.info(f"Fetching repository from GitHub: {self.current_repo_name}")
        repo = self.github.get_repo(self.current_repo_name)
        
        # Update cache if enabled
        if settings.cache_enabled:
            self._repos_cache[self.current_repo_name] = repo
            self._cache_timestamps[self.current_repo_name] = current_time
        
        return repo
    
    def get_feature_files(self) -> List[Dict]:
        """Fetch all feature files from the repository"""

        logger.info(f"Fetching feature files from {self.current_repo_name}")
        features = []
        contents = self._get_repo().get_contents("src/features")
        
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

        logger.info(f"Fetching step definitions from {self.current_repo_name}")
        step_defs = []
        contents = self._get_repo().get_contents("src/step-definitions")
        
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
    
    def create_branch(self, branch_name: str, force_update: bool = True) -> str:
        """Create a new branch from the repo's default branch.

        Args:
            branch_name: Name of the branch to create
            force_update: If True and branch exists, update it to latest default branch commit.
                         If False and branch exists, return existing branch without changes.
        """
        repo = self._get_repo()
        default_branch_name = repo.default_branch
        base_branch = repo.get_branch(default_branch_name)
        ref = f"refs/heads/{branch_name}"

        # Check if branch already exists
        try:
            existing_ref = self._get_repo().get_git_ref(f"heads/{branch_name}")
            if force_update:
                try:
                    # Try to update the existing branch to point to the latest default branch commit
                    existing_ref.edit(sha=base_branch.commit.sha)
                    logger.info(f"Updated existing branch '{branch_name}' to latest '{default_branch_name}' commit")
                except GithubException as update_error:
                    if update_error.status == 422 and "not a fast forward" in str(update_error):
                        # Can't fast-forward, need to force update
                        logger.warning(f"Cannot fast-forward update branch '{branch_name}'. Force updating...")
                        existing_ref.edit(sha=base_branch.commit.sha, force=True)
                        logger.info(f"Force updated existing branch '{branch_name}' to latest '{default_branch_name}' commit")
                    else:
                        logger.error(f"Failed updating branch {branch_name}: {update_error}")
                        raise
            else:
                # Return existing branch without changes
                logger.info(f"Branch already exists: {branch_name} (no update requested)")
            return branch_name
        except GithubException as e:
            if e.status != 404:
                logger.error(f"Failed checking for existing branch {branch_name}: {e}")
                raise

        # Branch doesn't exist, create it
        try:
            self._get_repo().create_git_ref(ref=ref, sha=base_branch.commit.sha)
            logger.info(f"Created branch '{branch_name}' from '{default_branch_name}'")
            return branch_name
        except GithubException as e:
            # 422 often indicates the ref exists or update failed due to race; re-check and return if present
            if e.status == 422:
                try:
                    self._get_repo().get_git_ref(f"heads/{branch_name}")
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
            repo = self._get_repo()
            file = repo.get_contents(file_path, ref=branch)
            repo.update_file(
                file_path, message, content, file.sha, branch=branch
            )
        except:
            # File doesn't exist, create it
            self._get_repo().create_file(
                file_path, message, content, branch=branch
            )
    
    def create_pull_request(self, branch: str, title: str, body: str):
        """Create a pull request"""
        pr = self._get_repo().create_pull(
            title=title,
            body=body,
            head=branch,
            base="main"
        )
        return pr.html_url
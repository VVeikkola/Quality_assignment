import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OLLAMA_MODEL = "llama2:7b"


class RepoComparisonTool:
    def __init__(self, github_token, output_dir="output"):
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {self.github_token}",
        }

    def get_repo_info(self):
        """Fetch repository information from GitHub API."""
        try:
            response = requests.get(
                "https://github.com/pallets/click", headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching repository info: {e}")
            return None

    def get_forks(self, max_forks=100):
        """Fetch forks of a given repository from GitHub API."""
        forks = []
        print(len(forks))
        page = 1
        while len(forks) < max_forks:
            try:
                response = requests.get(
                    f"https://api.github.com/repos/pallets/click/forks?per_page=100&page={page}",
                    headers=self.headers,
                )
                response.raise_for_status()
                new_forks = response.json()
                if not new_forks:
                    break
                forks.extend(new_forks)
                page += 1
                time.sleep(0.7)  # GitHub API rate limit
            except requests.RequestException as e:
                print(f"Error fetching forks: {e}")
                break
        return forks[:max_forks]

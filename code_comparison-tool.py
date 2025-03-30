from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import json
import os
import random
import subprocess
import time
from timeit import main
from typing import Dict, List, Optional, Any
import requests
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OLLAMA_MODEL = "mitral"


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
        try:
            response = requests.get(
                "https://api.github.com/repos/pallets/click", headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching repository info: {e}")
            return None

    def get_forks(self, max_forks=100):
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
                time.sleep(0.7)
            except requests.RequestException as e:
                print(f"Error fetching forks: {e}")
                break
        return forks[:max_forks]
    
    @lru_cache(maxsize=100)
    def get_repo_contents(self, repo_name, path=""):
        try:
            response = requests.get(
                f"https://api.github.com/repos/{repo_name}/contents/{path}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching contents for {repo_name}/{path}: {e}")
            return None

    def download_file(self, url, file_path):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded {file_path}")
        except requests.RequestException as e:
            print(f"Error downloading file {file_path}: {e}")

    def get_file_content(self, repo_name, file_path):
        try:
            response = requests.get(
                f"https://api.github.com/repos/{repo_name}/contents/{file_path}",
                headers=self.headers
            )
            response.raise_for_status()
            file_data = response.json()
            if file_data.get("content"):
                import base64
                content = base64.b64decode(file_data["content"]).decode('utf-8')
                print(f"Content of {file_path}:\n{content}")
                return content
            elif file_data.get("download_url"):
                response = requests.get(file_data["download_url"])
                print(f"Content of {file_path}:\n{response.text}")
                return response.text
            return None
        except requests.RequestException as e:
            print(f"Error fetching file content {repo_name}/{file_path}: {e}")
            return None

    def compare_repos(self, base: str, fork: Dict) -> Dict:
        comparison = {
            "fork": fork["full_name"],
            "fork_url": fork["html_url"],
            "files": [],
            "summary": {
                "average_similarity": 0,
                "refactoring_distribution": {
                    "none": 0,
                    "low": 0,
                    "medium": 0,
                    "high": 0
                },
                "files_compared": 0
            }
        }

        base_contents = self.get_repo_contents(base)
        fork_contents = self.get_repo_contents(fork["full_name"])

        if not base_contents or not fork_contents:
            return comparison

        base_files = {item['path']: item for item in base_contents if item['type'] == 'file'}
        fork_files = {item['path']: item for item in fork_contents if item['type'] == 'file'}

        common_files = set(base_files.keys()) & set(fork_files.keys())
        
        for file_path in common_files:
            base_content = self.get_file_content(base, file_path)
            fork_content = self.get_file_content(fork["full_name"], file_path)

            if base_content and fork_content:
                comparison_result = self.llm_compare_files(base_content, fork_content)
                comparison["files"].append({
                    "file_path": file_path,
                    "comparison": comparison_result
                })

        if comparison["files"]:
            total_similarity = sum(
                f["comparison"]["similarity_percentage"] 
                for f in comparison["files"]
            ) / len(comparison["files"])
            
            refactoring_levels = [
                f["comparison"]["refactoring_level"] 
                for f in comparison["files"]
            ]
            
            comparison["summary"] = {
                "average_similarity": total_similarity,
                "refactoring_distribution": {
                    "none": refactoring_levels.count("none"),
                    "low": refactoring_levels.count("low"),
                    "medium": refactoring_levels.count("medium"),
                    "high": refactoring_levels.count("high"),
                },
                "files_compared": len(comparison["files"])
            }

        return comparison

    def llm_compare_files(self, original_content, fork_content):
        prompt = f"""
        Compare the following two code files semantically and identify:
        1. Similarity percentage (0-100)
        2. Refactoring level (none, low, medium, high)
        3. Whether features were added or removed
        4. Brief description of changes

        ORIGINAL:
        ```
        {original_content[:10000]}
        ```

        FORK VERSION:
        ```
        {fork_content[:10000]}
        ```

        Return JSON format:
        {{
            "similarity_percentage": int, Fgene
            "refactoring_level": "none|low|medium|high", 
            "added_features": bool, 
            "removed_features": bool, 
            "notes": str,
            "quality_issues": [
                {{
                    "issue_type": str,
                    "severity": "low|medium|high",
                    "description": str,
                    "suggestion": str
                }}
            ]
        }}
        """
        try:
            result = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL, prompt],
                capture_output=True,
                text=True,
                timeout=120 
            )
            output = result.stdout.strip()

            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                output = output[json_start:json_end]
            comparison = json.loads(output)
            
            # DEDAULT VALUES
            comparison.setdefault("similarity_percentage", 0)
            comparison.setdefault("refactoring_level", "unknown")
            comparison.setdefault("added_features", False)
            comparison.setdefault("removed_features", False)
            comparison.setdefault("notes", "No analysis available")
            comparison.setdefault("quality_issues", [])
            
            return comparison
        except Exception as e:
            self.log_error(f"Error running Ollama model: {e}")
            return {
                "similarity_percentage": 0,
                "refactoring_level": "unknown",
                "added_features": False,
                "removed_features": False,
                "notes": "Error in analysis",
                "quality_issues": []
            }
    
    def llm_code_quality_analysis(self, code: str) -> Dict:
        prompt = f"""
        Analyze this code for quality issues that static analysis tools might miss:
        1. Architectural smells
        2. Testability issues
        3. Hidden bugs
        4. Security vulnerabilities
        5. Code smells
        
        Code:
        ```python
        {code[:10000]}
        ```
        
        Return JSON format:
        {{
            "issues": [
                {{
                    "type": str,
                    "severity": "low|medium|high",
                    "description": str,
                    "recommendation": str,
                    "tool_missed": bool
                }}
            ]
        }}
        """
        try:
            result = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL, prompt],
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()
            
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                output = output[json_start:json_end]
            
            return json.loads(output)
        except Exception as e:
            self.log_error(f"Error in quality analysis: {e}")
            return {"issues": []}
        
    def save_comparison(self, comparison, base_filename):   
        csv_path = self.output_dir / f"{base_filename}.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Similarity", "Refactoring", "Changes"])
            for file in comparison["files"]:
                writer.writerow([
                    file["file_path"],
                    file["comparison"]["similarity_percentage"],
                    file["comparison"]["refactoring_level"],
                    file["comparison"]["notes"]
                ])

    def analyze_fork(self, fork):
        comparison = self.compare_repos("pallets/click", fork)
        comparison_file = self.output_dir / f"{fork['full_name'].replace('/', '_')}_comparison.json"
        self.save_comparison(comparison, comparison_file)
        return comparison
    

    def analyze_all_forks(self, repo_name: str, max_forks: int = 5) -> List[Dict]:
        forks = self.get_forks(repo_name, max_forks)
        self.logger.info(f"Found {len(forks)} forks to analyze")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.analyze_fork, forks))
            
        return results
    
    def analyze_apache_project(self, project_name, sample_size=20):
        contents = self.get_repo_contents(project_name)
        if not contents:
            return None
            
        files = [f for f in contents if f['type'] == 'file']
        selected_files = random.sample(files, min(sample_size, len(files)))
        
        analysis_results = []
        for file in selected_files:
            content = self.get_file_content(project_name, file['path'])
            if content:
                analysis = self.llm_code_quality_analysis(content)
                analysis_results.append({
                    "file": file['path'],
                    "analysis": analysis
                })
        
        result_file = self.output_dir / f"apache_{project_name.replace('/', '_')}_analysis.json"
        with open(result_file, "w") as f:
            json.dump(analysis_results, f, indent=4)
            
        return analysis_results

    def main():
        """Main execution function."""
        if not GITHUB_TOKEN:
            print("Error: GitHub token not found in environment variables")
            return
        
        tool = RepoComparisonTool(GITHUB_TOKEN)
        
        repo_info = tool.get_repo_info("pallets/click")
        if repo_info:
            print(f"\nRepository Analysis:")
            print(f"Name: {repo_info['name']}")
            print(f"Description: {repo_info['description']}")
            print(f"URL: {repo_info['html_url']}")
            print(f"Forks: {repo_info['forks_count']}")
            print(f"Stars: {repo_info['stargazers_count']}")
        
        print("\nAnalyzing forks...")
        fork_results = tool.analyze_all_forks("pallets/click", 5)
        
        print("\nAnalyzing Apache project for comparison...")
        apache_results = tool.analyze_apache_project("apache/flink", 5)
        
        print("\nAnalysis complete. Check output directory for results.")

if __name__ == "__main__":
    main()
    
""" if __name__ == "__main__":
    tool = RepoComparisonTool(GITHUB_TOKEN)
    repo_info = tool.get_repo_info()
    if repo_info:
        print(f"Repository name: {repo_info['name']}")
        print(f"Repository description: {repo_info['description']}")
        print(f"Repository URL: {repo_info['html_url']}")
        print(f"Forks count: {repo_info['forks_count']}")
        contents = tool.get_repo_contents("pallets/click")
        print("Length", len(contents), "Contents:", contents)
        if contents:
            for item in contents:
                if item["type"] == "file":
                    file_path = tool.output_dir / item["name"]
                    tool.download_file(item["download_url"], file_path)
                    with open(file_path, "r", encoding="utf-8") as file:
                        print(f"Contents of {file_path}:")
                        print(file.read())
    print(f"Fetched content of pallets/click/{file_path}")

    print("Fetching forks...")
    forks = tool.get_forks(max_forks=5)
    for fork in forks:
        print(f"Fork: {fork['full_name']}")
        comparison = tool.compare_repos("pallets/click", fork)
        comparison_file = tool.output_dir / f"{fork['full_name'].replace('/', '_')}_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=4)
        print(f"Comparison saved to {comparison_file}")
        analysis_results = tool.analyze_forks()
        print(f"Analysis results for {fork['full_name']}:")
        print(json.dumps(analysis_results, indent=4))
 """
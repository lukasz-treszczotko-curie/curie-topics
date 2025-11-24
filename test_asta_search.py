"""
Test script for Allen AI ASTA MCP API
API Documentation: https://allenai.org/asta/resources/mcp
"""

import requests
import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv


class ASTASearchClient:
    """Client for interacting with the ASTA MCP API"""

    def __init__(self, api_key: str):
        self.base_url = "https://asta-tools.allen.ai/mcp/v1"
        self.api_key = api_key
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }

    def search_papers_by_relevance(
        self,
        query: str,
        limit: int = 10,
        fields_of_study: List[str] = None,
        year_min: int = None,
        year_max: int = None
    ) -> Dict[str, Any]:
        """
        Search papers by relevance to a query

        Args:
            query: Search query string
            limit: Maximum number of results to return
            fields_of_study: Filter by fields of study (e.g., ["Computer Science"])
            year_min: Minimum publication year
            year_max: Maximum publication year
        """
        params = {
            "query": query,
            "limit": limit
        }

        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        if year_min:
            params["yearMin"] = year_min
        if year_max:
            params["yearMax"] = year_max

        # MCP API uses JSON-RPC 2.0 format
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "search_papers_by_relevance",
                "arguments": params
            },
            "id": 1
        }

        response = requests.post(self.base_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def search_paper_by_title(self, title: str) -> Dict[str, Any]:
        """Search for a specific paper by title"""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "search_paper_by_title",
                "arguments": {"title": title}
            },
            "id": 1
        }

        response = requests.post(self.base_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def snippet_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for papers and return relevant snippets"""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "snippet_search",
                "arguments": {
                    "query": query,
                    "limit": limit
                }
            },
            "id": 1
        }

        response = requests.post(self.base_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()


def print_paper_results(results: Dict[str, Any]):
    """Pretty print paper search results"""
    if "result" in results:
        content = results["result"].get("content", [])
        for item in content:
            if item.get("type") == "text":
                print(item.get("text", ""))
    elif "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(json.dumps(results, indent=2))


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment variable
    api_key = os.environ.get("ASTA_API_KEY")

    if not api_key:
        print("Error: Please set ASTA_API_KEY environment variable")
        print("Option 1: Create a .env file with: ASTA_API_KEY=your-api-key-here")
        print("Option 2: export ASTA_API_KEY='your-api-key-here'")
        return

    # Initialize client
    client = ASTASearchClient(api_key)

    print("=" * 80)
    print("ASTA MCP API Search Test")
    print("=" * 80)

    # Test 1: Search papers by relevance
    print("\n1. Searching papers by relevance: 'machine learning transformers'")
    print("-" * 80)
    try:
        results = client.search_papers_by_relevance(
            query="machine learning transformers",
            limit=5
        )
        print_paper_results(results)
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Search by title
    print("\n\n2. Searching paper by title: 'Attention Is All You Need'")
    print("-" * 80)
    try:
        results = client.search_paper_by_title("Attention Is All You Need")
        print_paper_results(results)
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Snippet search
    print("\n\n3. Snippet search: 'large language models'")
    print("-" * 80)
    try:
        results = client.snippet_search(
            query="large language models",
            limit=3
        )
        print_paper_results(results)
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
